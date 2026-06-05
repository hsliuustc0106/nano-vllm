use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures_core::stream::Stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(
    name = "nanovllm-serve",
    about = "Run the Nano-vLLM Rust HTTP frontend."
)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value = "tcp://127.0.0.1:5557")]
    request_endpoint: String,
    #[arg(long, default_value = "tcp://127.0.0.1:5558")]
    event_endpoint: String,
}

#[derive(Clone)]
struct AppState {
    model: String,
    request_tx: std_mpsc::Sender<Vec<u8>>,
    pending: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<EngineEvent>>>>,
    _event_receiver: Arc<EventReceiverHandle>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EngineCompletionRequest {
    #[serde(rename = "type")]
    message_type: String,
    request_id: String,
    model: String,
    prompt: Value,
    max_tokens: u64,
    temperature: f64,
    stream: bool,
    n: u64,
    stop: Vec<String>,
    echo: bool,
    ignore_eos: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct EngineCancelRequest {
    #[serde(rename = "type")]
    message_type: String,
    request_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct EngineControlRequest {
    #[serde(rename = "type")]
    message_type: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
enum EngineEvent {
    #[serde(rename = "started")]
    Started {
        request_id: String,
        created: u64,
        model: String,
        num_choices: usize,
        prompt_tokens: u64,
    },
    #[serde(rename = "token")]
    Token {
        request_id: String,
        choice_index: usize,
        text: String,
        #[serde(rename = "token_id")]
        _token_id: Option<i64>,
    },
    #[serde(rename = "finished")]
    Finished {
        request_id: String,
        choice_index: usize,
        text: String,
        #[serde(rename = "token_ids")]
        _token_ids: Vec<i64>,
        finish_reason: String,
        #[serde(rename = "prompt_tokens")]
        _prompt_tokens: u64,
        completion_tokens: u64,
    },
    #[serde(rename = "error")]
    Error {
        request_id: Option<String>,
        message: String,
    },
}

impl EngineEvent {
    fn request_id(&self) -> Option<&str> {
        match self {
            EngineEvent::Started { request_id, .. }
            | EngineEvent::Token { request_id, .. }
            | EngineEvent::Finished { request_id, .. } => Some(request_id),
            EngineEvent::Error { request_id, .. } => request_id.as_deref(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct CompletionPayload {
    model: Option<String>,
    prompt: Option<Value>,
    max_tokens: Option<Value>,
    temperature: Option<Value>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    n: Option<Value>,
    stop: Option<Value>,
    echo: Option<bool>,
    ignore_eos: Option<bool>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug)]
struct ValidatedCompletion {
    model: String,
    prompt: Value,
    max_tokens: u64,
    temperature: f64,
    stream: bool,
    stream_include_usage: bool,
    n: u64,
    stop: Vec<String>,
    echo: bool,
    ignore_eos: bool,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorMessage,
}

#[derive(Debug, Serialize)]
struct ErrorMessage {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
}

#[derive(Default)]
struct BlockingAssembly {
    request_id: String,
    model: String,
    created: u64,
    expected: usize,
    prompt_tokens: u64,
    completion_tokens: u64,
    choices: BTreeMap<usize, CompletionChoice>,
}

#[derive(Clone, Debug, Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    logprobs: Option<Value>,
    finish_reason: String,
}

struct PendingGuard {
    request_id: String,
    pending: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<EngineEvent>>>>,
    request_tx: std_mpsc::Sender<Vec<u8>>,
    _event_receiver: Arc<EventReceiverHandle>,
    cancel_on_drop: bool,
}

struct EventReceiverHandle {
    endpoint: String,
    shutdown: Arc<AtomicBool>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for EventReceiverHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        wake_event_receiver(&self.endpoint);
        if let Some(join) = self.join.lock().unwrap().take() {
            let _ = join.join();
        }
    }
}

impl PendingGuard {
    fn complete(&mut self) {
        self.cancel_on_drop = false;
        self.pending.lock().unwrap().remove(&self.request_id);
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.pending.lock().unwrap().remove(&self.request_id);
        if self.cancel_on_drop {
            let cancel = EngineCancelRequest {
                message_type: "cancel".to_string(),
                request_id: self.request_id.clone(),
            };
            if let Ok(bytes) = rmp_serde::to_vec_named(&cancel) {
                let _ = self.request_tx.send(bytes);
            }
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let pending = Arc::new(Mutex::new(HashMap::new()));
    let request_tx = spawn_request_sender(args.request_endpoint.clone());
    let event_receiver = spawn_event_receiver(args.event_endpoint.clone(), pending.clone());

    let state = AppState {
        model: args.model,
        request_tx,
        pending,
        _event_receiver: event_receiver,
    };
    let app = build_router(state);
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("invalid host or port");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind HTTP listener");
    tracing::info!("listening on http://{}", addr);
    axum::serve(listener, app)
        .await
        .expect("HTTP server failed");
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/completions", post(create_completion))
        .route("/v1/models", get(list_models))
        .route("/_debug/profile/{action}", post(profile_control))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

fn spawn_request_sender(endpoint: String) -> std_mpsc::Sender<Vec<u8>> {
    let (tx, rx) = std_mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let context = zmq::Context::new();
        let socket = context
            .socket(zmq::PUSH)
            .expect("failed to create ZMQ PUSH socket");
        socket
            .connect(&endpoint)
            .expect("failed to connect request endpoint");
        while let Ok(bytes) = rx.recv() {
            if let Err(err) = socket.send(bytes, 0) {
                tracing::error!("failed to send engine request: {err}");
            }
        }
    });
    tx
}

fn spawn_event_receiver(
    endpoint: String,
    pending: Arc<Mutex<HashMap<String, mpsc::UnboundedSender<EngineEvent>>>>,
) -> Arc<EventReceiverHandle> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let thread_shutdown = shutdown.clone();
    let thread_endpoint = endpoint.clone();
    let join = std::thread::spawn(move || {
        let context = zmq::Context::new();
        let socket = context
            .socket(zmq::PULL)
            .expect("failed to create ZMQ PULL socket");
        socket
            .bind(&thread_endpoint)
            .expect("failed to bind event endpoint");
        socket
            .set_rcvtimeo(100)
            .expect("failed to set event receive timeout");
        while !thread_shutdown.load(Ordering::SeqCst) {
            match socket.recv_bytes(0) {
                Ok(bytes) => match rmp_serde::from_slice::<EngineEvent>(&bytes) {
                    Ok(event) => {
                        if let Some(request_id) = event.request_id() {
                            if let Some(tx) = pending.lock().unwrap().get(request_id) {
                                let _ = tx.send(event);
                            }
                        }
                    }
                    Err(err) => tracing::error!("failed to decode engine event: {err}"),
                },
                Err(zmq::Error::EAGAIN) => {}
                Err(err) => tracing::error!("failed to receive engine event: {err}"),
            }
        }
    });
    Arc::new(EventReceiverHandle {
        endpoint,
        shutdown,
        join: Mutex::new(Some(join)),
    })
}

fn wake_event_receiver(endpoint: &str) {
    let context = zmq::Context::new();
    let Ok(socket) = context.socket(zmq::PUSH) else {
        return;
    };
    socket.set_linger(0).ok();
    if socket.connect(endpoint).is_err() {
        return;
    }
    let event = json!({
        "type": "error",
        "request_id": null,
        "message": "shutdown"
    });
    if let Ok(bytes) = rmp_serde::to_vec_named(&event) {
        let _ = socket.send(bytes, 0);
    }
}

async fn create_completion(
    State(state): State<AppState>,
    Json(payload): Json<CompletionPayload>,
) -> Response {
    let request = match validate_completion_payload(payload) {
        Ok(request) => request,
        Err(message) => return bad_request(message).into_response(),
    };
    let request_id = Uuid::new_v4().simple().to_string();
    let (tx, rx) = mpsc::unbounded_channel::<EngineEvent>();
    state.pending.lock().unwrap().insert(request_id.clone(), tx);
    let mut guard = PendingGuard {
        request_id: request_id.clone(),
        pending: state.pending.clone(),
        request_tx: state.request_tx.clone(),
        _event_receiver: state._event_receiver.clone(),
        cancel_on_drop: true,
    };
    let engine_request = EngineCompletionRequest {
        message_type: "completion".to_string(),
        request_id: request_id.clone(),
        model: request.model.clone(),
        prompt: request.prompt.clone(),
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        stream: request.stream,
        n: request.n,
        stop: request.stop.clone(),
        echo: request.echo,
        ignore_eos: request.ignore_eos,
    };
    let bytes = match rmp_serde::to_vec_named(&engine_request) {
        Ok(bytes) => bytes,
        Err(err) => {
            guard.complete();
            return internal_error(format!("failed to encode engine request: {err}"))
                .into_response();
        }
    };
    if let Err(err) = state.request_tx.send(bytes) {
        guard.complete();
        return internal_error(format!("failed to send engine request: {err}")).into_response();
    }
    if request.stream {
        Sse::new(stream_completion(
            request_id,
            request.model,
            request.stream_include_usage,
            rx,
            guard,
        ))
        .into_response()
    } else {
        match collect_completion(request_id, request.model, rx, &mut guard).await {
            Ok(response) => Json(response).into_response(),
            Err((status, body)) => (status, Json(body)).into_response(),
        }
    }
}

async fn collect_completion(
    request_id: String,
    fallback_model: String,
    mut rx: mpsc::UnboundedReceiver<EngineEvent>,
    guard: &mut PendingGuard,
) -> Result<Value, (StatusCode, ErrorBody)> {
    let mut assembly = BlockingAssembly {
        request_id,
        model: fallback_model,
        created: unix_timestamp(),
        expected: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        choices: BTreeMap::new(),
    };
    loop {
        let event = match tokio::time::timeout(Duration::from_secs(600), rx.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => {
                return Err(error_body(
                    StatusCode::BAD_GATEWAY,
                    "engine event channel closed",
                ))
            }
            Err(_) => {
                return Err(error_body(
                    StatusCode::GATEWAY_TIMEOUT,
                    "timed out waiting for engine response",
                ))
            }
        };
        match event {
            EngineEvent::Started {
                created,
                model,
                num_choices,
                prompt_tokens,
                ..
            } => {
                assembly.created = created;
                assembly.model = model;
                assembly.expected = num_choices;
                assembly.prompt_tokens = prompt_tokens;
            }
            EngineEvent::Token { .. } => {}
            EngineEvent::Finished {
                choice_index,
                text,
                finish_reason,
                completion_tokens,
                ..
            } => {
                assembly.completion_tokens += completion_tokens;
                assembly.choices.insert(
                    choice_index,
                    CompletionChoice {
                        text,
                        index: choice_index,
                        logprobs: None,
                        finish_reason,
                    },
                );
                if assembly.expected > 0 && assembly.choices.len() >= assembly.expected {
                    guard.complete();
                    return Ok(completion_response(assembly));
                }
            }
            EngineEvent::Error { message, .. } => {
                return Err(error_body(StatusCode::BAD_REQUEST, &message))
            }
        }
    }
}

fn stream_completion(
    request_id: String,
    fallback_model: String,
    include_usage: bool,
    mut rx: mpsc::UnboundedReceiver<EngineEvent>,
    mut guard: PendingGuard,
) -> impl Stream<Item = Result<Event, Infallible>> {
    stream! {
        let mut created = unix_timestamp();
        let mut model = fallback_model;
        let mut expected = 0usize;
        let mut prompt_tokens = 0u64;
        let mut completion_tokens = 0u64;
        let mut finished = HashSet::new();
        while let Some(event) = rx.recv().await {
            match event {
                EngineEvent::Started { created: event_created, model: event_model, num_choices, prompt_tokens: event_prompt_tokens, .. } => {
                    created = event_created;
                    model = event_model;
                    expected = num_choices;
                    prompt_tokens = event_prompt_tokens;
                }
                EngineEvent::Token { choice_index, text, .. } => {
                    let chunk = stream_chunk(&request_id, &model, created, choice_index, text, None);
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                EngineEvent::Finished { choice_index, finish_reason, completion_tokens: event_completion_tokens, .. } => {
                    finished.insert(choice_index);
                    completion_tokens += event_completion_tokens;
                    let chunk = stream_chunk(&request_id, &model, created, choice_index, String::new(), Some(finish_reason));
                    yield Ok(Event::default().data(chunk.to_string()));
                    if expected > 0 && finished.len() >= expected {
                        guard.complete();
                        if include_usage {
                            let chunk = stream_usage_chunk(&request_id, &model, created, prompt_tokens, completion_tokens);
                            yield Ok(Event::default().data(chunk.to_string()));
                        }
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
                EngineEvent::Error { message, .. } => {
                    let body = json!({"error": {"message": message, "type": "invalid_request_error", "code": "bad_request"}});
                    yield Ok(Event::default().data(body.to_string()));
                    guard.complete();
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    }
}

async fn list_models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model,
            "object": "model",
            "created": 0,
            "owned_by": "nanovllm"
        }]
    }))
}

async fn profile_control(State(state): State<AppState>, Path(action): Path<String>) -> Response {
    let message_type = match action.as_str() {
        "start" => "profile_start",
        "stop" => "profile_stop",
        _ => return bad_request(format!("unknown profile action: {action}")).into_response(),
    };
    let request = EngineControlRequest {
        message_type: message_type.to_string(),
    };
    let bytes = match rmp_serde::to_vec_named(&request) {
        Ok(bytes) => bytes,
        Err(err) => {
            return internal_error(format!("failed to encode profile control: {err}"))
                .into_response()
        }
    };
    if let Err(err) = state.request_tx.send(bytes) {
        return internal_error(format!("failed to send profile control: {err}")).into_response();
    }
    Json(json!({"ok": true, "action": action})).into_response()
}

fn completion_response(assembly: BlockingAssembly) -> Value {
    let choices: Vec<CompletionChoice> = assembly.choices.into_values().collect();
    json!({
        "id": format!("cmpl-{}", assembly.request_id),
        "object": "text_completion",
        "created": assembly.created,
        "model": assembly.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": assembly.prompt_tokens,
            "completion_tokens": assembly.completion_tokens,
            "total_tokens": assembly.prompt_tokens + assembly.completion_tokens
        }
    })
}

fn stream_chunk(
    request_id: &str,
    model: &str,
    created: u64,
    choice_index: usize,
    text: String,
    finish_reason: Option<String>,
) -> Value {
    json!({
        "id": format!("cmpl-{request_id}"),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": choice_index,
            "logprobs": null,
            "finish_reason": finish_reason
        }]
    })
}

fn stream_usage_chunk(
    request_id: &str,
    model: &str,
    created: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> Value {
    json!({
        "id": format!("cmpl-{request_id}"),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

fn validate_completion_payload(payload: CompletionPayload) -> Result<ValidatedCompletion, String> {
    validate_extra_fields(&payload.extra)?;
    validate_stream_options(payload.stream_options.as_ref())?;
    let model = payload
        .model
        .filter(|model| !model.is_empty())
        .ok_or_else(|| "model must be a non-empty string".to_string())?;
    let prompt = payload
        .prompt
        .ok_or_else(|| "prompt is required".to_string())?;
    validate_prompt_shape(&prompt)?;
    let max_tokens = parse_positive_u64(payload.max_tokens.as_ref(), 16, "max_tokens")?;
    let temperature = parse_positive_f64(payload.temperature.as_ref(), 1.0, "temperature")?;
    let n = parse_positive_u64(payload.n.as_ref(), 1, "n")?;
    let stop = parse_stop(payload.stop.as_ref())?;
    Ok(ValidatedCompletion {
        model,
        prompt,
        max_tokens,
        temperature,
        stream: payload.stream.unwrap_or(false),
        stream_include_usage: payload
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false),
        n,
        stop,
        echo: payload.echo.unwrap_or(false),
        ignore_eos: payload.ignore_eos.unwrap_or(false),
    })
}

fn validate_extra_fields(extra: &BTreeMap<String, Value>) -> Result<(), String> {
    for (name, value) in extra {
        let allowed_default = match name.as_str() {
            "logprobs" | "suffix" => value.is_null(),
            "top_p" | "best_of" => value.is_null() || value.as_f64() == Some(1.0),
            "presence_penalty" | "frequency_penalty" => {
                value.is_null() || value.as_f64() == Some(0.0)
            }
            "repetition_penalty" => value.is_null() || value.as_f64() == Some(1.0),
            "logit_bias" => {
                value.is_null() || value.as_object().is_some_and(|object| object.is_empty())
            }
            _ => false,
        };
        if !allowed_default {
            return Err(format!("unsupported field: {name}"));
        }
    }
    Ok(())
}

fn validate_stream_options(options: Option<&StreamOptions>) -> Result<(), String> {
    if let Some(options) = options {
        if !options.extra.is_empty() {
            return Err("stream_options only supports include_usage".to_string());
        }
        if options.include_usage.is_none() {
            return Err("stream_options only supports include_usage".to_string());
        }
    }
    Ok(())
}

fn validate_prompt_shape(value: &Value) -> Result<(), String> {
    match value {
        Value::String(_) => Ok(()),
        Value::Array(items) if items.iter().all(Value::is_string) => Ok(()),
        Value::Array(items) if items.iter().all(|item| item.as_i64().is_some()) => Ok(()),
        Value::Array(items)
            if items.iter().all(|item| match item {
                Value::Array(tokens) => tokens.iter().all(|token| token.as_i64().is_some()),
                _ => false,
            }) =>
        {
            Ok(())
        }
        _ => Err(
            "prompt must be a string, list of strings, token ids, or list of token id lists"
                .to_string(),
        ),
    }
}

fn parse_positive_u64(value: Option<&Value>, default: u64, name: &str) -> Result<u64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => number
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{name} must be a positive integer")),
        _ => Err(format!("{name} must be a positive integer")),
    }
}

fn parse_positive_f64(value: Option<&Value>, default: f64, name: &str) -> Result<f64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => number
            .as_f64()
            .filter(|value| *value > 1e-10)
            .ok_or_else(|| format!("{name} must be greater than 1e-10")),
        _ => Err(format!("{name} must be greater than 1e-10")),
    }
}

fn parse_stop(value: Option<&Value>) -> Result<Vec<String>, String> {
    let stops = match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(stop)) => vec![stop.clone()],
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
            .iter()
            .map(|item| item.as_str().unwrap().to_string())
            .collect(),
        _ => return Err("stop must be a string or a list of strings".to_string()),
    };
    if stops.len() > 4 {
        return Err("stop supports at most 4 sequences".to_string());
    }
    if stops.iter().any(String::is_empty) {
        return Err("stop sequences must be non-empty".to_string());
    }
    Ok(stops)
}

fn bad_request(message: String) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::BAD_REQUEST,
        Json(make_error(message, "bad_request")),
    )
}

fn internal_error(message: String) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(make_error(message, "internal_error")),
    )
}

fn error_body(status: StatusCode, message: &str) -> (StatusCode, ErrorBody) {
    (status, make_error(message.to_string(), "engine_error"))
}

fn make_error(message: String, code: &'static str) -> ErrorBody {
    ErrorBody {
        error: ErrorMessage {
            message,
            error_type: "invalid_request_error",
            code,
        },
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    #[test]
    fn validates_completion_payload_core_fields() {
        let payload = CompletionPayload {
            model: Some("model".to_string()),
            prompt: Some(json!(["a", "b"])),
            max_tokens: Some(json!(8)),
            temperature: Some(json!(0.7)),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
                extra: BTreeMap::new(),
            }),
            n: Some(json!(2)),
            stop: Some(json!(["\n"])),
            echo: Some(true),
            ignore_eos: Some(true),
            extra: BTreeMap::new(),
        };
        let validated = validate_completion_payload(payload).unwrap();
        assert_eq!(validated.max_tokens, 8);
        assert_eq!(validated.n, 2);
        assert!(validated.stream);
        assert!(validated.stream_include_usage);
        assert!(validated.echo);
        assert!(validated.ignore_eos);
    }

    #[test]
    fn accepts_vllm_bench_completion_defaults() {
        let mut extra = BTreeMap::new();
        extra.insert("logprobs".to_string(), Value::Null);
        extra.insert("repetition_penalty".to_string(), json!(1.0));
        let payload = CompletionPayload {
            model: Some("model".to_string()),
            prompt: Some(json!("hello")),
            max_tokens: Some(json!(8)),
            temperature: Some(json!(1.0)),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
                extra: BTreeMap::new(),
            }),
            n: None,
            stop: None,
            echo: None,
            ignore_eos: Some(true),
            extra,
        };
        let validated = validate_completion_payload(payload).unwrap();
        assert!(validated.stream);
        assert!(validated.stream_include_usage);
    }

    #[test]
    fn rejects_unsupported_non_default_fields() {
        let mut extra = BTreeMap::new();
        extra.insert("top_p".to_string(), json!(0.5));
        let payload = CompletionPayload {
            model: Some("model".to_string()),
            prompt: Some(json!("hello")),
            max_tokens: None,
            temperature: None,
            stream: None,
            stream_options: None,
            n: None,
            stop: None,
            echo: None,
            ignore_eos: None,
            extra,
        };
        assert!(validate_completion_payload(payload)
            .unwrap_err()
            .contains("top_p"));
    }

    #[test]
    fn builds_completion_response() {
        let mut choices = BTreeMap::new();
        choices.insert(
            0,
            CompletionChoice {
                text: " world".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "length".to_string(),
            },
        );
        let response = completion_response(BlockingAssembly {
            request_id: "abc".to_string(),
            model: "m".to_string(),
            created: 1,
            expected: 1,
            prompt_tokens: 2,
            completion_tokens: 3,
            choices,
        });
        assert_eq!(response["id"], "cmpl-abc");
        assert_eq!(response["usage"]["total_tokens"], 5);
    }

    #[test]
    fn builds_stream_chunk() {
        let chunk = stream_chunk("abc", "m", 1, 0, "x".to_string(), None);
        assert_eq!(chunk["object"], "text_completion");
        assert_eq!(chunk["choices"][0]["text"], "x");
    }

    #[test]
    fn builds_stream_usage_chunk() {
        let chunk = stream_usage_chunk("abc", "m", 1, 2, 3);
        assert_eq!(chunk["choices"].as_array().unwrap().len(), 0);
        assert_eq!(chunk["usage"]["completion_tokens"], 3);
        assert_eq!(chunk["usage"]["total_tokens"], 5);
    }

    #[tokio::test]
    async fn handles_blocking_completion_with_fake_engine() {
        let request_endpoint = endpoint();
        let event_endpoint = endpoint();
        let _fake_engine = spawn_fake_engine(request_endpoint.clone(), event_endpoint.clone());
        let app = fake_app(request_endpoint, event_endpoint);
        let response = app
            .oneshot(json_request(json!({
                "model": "fake",
                "prompt": "hello",
                "max_tokens": 2,
                "n": 2
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "text_completion");
        assert_eq!(value["choices"].as_array().unwrap().len(), 2);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn handles_streaming_completion_with_fake_engine() {
        let request_endpoint = endpoint();
        let event_endpoint = endpoint();
        let _fake_engine = spawn_fake_engine(request_endpoint.clone(), event_endpoint.clone());
        let app = fake_app(request_endpoint, event_endpoint);
        let response = app
            .oneshot(json_request(json!({
                "model": "fake",
                "prompt": "hello",
                "max_tokens": 2,
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("\"object\":\"text_completion\""));
        assert!(body.contains("\"usage\""));
        assert!(body.contains("\"completion_tokens\":1"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn lists_models() {
        let app = fake_app(endpoint(), endpoint());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "list");
        assert_eq!(value["data"][0]["id"], "fake");
    }

    #[tokio::test]
    async fn profile_control_sends_engine_message() {
        let request_endpoint = endpoint();
        let event_endpoint = endpoint();
        let (control_tx, control_rx) = std_mpsc::channel();
        let _fake_engine = spawn_control_observer(request_endpoint.clone(), control_tx);
        let app = fake_app(request_endpoint, event_endpoint);
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/_debug/profile/start")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            control_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            "profile_start"
        );
    }

    #[tokio::test]
    async fn rejects_unsupported_fields_over_http() {
        let app = fake_app(endpoint(), endpoint());
        let response = app
            .oneshot(json_request(json!({
                "model": "fake",
                "prompt": "hello",
                "top_p": 0.5
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "bad_request");
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("top_p"));
    }

    #[tokio::test]
    async fn sends_cancel_when_stream_response_is_dropped() {
        let request_endpoint = endpoint();
        let event_endpoint = endpoint();
        let (cancel_tx, cancel_rx) = std_mpsc::channel();
        let _fake_engine = spawn_cancel_observer(request_endpoint.clone(), cancel_tx);
        let app = fake_app(request_endpoint, event_endpoint);
        let response = app
            .oneshot(json_request(json!({
                "model": "fake",
                "prompt": "hello",
                "stream": true
            })))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        assert!(cancel_rx.recv_timeout(Duration::from_secs(2)).unwrap());
    }

    fn fake_app(request_endpoint: String, event_endpoint: String) -> Router {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let event_receiver = spawn_event_receiver(event_endpoint, pending.clone());
        let state = AppState {
            model: "fake".to_string(),
            request_tx: spawn_request_sender(request_endpoint),
            pending,
            _event_receiver: event_receiver,
        };
        build_router(state)
    }

    fn json_request(value: Value) -> Request<Body> {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(value.to_string()))
            .unwrap()
    }

    fn endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        format!("tcp://127.0.0.1:{port}")
    }

    fn spawn_fake_engine(
        request_endpoint: String,
        event_endpoint: String,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let context = zmq::Context::new();
            let requests = context.socket(zmq::PULL).unwrap();
            requests.bind(&request_endpoint).unwrap();
            let events = context.socket(zmq::PUSH).unwrap();
            events.connect(&event_endpoint).unwrap();
            let bytes = requests.recv_bytes(0).unwrap();
            let request: EngineCompletionRequest = rmp_serde::from_slice(&bytes).unwrap();
            let num_choices = request.n as usize;
            send_event(
                &events,
                json!({
                    "type": "started",
                    "request_id": request.request_id,
                    "created": 1,
                    "model": request.model,
                    "num_choices": num_choices,
                    "prompt_tokens": num_choices
                }),
            );
            for choice_index in 0..num_choices {
                send_event(
                    &events,
                    json!({
                        "type": "token",
                        "request_id": request.request_id,
                        "choice_index": choice_index,
                        "text": "x",
                        "token_id": 1
                    }),
                );
                send_event(
                    &events,
                    json!({
                        "type": "finished",
                        "request_id": request.request_id,
                        "choice_index": choice_index,
                        "text": "x",
                        "token_ids": [1],
                        "finish_reason": "length",
                        "prompt_tokens": 1,
                        "completion_tokens": 1
                    }),
                );
            }
        })
    }

    fn spawn_cancel_observer(
        request_endpoint: String,
        result_tx: std_mpsc::Sender<bool>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let context = zmq::Context::new();
            let requests = context.socket(zmq::PULL).unwrap();
            requests.bind(&request_endpoint).unwrap();
            requests.set_rcvtimeo(2_000).unwrap();
            let bytes = requests.recv_bytes(0).unwrap();
            let request: EngineCompletionRequest = rmp_serde::from_slice(&bytes).unwrap();
            let bytes = requests.recv_bytes(0).unwrap();
            let cancel: EngineCancelRequest = rmp_serde::from_slice(&bytes).unwrap();
            let _ = result_tx.send(cancel.request_id == request.request_id);
        })
    }

    fn spawn_control_observer(
        request_endpoint: String,
        result_tx: std_mpsc::Sender<String>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let context = zmq::Context::new();
            let requests = context.socket(zmq::PULL).unwrap();
            requests.bind(&request_endpoint).unwrap();
            requests.set_rcvtimeo(2_000).unwrap();
            let bytes = requests.recv_bytes(0).unwrap();
            let request: EngineControlRequest = rmp_serde::from_slice(&bytes).unwrap();
            let _ = result_tx.send(request.message_type);
        })
    }

    fn send_event(socket: &zmq::Socket, value: Value) {
        let bytes = rmp_serde::to_vec_named(&value).unwrap();
        socket.send(bytes, 0).unwrap();
    }
}
