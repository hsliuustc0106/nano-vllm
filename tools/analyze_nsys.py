import argparse
import os
import sqlite3
import subprocess
from collections import defaultdict


NS_TO_MS = 1e-6


def export_sqlite(report: str) -> str:
    sqlite_path = report[:-9] + ".sqlite" if report.endswith(".nsys-rep") else report + ".sqlite"
    if os.path.exists(sqlite_path):
        return sqlite_path
    subprocess.run([
        "nsys",
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        sqlite_path,
        report,
    ], check=True)
    return sqlite_path


def table_names(conn):
    rows = conn.execute("select name from sqlite_master where type='table' order by name").fetchall()
    return [row[0] for row in rows]


def first_table(tables, candidates):
    for name in candidates:
        if name in tables:
            return name
    return None


def duration_query(conn, table):
    columns = {row[1] for row in conn.execute(f"pragma table_info({table})")}
    if {"start", "end"} <= columns:
        return "end - start"
    if "duration" in columns:
        return "duration"
    return None


def string_value(conn, value):
    if value is None:
        return "<null>"
    try:
        row = conn.execute("select value from StringIds where id = ?", (value,)).fetchone()
    except sqlite3.Error:
        row = None
    return row[0] if row else str(value)


def enum_value(conn, table, value):
    if value is None:
        return "<null>"
    if table not in table_names(conn):
        return str(value)
    row = conn.execute(f"select name from {table} where id = ?", (value,)).fetchone()
    return row[0] if row else str(value)


def summarize_kernels(conn, tables):
    table = first_table(tables, ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_KERNEL_NAMED"])
    if table is None:
        return None
    duration = duration_query(conn, table)
    columns = {row[1] for row in conn.execute(f"pragma table_info({table})")}
    name_col = "demangledName" if "demangledName" in columns else "shortName" if "shortName" in columns else "name"
    rows = conn.execute(
        f"select {name_col}, sum({duration}) total, count(*) count from {table} group by {name_col} order by total desc limit 15"
    ).fetchall()
    total = conn.execute(f"select sum({duration}), min(start), max(end) from {table}").fetchone()
    return {
        "table": table,
        "total_kernel_ms": (total[0] or 0) * NS_TO_MS,
        "first_kernel_ns": total[1],
        "last_kernel_ns": total[2],
        "top": [(string_value(conn, row[0]), row[1] * NS_TO_MS, row[2]) for row in rows],
    }


def summarize_cuda_api(conn, tables):
    table = first_table(tables, ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"])
    if table is None:
        return None
    duration = duration_query(conn, table)
    columns = {row[1] for row in conn.execute(f"pragma table_info({table})")}
    name_col = "nameId" if "nameId" in columns else "name"
    rows = conn.execute(
        f"select {name_col}, sum({duration}) total, count(*) count from {table} group by {name_col} order by total desc limit 15"
    ).fetchall()
    total = conn.execute(f"select sum({duration}) from {table}").fetchone()[0] or 0
    return {
        "table": table,
        "total_cuda_api_ms": total * NS_TO_MS,
        "top": [(string_value(conn, row[0]), row[1] * NS_TO_MS, row[2]) for row in rows],
    }


def summarize_gaps(conn, kernel_summary, top=10):
    if not kernel_summary:
        return []
    table = kernel_summary["table"]
    rows = conn.execute(f"select start, end from {table} order by start").fetchall()
    gaps = []
    last_end = None
    for start, end in rows:
        if last_end is not None and start > last_end:
            gaps.append((start - last_end, last_end, start))
        last_end = max(last_end or end, end)
    gaps.sort(reverse=True)
    return [(gap * NS_TO_MS, start, end) for gap, start, end in gaps[:top]]


def summarize_memory(conn, tables):
    result = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in tables:
            continue
        duration = duration_query(conn, table)
        columns = {row[1] for row in conn.execute(f"pragma table_info({table})")}
        breakdown = []
        if "bytes" in columns:
            count, total, bytes_total = conn.execute(
                f"select count(*), sum({duration}), sum(bytes) from {table}"
            ).fetchone()
            if "copyKind" in columns:
                rows = conn.execute(
                    f"""
                    select copyKind, srcKind, dstKind, count(*), sum(bytes), sum({duration})
                    from {table}
                    group by copyKind, srcKind, dstKind
                    order by count(*) desc
                    """
                ).fetchall()
                breakdown = [
                    (
                        enum_value(conn, "ENUM_CUDA_MEMCPY_OPER", row[0]),
                        enum_value(conn, "ENUM_CUDA_MEM_KIND", row[1]),
                        enum_value(conn, "ENUM_CUDA_MEM_KIND", row[2]),
                        row[3],
                        row[4] or 0,
                        (row[5] or 0) * NS_TO_MS,
                    )
                    for row in rows
                ]
        else:
            count, total = conn.execute(f"select count(*), sum({duration}) from {table}").fetchone()
            bytes_total = None
        result.append((table, count, (total or 0) * NS_TO_MS, bytes_total, breakdown))
    return result


def summarize_cuda_graphs(conn, tables):
    table = first_table(tables, ["CUPTI_ACTIVITY_KIND_GRAPH_TRACE"])
    if table is None:
        return None
    duration = duration_query(conn, table)
    count, total = conn.execute(f"select count(*), sum({duration}) from {table}").fetchone()
    return count, (total or 0) * NS_TO_MS


def main():
    parser = argparse.ArgumentParser(description="Summarize an Nsight Systems SQLite export for Nano-vLLM serving.")
    parser.add_argument("report")
    args = parser.parse_args()

    sqlite_path = export_sqlite(args.report)
    conn = sqlite3.connect(sqlite_path)
    tables = table_names(conn)
    print(f"sqlite {sqlite_path}")
    print("tables", ", ".join(tables))

    kernels = summarize_kernels(conn, tables)
    if kernels:
        span_ms = (kernels["last_kernel_ns"] - kernels["first_kernel_ns"]) * NS_TO_MS if kernels["first_kernel_ns"] else 0
        busy = kernels["total_kernel_ms"] / span_ms * 100 if span_ms else 0
        print(f"gpu_kernel_total_ms {kernels['total_kernel_ms']:.3f}")
        print(f"gpu_kernel_span_ms {span_ms:.3f}")
        print(f"gpu_kernel_busy_pct_within_kernel_span {busy:.1f}")
        print("top_kernels")
        for name, total_ms, count in kernels["top"]:
            print(f"  {total_ms:.3f} ms {count}x {name}")

    cuda_api = summarize_cuda_api(conn, tables)
    if cuda_api:
        print(f"cuda_api_total_ms {cuda_api['total_cuda_api_ms']:.3f}")
        print("top_cuda_api")
        for name, total_ms, count in cuda_api["top"]:
            print(f"  {total_ms:.3f} ms {count}x {name}")

    graphs = summarize_cuda_graphs(conn, tables)
    if graphs:
        count, total_ms = graphs
        print(f"cuda_graph_trace {count} events, {total_ms:.3f} ms")

    gaps = summarize_gaps(conn, kernels)
    if gaps:
        print("top_gpu_idle_gaps")
        for gap_ms, start, end in gaps:
            print(f"  {gap_ms:.3f} ms start_ns={start} end_ns={end}")

    memory = summarize_memory(conn, tables)
    if memory:
        print("memory_activity")
        for table, count, total_ms, bytes_total, breakdown in memory:
            bytes_text = f", {bytes_total / (1024 * 1024):.3f} MiB" if bytes_total is not None else ""
            print(f"  {table}: {count} events, {total_ms:.3f} ms{bytes_text}")
            for copy_kind, src_kind, dst_kind, kind_count, kind_bytes, kind_ms in breakdown:
                print(
                    f"    {copy_kind} {src_kind}->{dst_kind}: "
                    f"{kind_count} events, {kind_ms:.3f} ms, {kind_bytes / 1024:.3f} KiB"
                )


if __name__ == "__main__":
    main()
