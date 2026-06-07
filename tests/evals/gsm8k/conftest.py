from pathlib import Path


def pytest_addoption(parser):
    """Add custom command line options used by GSM8K accuracy tests."""

    parser.addoption(
        "--config-list-file",
        default="configs/models-small.txt",
        help="File containing list of GSM8K YAML/JSON config files to test",
    )


def pytest_generate_tests(metafunc):
    """Generate GSM8K tests from a config list file."""

    if "config_filename" not in metafunc.fixturenames:
        return

    config_list_file = metafunc.config.getoption("--config-list-file")

    config_list_path = Path(config_list_file)
    if not config_list_path.is_absolute():
        # Prefer paths relative to this directory, fallback to cwd.
        candidate_path = Path(__file__).parent / config_list_file
        if candidate_path.exists():
            config_list_path = candidate_path
        else:
            config_list_path = Path.cwd() / config_list_file

    config_files = []
    if config_list_path.exists():
        config_dir = config_list_path.parent
        with open(config_list_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                config_path = config_dir / line
                if config_path.exists():
                    config_files.append(config_path)
                else:
                    print(f"✗ Missing GSM8K config: {config_path}")

    if config_files:
        metafunc.parametrize(
            "config_filename",
            config_files,
            ids=[path.stem for path in config_files],
        )
    else:
        print(f"No GSM8K configs found for: {config_list_path}")
