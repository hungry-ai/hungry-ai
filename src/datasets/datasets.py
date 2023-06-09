from pathlib import Path


def get_data_root() -> Path:
    return Path(__file__).parents[2] / "data"


def get_raw_data_root() -> Path:
    return get_data_root() / "raw"


def get_clean_data_root() -> Path:
    return get_data_root() / "clean"
