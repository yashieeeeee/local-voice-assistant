"""Small shared helpers: logging setup and RAM reporting."""
from __future__ import annotations

import logging
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # Quiet noisy libraries
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_ram(label: str) -> float:
    """Log this process' RSS and system RAM usage. Returns process RSS in MB."""
    import psutil

    proc_mb = psutil.Process().memory_info().rss / 1e6
    vm = psutil.virtual_memory()
    logging.getLogger("ram").info(
        "%-28s process=%6.0f MB | system used=%4.1f/%4.1f GB (%d%%)",
        label, proc_mb, vm.used / 1e9, vm.total / 1e9, vm.percent,
    )
    return proc_mb
