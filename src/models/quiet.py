"""Utilities for keeping third-party model examples quiet by default."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from collections.abc import Iterator


def configure_quiet_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message="builtin type SwigPyPacked has no __module__ attribute")
    warnings.filterwarnings("ignore", message="builtin type SwigPyObject has no __module__ attribute")
    warnings.filterwarnings("ignore", message="builtin type swigvarlink has no __module__ attribute")
    warnings.filterwarnings("ignore", message="'audioop' is deprecated.*")
    warnings.filterwarnings("ignore", message="'aifc' is deprecated.*")
    warnings.filterwarnings("ignore", message="'sunau' is deprecated.*")
    warnings.filterwarnings("ignore", message="Deprecated in 0.9.0.*")
    warnings.filterwarnings("ignore", message=".*Current AMD GPU is still experimental.*")


@contextlib.contextmanager
def suppress_output(enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
