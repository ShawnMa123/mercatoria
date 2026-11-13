"""Lightweight import/export legal retrieval and answer service."""

from .answer import answer_question
from .config import AppConfig, load_config
from .retrieve import Retriever
from .store import Index, build_index, open_current, open_index

__all__ = [
    "AppConfig",
    "Index",
    "Retriever",
    "answer_question",
    "build_index",
    "load_config",
    "open_current",
    "open_index",
]

__version__ = "0.1.0"
