"""
LLM — a GPT-style language model built entirely from scratch.

Public API
----------
    from llm import CharTokenizer, GPT, train, generate
"""

from .tokenizer import CharTokenizer
from .model import GPT, cross_entropy_loss
from .trainer import train, get_batch
from .generate import generate
from .optimizer import Adam

__all__ = [
    "CharTokenizer",
    "GPT",
    "cross_entropy_loss",
    "train",
    "get_batch",
    "generate",
    "Adam",
]
