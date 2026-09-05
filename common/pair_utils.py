"""Shared forex pair / currency string helpers."""

from __future__ import annotations


def clip_currency(pair: str) -> str:
    """Normalise a forex pair string to uppercase, returning it unchanged if not 6 chars."""
    return (pair or "").upper()
