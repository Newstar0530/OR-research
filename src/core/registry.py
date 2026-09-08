from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Small explicit registry used instead of hard-coded topic branches."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, key: str, item: T) -> None:
        if not key:
            raise ValueError("Registry key cannot be empty.")
        self._items[key] = item

    def get(self, key: str, default_key: str | None = None) -> T:
        if key in self._items:
            return self._items[key]
        if default_key and default_key in self._items:
            return self._items[default_key]
        raise KeyError(f"Unknown registry key: {key}")

    def keys(self) -> list[str]:
        return sorted(self._items)

    def values(self) -> Iterable[T]:
        return self._items.values()

