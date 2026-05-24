from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def wall_time() -> Iterator[dict[str, float]]:
    box: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["elapsed_s"] = time.perf_counter() - start


def tokens_per_second(tokens: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return tokens / elapsed_s


def block_until_ready(value):
    try:
        return value.block_until_ready()
    except AttributeError:
        pass

    if isinstance(value, dict):
        for item in value.values():
            block_until_ready(item)
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            block_until_ready(item)
        return value

    return value
