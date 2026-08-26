from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def log_duration(logger, event: str, **fields):
    start = perf_counter()
    logger.info("%s started", event, **fields)
    try:
        yield
    except Exception:
        logger.exception("%s failed", event, elapsed_seconds=round(perf_counter() - start, 3), **fields)
        raise
    else:
        logger.info("%s completed", event, elapsed_seconds=round(perf_counter() - start, 3), **fields)


def count_parameters(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
