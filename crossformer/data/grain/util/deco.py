# decorators
from __future__ import annotations

from functools import wraps
import logging
import time

from tqdm import tqdm

log = logging.getLogger(__name__)


def logged(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        log.debug(f"Calling {fn.__name__}")
        return fn(*args, **kwargs)

    return wrapper


def logbar(it, **kwargs):
    """Wrap iterable in tqdm if logging level is DEBUG, else pass through."""
    logger = logging.getLogger()
    if logger.getEffectiveLevel() <= logging.INFO:
        return tqdm(it, **kwargs)
    return it


def timeit(fn, log_fn=log.info):
    """Decorator that logs how long a function call takes at log level."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Detect class context
        qualname = fn.__qualname__
        if "." in qualname:
            # e.g. "MyClass.method"
            log_fn(f"{qualname} took {elapsed:.2f} seconds")
        else:
            log_fn(f"{fn.__name__} took {elapsed:.2f} seconds")

        return result

    return wrapper
