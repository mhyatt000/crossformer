# decorators
from __future__ import annotations

from functools import wraps
import logging

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
    if logger.getEffectiveLevel() <= logging.DEBUG:
        return tqdm(it, **kwargs)
    return it
