import logging
import os
import time
from functools import wraps
from typing import Any, Callable

import joblib


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_execution(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = setup_logger(func.__module__)
        logger.info(f"▶ Iniciando: {func.__name__}")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"✔ Completado: {func.__name__} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"✘ Error en {func.__name__} ({elapsed:.2f}s): {e}", exc_info=True)
            raise
    return wrapper


def save_model(model: Any, filepath: str) -> None:
    logger = setup_logger(__name__)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Modelo guardado en: {filepath}")


def load_model(filepath: str) -> Any:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo no encontrado: {filepath}")
    return joblib.load(filepath)