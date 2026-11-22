"""Optional, isolated JSON Lines log of LLM requests and responses."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LLMTraceLog:
    def __init__(self, path: str):
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.Logger("llm_trace", level=logging.INFO)
        self._logger.propagate = False
        handler = RotatingFileHandler(log_path, maxBytes=50 * 1024 * 1024, backupCount=5,
                                      encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def write(self, event: str, **fields: object) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self._logger.info(json.dumps(record, ensure_ascii=False))

    def close(self) -> None:
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
            handler.close()
