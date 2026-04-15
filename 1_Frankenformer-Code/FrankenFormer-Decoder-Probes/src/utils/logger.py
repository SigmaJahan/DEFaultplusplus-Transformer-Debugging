"""
Lightweight experiment logger used by the trainer and pipeline scripts.

Writes structured messages to both stdout and a rotating log file so we
can inspect runs after they finish on HPC clusters.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional


class Logger:
    """Simple console + file logger with section helpers."""

    def __init__(self, name: str, log_dir: str):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{self.name}.log"

    # ------------------------------------------------------------------ #
    def section(self, title: str):
        border = "=" * 60
        message = f"\n{border}\n{title}\n{border}"
        self._write(message)

    def subsection(self, title: str):
        border = "-" * 60
        message = f"\n{border}\n{title}\n{border}"
        self._write(message)

    def info(self, message: str, *args):
        if args:
            message = message % args
        self._write(f"[INFO] {message}")

    def warning(self, message: str, *args):
        if args:
            message = message % args
        self._write(f"[WARN] {message}")

    def error(self, message: str, *args):
        if args:
            message = message % args
        self._write(f"[ERROR] {message}")

    # ------------------------------------------------------------------ #
    def _write(self, message: str):
        timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{timestamp} | {message}"
        print(formatted)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(formatted + "\n")
