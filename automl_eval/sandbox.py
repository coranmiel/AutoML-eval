"""
Sandbox — safe execution of agent code in an isolated namespace.

Restrictions:
- execution timeout,
- forbidden dangerous modules (os, subprocess, shutil, etc.),
- stdout/stderr capture.
"""

from __future__ import annotations

import io
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any

FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "sys", "importlib", "ctypes",
})


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    error: str | None = None
    returned_value: Any = None


class TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Code execution timed out")


_original_import = __import__


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    if name in FORBIDDEN_MODULES:
        raise ImportError(f"Import of '{name}' is forbidden in the sandbox.")
    return _original_import(name, *args, **kwargs)


class Sandbox:
    """Agent code executor with restrictions."""

    def __init__(self, timeout_seconds: int = 60) -> None:
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str, namespace: dict[str, Any]) -> ExecutionResult:
        """Execute code in the given namespace and return the result."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        safe_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else vars(__builtins__).copy()
        safe_builtins["__import__"] = _safe_import
        namespace["__builtins__"] = safe_builtins

        old_handler = None
        try:
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.timeout_seconds)

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<agent_code>", "exec"), namespace)

            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            return ExecutionResult(
                success=True,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )

        except TimeoutError:
            return ExecutionResult(
                success=False,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error=f"Execution timed out after {self.timeout_seconds}s",
            )
        except Exception as exc:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            tb = traceback.format_exc()
            return ExecutionResult(
                success=False,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error=f"{type(exc).__name__}: {exc}\n{tb}",
            )
        finally:
            if old_handler is not None and hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, old_handler)
