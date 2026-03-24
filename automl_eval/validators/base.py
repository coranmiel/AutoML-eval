"""
BaseValidator — base class for all validators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession


@dataclass
class ValidationResult:
    """Result of a single validator."""

    validator_name: str
    passed: bool
    score: float          # 0.0 .. 1.0
    details: str = ""
    penalty: float = 0.0  # subtracted from reward on violation


class BaseValidator(ABC):
    """Validator interface: accepts a session, returns ValidationResult."""

    name: str = "base"

    @abstractmethod
    def validate(self, session: RuntimeSession) -> ValidationResult:
        ...
