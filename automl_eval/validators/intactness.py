"""
IntactnessValidator — checks that the agent did not modify the original data
(train_df, valid_df) in the sandbox.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from automl_eval.validators.base import BaseValidator, ValidationResult

if TYPE_CHECKING:
    from automl_eval.session import RuntimeSession


class IntactnessValidator(BaseValidator):
    name = "intactness"

    def validate(self, session: RuntimeSession) -> ValidationResult:
        intact = session.check_data_intact()

        if intact:
            return ValidationResult(
                validator_name=self.name,
                passed=True,
                score=1.0,
                details="Original train and valid data are unchanged.",
            )

        return ValidationResult(
            validator_name=self.name,
            passed=False,
            score=0.0,
            details="Agent modified original train_df or valid_df in the sandbox!",
            penalty=0.3,
        )
