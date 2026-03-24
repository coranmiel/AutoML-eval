"""
ActionParser — parses the agent's text action and determines its type.

Supported action formats:

  ACTION: PLAN
  <plan text>

  ACTION: FEATURE_ENGINEERING
  <DSL command or code>

  ACTION: MODEL
  <model description / training code>

  ACTION: CODE
  ```python
  <arbitrary code>
  ```

  ACTION: CODE_FIX
  <fix for the previous error>

  ACTION: FINAL_SUBMIT

If the format is not recognized, the type is determined heuristically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from automl_eval.session import ActionType


@dataclass
class ParsedAction:
    action_type: ActionType
    body: str
    raw_text: str


_ACTION_HEADER = re.compile(
    r"^\s*ACTION\s*:\s*(PLAN|FEATURE_ENGINEERING|MODEL|CODE|CODE_FIX|FINAL_SUBMIT)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_CODE_BLOCK = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)

_HEURISTIC_KEYWORDS: dict[ActionType, list[str]] = {
    ActionType.PLAN: [
        "plan:", "objective:", "strategy:", "approach:", "steps:",
        "шаг 1", "step 1", "pipeline:", "plan\n",
    ],
    ActionType.FEATURE_ENGINEERING: [
        "standardscaler", "targetencod", "onehotencod", "labelencod",
        "fillna", "imputer", "add_feature", "drop_column", "feature",
        "transform", "polynomial", "interaction",
    ],
    ActionType.MODEL: [
        "lightgbm", "xgboost", "catboost", "randomforest",
        "logisticregression", "linearregression", "model",
        ".fit(", ".predict(", "gridsearch", "hyperparameter",
    ],
    ActionType.CODE_FIX: [
        "fix", "исправ", "ошибк", "error", "traceback", "bug",
    ],
    ActionType.FINAL_SUBMIT: [
        "final_submit", "submit", "финальн",
    ],
}


class ActionParser:
    """Parse text from the agent into a structured action."""

    def parse(self, text: str) -> ParsedAction:
        text = text.strip()

        match = _ACTION_HEADER.search(text)
        if match:
            action_type = ActionType(match.group(1).upper())
            body = text[match.end():].strip()
            code_in_body = _CODE_BLOCK.search(body)
            if code_in_body:
                body = code_in_body.group(1).strip()
            return ParsedAction(action_type=action_type, body=body, raw_text=text)

        code_match = _CODE_BLOCK.search(text)
        if code_match:
            return ParsedAction(
                action_type=ActionType.CODE,
                body=code_match.group(1).strip(),
                raw_text=text,
            )

        return ParsedAction(
            action_type=self._guess_type(text),
            body=text,
            raw_text=text,
        )

    def _guess_type(self, text: str) -> ActionType:
        text_lower = text.lower()

        scores: dict[ActionType, int] = {t: 0 for t in ActionType}
        for action_type, keywords in _HEURISTIC_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    scores[action_type] += 1

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] > 0:
            return best

        return ActionType.CODE
