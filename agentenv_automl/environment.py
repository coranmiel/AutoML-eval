"""
AutoML Evaluation environment (gym-like interface for AgentGym).

Wraps the automl_eval HTTP server: each instance talks to the running
automl_eval/run_server.py via HTTP, same way TextCraft talks to its server.
But here the "gym Env" itself is a thin HTTP proxy — the real logic lives
in automl_eval/environment.py which is already running as a separate process.
"""

from __future__ import annotations

from typing import Any

import requests


class AutoMLEnv:
    """Single-episode AutoML evaluation environment.

    Unlike TextCraft (which embeds game logic), this Env delegates everything
    to the automl_eval HTTP server that is already running.
    """

    def __init__(self, automl_server_base: str = "http://localhost:8766"):
        self.base_url = automl_server_base
        self.observation: str = ""
        self._done: bool = False
        self._reward: float = 0.0

        resp = requests.get(f"{self.base_url}/tasks", timeout=30)
        resp.raise_for_status()
        self.task_ids: list[str] = resp.json().get("tasks", [])

    def reset(self, idx: int = 0) -> str:
        task_id = self.task_ids[idx % len(self.task_ids)] if self.task_ids else "titanic_binary"
        requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30).raise_for_status()
        obs_resp = requests.post(f"{self.base_url}/observe", json={}, timeout=30)
        obs_resp.raise_for_status()
        self.observation = obs_resp.json().get("state", "")
        self._done = False
        self._reward = 0.0
        return self.observation

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        resp = requests.post(
            f"{self.base_url}/step",
            json={"content": action},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self.observation = data.get("state", "")
        self._reward = data.get("reward", 0.0)
        self._done = data.get("done", False)
        return self.observation, self._reward, self._done, {}

    def close(self) -> None:
        try:
            requests.post(f"{self.base_url}/close", json={}, timeout=10)
        except Exception:
            pass
