"""
AgentGym EnvClient and Task for the AutoML evaluation environment.

The client talks to the agentenv_automl FastAPI proxy server (port 8080),
which in turn proxies to the automl_eval HTTP server (port 8766).

This file is installed into AgentGym by setup_agentgym.sh:
    AgentGym/agentenv/agentenv/envs/automl.py
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class AutoMLEnvClient(BaseEnvClient):
    """Client for the AutoML evaluation environment."""

    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": "",
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "",
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            json={},
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        self.env_id = ok["id"]
        self.info = {
            "observation": ok["observation"],
            "reward": 0.0,
            "done": False,
        }

    def __len__(self) -> int:
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200, f"POST /{path} failed: {res.status_code} {res.text}"
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        action_match = re.search(r"Action:\s*(.*)", action, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()

        response = self._post("step", {"action": action})

        if "error" in response:
            return StepOutput(
                state=f"Error: {response['error']}",
                reward=0.0,
                done=False,
            )

        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": idx})
        self.info.update(
            {
                "observation": response.get("observation", ""),
                "reward": 0.0,
                "done": False,
            }
        )
        return response

    def close(self) -> dict[str, Any]:
        return self._post("close", {})


class AutoMLTask(BaseTask):
    env_client_cls = AutoMLEnvClient
    env_name = "AutoML"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
