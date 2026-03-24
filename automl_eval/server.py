"""
HTTP server for the RL environment, compatible with the AgentGym protocol.

The environment only evaluates actions; the agent (LLM) is launched via AgentGym/AgentGym-RL.
AgentGym calls /observe -> receives state (including the task prompt) -> passes it to the LLM ->
sends the LLM response to /step. There is no need to connect an LLM inside automl_eval.

Prompt selection:
  - Each prompt = a separate task with its own task_id.
  - GET /tasks returns a list of all task_ids. AgentGym requests it and for each
    episode selects one task_id (round-robin, random, or by its own split).
  - POST /reset with the chosen task_id starts an episode for that task. The next /observe
    returns state where Description: = the task prompt. Choosing a prompt = choosing task_id in reset().

Endpoints:
  POST /reset          { "task_id": "..." }                -> { "ok": true }
  POST /observe        {}                                  -> { "state": "..." }
  POST /step           { "content": "..." }                -> { "state": "...", "reward": 0.5, "done": false }
  POST /close          {}                                  -> { "ok": true }
  GET  /tasks          {}                                  -> { "tasks": [...] }
  GET  /health         {}                                  -> { "status": "ok" }
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from automl_eval.environment import AutoMLEnvironment
from automl_eval.reward import RewardWeights
from automl_eval.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class AutoMLEnvHandler(BaseHTTPRequestHandler):
    """HTTP handler for the AutoML RL environment."""

    env: AutoMLEnvironment

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response({"status": "ok"})
        elif self.path == "/tasks":
            self._json_response({"tasks": self.env.registry.list_ids()})
        else:
            self._json_response({"error": f"Unknown GET path: {self.path}"}, status=404)

    def do_POST(self) -> None:
        body = self._read_body()

        try:
            if self.path == "/reset":
                task_id = body.get("task_id")
                if not task_id:
                    self._json_response({"error": "task_id is required"}, status=400)
                    return
                self.env.reset(task_id)
                self._json_response({"ok": True})

            elif self.path == "/observe":
                state = self.env.observe()
                self._json_response({"state": state})

            elif self.path == "/step":
                content = body.get("content", "")
                output = self.env.step(content)
                self._json_response({
                    "state": output.state,
                    "reward": output.reward,
                    "done": output.done,
                })

            elif self.path == "/close":
                self.env.close()
                self._json_response({"ok": True})

            else:
                self._json_response({"error": f"Unknown POST path: {self.path}"}, status=404)

        except Exception as exc:
            logger.exception("Error handling %s", self.path)
            self._json_response({"error": str(exc)}, status=500)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    def _json_response(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)


def run_server(
    registry: TaskRegistry,
    host: str = "0.0.0.0",
    port: int = 8766,
    reward_weights: RewardWeights | None = None,
    seed: int = 42,
) -> None:
    """Start the environment HTTP server."""
    env = AutoMLEnvironment(registry, reward_weights=reward_weights, seed=seed)
    AutoMLEnvHandler.env = env  # type: ignore[attr-defined]

    server = HTTPServer((host, port), AutoMLEnvHandler)
    logger.info("AutoML Environment server running on %s:%d", host, port)
    logger.info("Registered tasks: %s", registry.list_ids())

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        server.server_close()
