"""
Server-side wrapper that manages multiple AutoMLEnv instances.

Follows the same pattern as TextCraft_Wrapper: allocates IDs, maps
id -> Env, provides thread-safe create/step/reset/close.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from .environment import AutoMLEnv

_DEFAULT_AUTOML_BASE = os.environ.get("AUTOML_SERVER_BASE", "http://localhost:8766")


class AutoML_Wrapper:
    def __init__(self, automl_server_base: str = _DEFAULT_AUTOML_BASE):
        self.automl_server_base = automl_server_base
        self._max_id: int = 0
        self.env: dict[int, AutoMLEnv] = {}
        self.info: dict[int, dict[str, Any]] = {}
        self.ls: list[int] = []
        self._lock = threading.Lock()

    def create(self) -> dict[str, Any]:
        try:
            with self._lock:
                env_id = self._max_id
                self._max_id += 1
            new_env = AutoMLEnv(automl_server_base=self.automl_server_base)
            ob = new_env.reset(idx=0)
            self.ls.append(env_id)
            self.env[env_id] = new_env
            self.info[env_id] = {
                "observation": ob,
                "done": False,
                "reward": 0.0,
                "deleted": False,
            }
            print(f"-------AutoML Env {env_id} created--------")
            return {"id": env_id, "observation": ob, "done": False, "reward": 0.0}
        except Exception as e:
            return {"error": str(e)}

    def step(self, env_id: int, action: str) -> dict[str, Any]:
        try:
            self._check_id(env_id)
            ob, reward, done, _ = self.env[env_id].step(action)
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[env_id].update(payload)
            return payload
        except Exception as e:
            return {"error": str(e)}

    def reset(self, env_id: int, data_idx: int) -> dict[str, Any]:
        try:
            self._check_id(env_id)
            ob = self.env[env_id].reset(idx=data_idx)
            payload = {"id": env_id, "observation": ob, "done": False, "reward": 0.0}
            self.info[env_id].update(
                {"observation": ob, "done": False, "reward": 0.0, "deleted": False}
            )
            return payload
        except Exception as e:
            return {"error": str(e)}

    def get_observation(self, env_id: int) -> Any:
        try:
            self._check_id(env_id)
            return self.info[env_id]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def observe(self, env_id: int) -> dict[str, Any]:
        try:
            self._check_id(env_id)
            return {"observation": self.env[env_id].observation}
        except Exception as e:
            return {"error": str(e)}

    def close(self, env_id: int) -> dict[str, Any]:
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)
            env = self.env.pop(env_id)
            env.close()
            self.info.pop(env_id, None)
            print(f"-------AutoML Env {env_id} closed--------")
            return {"closed": True}
        except KeyError:
            return {"closed": False, "error": "Env not exist"}
        except Exception as e:
            return {"closed": False, "error": str(e)}

    def _check_id(self, env_id: int) -> None:
        if env_id not in self.info:
            raise NameError(f"The id {env_id} is not valid.")
        if self.info[env_id].get("deleted"):
            raise NameError(f"The task with environment {env_id} has been deleted.")

    def __del__(self) -> None:
        for idx in list(self.ls):
            try:
                self.close(idx)
            except Exception:
                pass


server = AutoML_Wrapper()
