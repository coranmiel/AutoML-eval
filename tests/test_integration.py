"""
Integration test: starts both servers, tests the full HTTP chain, then shuts down.

This test verifies that:
  1. automl_eval server starts and serves tasks
  2. agentenv_automl proxy starts and forwards requests
  3. Full PLAN -> CODE -> FINAL_SUBMIT episode completes through the chain
  4. Reward is returned and the episode terminates correctly

Usage:
    python tests/test_integration.py

Requirements:
    - pip install requests fastapi uvicorn
    - No servers need to be running (the test starts them automatically)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import signal
import subprocess
import time

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUTOML_PORT = 18766
PROXY_PORT = 18080
AUTOML_URL = f"http://localhost:{AUTOML_PORT}"
PROXY_URL = f"http://localhost:{PROXY_PORT}"

SEP = "=" * 60


def wait_for_server(url: str, label: str, timeout: int = 30) -> bool:
    """Poll a server until it responds or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"  {label} is ready ({time.time() - start:.1f}s)")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    print(f"  {label} FAILED to start within {timeout}s")
    return False


def start_automl_server():
    """Start the automl_eval HTTP server."""
    proc = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "run_server.py"),
            "--port", str(AUTOML_PORT),
            "--tasks-dir", str(PROJECT_ROOT / "automl_eval" / "tasks"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    return proc


def start_proxy_server():
    """Start the agentenv_automl proxy server."""
    env = os.environ.copy()
    env["AUTOML_SERVER_BASE"] = AUTOML_URL
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "agentenv_automl.server:app",
            "--port", str(PROXY_PORT),
            "--host", "0.0.0.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return proc


def kill_proc(proc):
    """Kill a process and its children."""
    if proc and proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def test_direct_server():
    """Test the automl_eval server directly (no proxy)."""
    print(f"\n{SEP}")
    print("TEST 1: Direct automl_eval server")
    print(SEP)

    r = requests.get(f"{AUTOML_URL}/tasks", timeout=10)
    assert r.status_code == 200, f"GET /tasks failed: {r.status_code}"
    tasks = r.json()["tasks"]
    assert len(tasks) > 0, "No tasks loaded"
    print(f"  Tasks: {tasks}")

    r = requests.post(f"{AUTOML_URL}/reset", json={"task_id": tasks[0]}, timeout=10)
    assert r.status_code == 200, f"POST /reset failed: {r.status_code}"
    print(f"  Reset OK")

    r = requests.post(f"{AUTOML_URL}/observe", json={}, timeout=10)
    assert r.status_code == 200
    obs = r.json()["state"]
    assert len(obs) > 0, "Empty observation"
    print(f"  Observation: {obs[:100]}...")

    r = requests.post(
        f"{AUTOML_URL}/step",
        json={"content": "ACTION: PLAN\nHandle missing values, encode categoricals, train RandomForest"},
        timeout=30,
    )
    assert r.status_code == 200
    data = r.json()
    assert "state" in data and "reward" in data
    print(f"  Step (PLAN): reward={data['reward']}, done={data['done']}")

    r = requests.post(f"{AUTOML_URL}/close", json={}, timeout=10)
    assert r.status_code == 200
    print("  Close OK")
    print("  -> PASSED")


def test_proxy_chain():
    """Test the full chain: client -> proxy -> automl_eval."""
    print(f"\n{SEP}")
    print("TEST 2: Full proxy chain (agentenv_automl -> automl_eval)")
    print(SEP)

    r = requests.post(f"{PROXY_URL}/create", json={}, timeout=30)
    assert r.status_code == 200, f"POST /create failed: {r.status_code} {r.text}"
    data = r.json()
    assert "id" in data, f"No id in response: {data}"
    env_id = data["id"]
    print(f"  Created env_id={env_id}")
    print(f"  Initial obs: {data.get('observation', '')[:100]}...")

    plan = "ACTION: PLAN\nI will handle missing values with fillna, encode categorical features, and train a GradientBoosting model."
    r = requests.post(
        f"{PROXY_URL}/step",
        json={"id": env_id, "action": plan},
        timeout=30,
    )
    assert r.status_code == 200
    data = r.json()
    assert "observation" in data
    print(f"  Step (PLAN): reward={data.get('reward', 0):.4f}")

    code = (
        "ACTION: CODE\n"
        "```python\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "import numpy as np\n"
        "X_train = train_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "y_train = train_df['Survived']\n"
        "model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)\n"
        "model.fit(X_train, y_train)\n"
        "X_val = valid_df.drop(columns=['Survived']).select_dtypes(include='number').fillna(0)\n"
        "predictions = model.predict_proba(X_val)[:, 1]\n"
        "```"
    )
    r = requests.post(
        f"{PROXY_URL}/step",
        json={"id": env_id, "action": code},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    print(f"  Step (CODE): reward={data.get('reward', 0):.4f}")

    r = requests.post(
        f"{PROXY_URL}/step",
        json={"id": env_id, "action": "ACTION: FINAL_SUBMIT"},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("done") is True, f"Expected done=True, got {data}"
    reward = data.get("reward", 0)
    print(f"  Step (SUBMIT): reward={reward:.4f}, done={data['done']}")

    assert reward >= 0.0, f"Negative reward: {reward}"
    print(f"  Final reward: {reward:.4f}")

    r = requests.post(
        f"{PROXY_URL}/close",
        json={"id": env_id},
        timeout=10,
    )
    assert r.status_code == 200
    print("  Close OK")
    print("  -> PASSED")


def test_reset_new_episode():
    """Test that reset starts a fresh episode through the proxy."""
    print(f"\n{SEP}")
    print("TEST 3: Reset and new episode through proxy")
    print(SEP)

    r = requests.post(f"{PROXY_URL}/create", json={}, timeout=30)
    assert r.status_code == 200
    env_id = r.json()["id"]

    r = requests.post(
        f"{PROXY_URL}/reset",
        json={"id": env_id, "data_idx": 0},
        timeout=30,
    )
    assert r.status_code == 200
    data = r.json()
    assert "observation" in data
    print(f"  Reset OK, obs length: {len(data['observation'])}")

    r = requests.get(f"{PROXY_URL}/observation?id={env_id}", timeout=10)
    assert r.status_code == 200
    obs = r.json()
    print(f"  GET /observation OK")

    r = requests.post(f"{PROXY_URL}/close", json={"id": env_id}, timeout=10)
    assert r.status_code == 200
    print("  -> PASSED")


def main():
    print(SEP)
    print("  INTEGRATION TESTS: automl_eval + agentenv_automl")
    print(SEP)

    automl_proc = None
    proxy_proc = None

    try:
        print("\nStarting automl_eval server...")
        automl_proc = start_automl_server()
        if not wait_for_server(f"{AUTOML_URL}/tasks", "automl_eval"):
            raise RuntimeError("automl_eval server failed to start")

        print("\nStarting agentenv_automl proxy...")
        proxy_proc = start_proxy_server()
        if not wait_for_server(PROXY_URL, "agentenv_automl"):
            raise RuntimeError("agentenv_automl proxy failed to start")

        test_direct_server()
        test_proxy_chain()
        test_reset_new_episode()

        print(f"\n{SEP}")
        print("  ALL 3 INTEGRATION TESTS PASSED")
        print(SEP)

    except Exception as e:
        print(f"\n  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        print("\nShutting down servers...")
        kill_proc(proxy_proc)
        kill_proc(automl_proc)
        print("Done.")


if __name__ == "__main__":
    main()
