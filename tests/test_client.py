"""
Test client for the AutoML RL environment HTTP server.

Usage (in project root, with .venv activated and run_server.py running):

    python tests/test_client.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import requests


BASE_URL = "http://localhost:8766"


def main() -> None:
    # 1) List tasks
    tasks_resp = requests.get(f"{BASE_URL}/tasks")
    tasks_resp.raise_for_status()
    tasks = tasks_resp.json().get("tasks", [])
    print("Tasks:", tasks)
    if not tasks:
        print("No tasks registered on server.")
        return

    task_id = tasks[0]
    print("Using task:", task_id)

    # 2) Reset episode
    reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    print("RESET:", reset_resp.json())

    # 3) First observation
    obs_resp = requests.post(f"{BASE_URL}/observe", json={})
    obs_resp.raise_for_status()
    state = obs_resp.json().get("state", "")
    print("\n=== OBSERVE (first 600 chars) ===")
    print(state[:600])
    if len(state) > 600:
        print("...")

    # 4) Simple PLAN step
    plan_text = """ACTION: PLAN
I will handle missing values, encode categorical features, do some feature engineering and train a model using ROC AUC on the validation set.
"""
    step1_resp = requests.post(f"{BASE_URL}/step", json={"content": plan_text})
    step1_resp.raise_for_status()
    step1 = step1_resp.json()

    print("\n=== STEP 1: PLAN ===")
    print("reward:", step1.get("reward"), "done:", step1.get("done"))
    state1 = step1.get("state", "")
    print(state1[:800])
    if len(state1) > 800:
        print("...")


if __name__ == "__main__":
    main()

