"""
Entrypoint for the AutoML agent environment server.

Usage:
    automl-eval-env --port 8080 --host 0.0.0.0
"""

import argparse
import uvicorn


def launch():
    parser = argparse.ArgumentParser(description="AutoML-Eval AgentGym environment server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("agentenv_automl:app", host=args.host, port=args.port)
