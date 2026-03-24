"""
Launch the AutoML RL Environment HTTP server.

    python run_server.py --port 8766 --tasks-dir automl_eval/tasks
"""
import argparse
import logging
from automl_eval.server import run_server
from automl_eval.task_registry import TaskRegistry


def main():
    parser = argparse.ArgumentParser(description="AutoML RL Environment Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--tasks-dir", default="automl_eval/tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    registry = TaskRegistry()
    n = registry.load_directory(args.tasks_dir)
    logging.info("Loaded %d tasks from %s", n, args.tasks_dir)

    run_server(registry, host=args.host, port=args.port, seed=args.seed)


if __name__ == "__main__":
    main()
