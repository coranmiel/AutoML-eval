"""
FastAPI HTTP service for the AutoML evaluation environment.

Follows the AgentGym protocol: /create, /step, /reset, /observation, /close.
This service acts as a proxy layer between AgentGym-RL and the automl_eval
HTTP server (run_server.py).
"""

from fastapi import FastAPI

from .model import CreateRequestBody, StepRequestBody, ResetRequestBody, CloseRequestBody
from .env_wrapper import server

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment AutoML-Eval."


@app.post("/create")
def create(body: CreateRequestBody):
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    print(f"/step {body.id} {body.action[:80]}...")
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    print(f"/reset {body.id} {body.data_idx}")
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
def get_observation(id: int):
    print(f"/observation {id}")
    return server.observe(id)


@app.post("/close")
def close(body: CloseRequestBody):
    print(f"/close {body.id}")
    return server.close(body.id)
