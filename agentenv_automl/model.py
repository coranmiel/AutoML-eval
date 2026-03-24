from pydantic import BaseModel
from typing import Optional


class CreateRequestBody(BaseModel):
    automl_server_base: Optional[str] = None


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: Optional[int] = 0


class CloseRequestBody(BaseModel):
    id: int
