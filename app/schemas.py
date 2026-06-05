from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    subject: str = Field(..., min_length=2)


class QuestionResponse(BaseModel):
    answer: str
    subject: str
    sources: List[str]


class IngestResponse(BaseModel):
    subject: str
    source: str
    chunks_added: int


class SummaryResponse(BaseModel):
    summary: str


class QuestionsResponse(BaseModel):
    questions: List[str]
    raw_output: str

