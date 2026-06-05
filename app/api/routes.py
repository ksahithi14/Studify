from functools import lru_cache

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    HealthResponse,
    IngestResponse,
    QuestionRequest,
    QuestionResponse,
    QuestionsResponse,
    SummaryResponse,
)
from app.services.study_service import StudyService


router = APIRouter()


@lru_cache(maxsize=1)
def get_service() -> StudyService:
    return StudyService()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/api/notes/ingest", response_model=IngestResponse, tags=["notes"])
async def ingest_notes(
    subject: str = Form(...),
    pdf_file: UploadFile = File(...),
) -> IngestResponse:
    try:
        result = await get_service().ingest_upload(subject=subject, upload=pdf_file)
        return IngestResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/agent/ask", response_model=QuestionResponse, tags=["agent"])
@router.post("/ask_question", response_model=QuestionResponse, tags=["legacy"])
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    try:
        result = get_service().answer_question(
            question=request.question,
            subject=request.subject,
        )
        return QuestionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/content/summary", response_model=SummaryResponse, tags=["content"])
@router.post("/generate_summary", response_model=SummaryResponse, tags=["legacy"])
async def generate_summary(pdf_file: UploadFile = File(...)) -> SummaryResponse:
    try:
        summary = await get_service().summarize_upload(upload=pdf_file)
        return SummaryResponse(summary=summary)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/content/questions", response_model=QuestionsResponse, tags=["content"])
@router.post("/generate_questions", response_model=QuestionsResponse, tags=["legacy"])
async def generate_questions(
    pdf_file: UploadFile = File(...),
    question_count: int = Form(5),
) -> QuestionsResponse:
    try:
        result = await get_service().generate_questions_upload(
            upload=pdf_file,
            question_count=question_count,
        )
        return QuestionsResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
