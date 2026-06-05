from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.config import get_settings
from app.prompts import (
    ANSWER_PROMPT,
    QUESTION_PROMPT,
    SUMMARY_PROMPT,
    SUMMARY_REDUCE_PROMPT,
)
from app.services.llm import get_llm
from app.services.pdf_utils import (
    chunk_sentences,
    chunk_text_by_chars,
    clean_text,
    extract_text_from_pdf,
    normalize_subject,
    parse_numbered_lines,
    safe_filename,
    split_sentences,
)
from app.services.vector_store import get_vector_store


class StudyService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.llm = get_llm()
        self.answer_chain = (
            PromptTemplate.from_template(ANSWER_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        self.summary_chain = (
            PromptTemplate.from_template(SUMMARY_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        self.summary_reduce_chain = (
            PromptTemplate.from_template(SUMMARY_REDUCE_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        self.question_chain = (
            PromptTemplate.from_template(QUESTION_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    async def ingest_upload(self, subject: str, upload: UploadFile) -> Dict[str, object]:
        temp_path = await self._save_upload(upload)
        try:
            chunks_added = self.ingest_pdf(temp_path, subject, upload.filename or "upload.pdf")
            return {
                "subject": normalize_subject(subject),
                "source": safe_filename(upload.filename or "upload.pdf"),
                "chunks_added": chunks_added,
            }
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def summarize_upload(self, upload: UploadFile) -> str:
        temp_path = await self._save_upload(upload)
        try:
            return self.summarize_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def generate_questions_upload(
        self, upload: UploadFile, question_count: int
    ) -> Dict[str, object]:
        temp_path = await self._save_upload(upload)
        try:
            questions = self.generate_questions_from_pdf(
                pdf_path=temp_path,
                question_count=question_count,
            )
            return {
                "questions": questions,
                "raw_output": "\n".join(questions),
            }
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def ingest_pdf(self, pdf_path: str, subject: str, source_name: str) -> int:
        subject_code = normalize_subject(subject)
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("The uploaded PDF did not contain readable text.")

        chunks = chunk_sentences(split_sentences(text), max_sentences=7)
        if not chunks:
            raise ValueError("Unable to create chunks from the uploaded PDF.")

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "subject": subject_code,
                    "source": safe_filename(source_name),
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        self.vector_store.add_documents(documents)
        return len(documents)

    def answer_question(self, question: str, subject: str) -> Dict[str, object]:
        subject_code = normalize_subject(subject)
        docs = self.vector_store.similarity_search(
            question,
            k=self.settings.retrieval_k,
            filter={"subject": subject_code},
        )
        if not docs:
            raise ValueError(
                f"No indexed notes found for subject '{subject_code}'. Ingest notes first."
            )

        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self.answer_chain.invoke(
            {
                "subject": subject_code,
                "question": question.strip(),
                "context": context,
            }
        ).strip()

        sources = sorted(
            {
                doc.metadata.get("source", "unknown")
                for doc in docs
                if doc.metadata.get("source")
            }
        )

        return {
            "answer": answer,
            "subject": subject_code,
            "sources": sources,
        }

    def summarize_pdf(self, pdf_path: str) -> str:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("The uploaded PDF did not contain readable text.")

        chunks = chunk_text_by_chars(
            text,
            max_chars=self.settings.summary_chunk_chars,
        )
        partial_summaries: List[str] = []
        for chunk in chunks:
            partial_summaries.append(
                self.summary_chain.invoke({"text": chunk}).strip()
            )

        if len(partial_summaries) == 1:
            return partial_summaries[0]

        combined = "\n\n".join(partial_summaries)
        return self.summary_reduce_chain.invoke({"summaries": combined}).strip()

    def generate_questions_from_pdf(self, pdf_path: str, question_count: int = 5) -> List[str]:
        if question_count < 1 or question_count > 15:
            raise ValueError("question_count must be between 1 and 15.")

        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("The uploaded PDF did not contain readable text.")

        compact_text = clean_text(text)[: self.settings.question_source_chars]
        raw_output = self.question_chain.invoke(
            {
                "text": compact_text,
                "question_count": question_count,
            }
        ).strip()
        parsed = parse_numbered_lines(raw_output)
        return parsed[:question_count] if parsed else [raw_output]

    async def _save_upload(self, upload: UploadFile) -> str:
        filename = safe_filename(upload.filename or "upload.pdf")
        suffix = Path(filename).suffix or ".pdf"

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await upload.read()
            temp_file.write(content)
            return temp_file.name

