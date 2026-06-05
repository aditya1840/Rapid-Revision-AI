import os
import re
from pathlib import Path
from time import perf_counter
from typing import Optional

from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from logging_utils import get_logger

DATA_DIR = Path("data")
VECTOR_DB_PATH = Path("vector_store/faiss_index")
CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "850"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "120"))
logger = get_logger("rag_chatbot.ingest")

UNIT_REGEX = re.compile(
    r"\bUNIT\s*[-–:]?\s*(IV|III|II|I|V|[1-5])\b",
    re.IGNORECASE,
)

UNIT_ALIASES = {
    "1": "UNIT-I",
    "2": "UNIT-II",
    "3": "UNIT-III",
    "4": "UNIT-IV",
    "5": "UNIT-V",
    "I": "UNIT-I",
    "II": "UNIT-II",
    "III": "UNIT-III",
    "IV": "UNIT-IV",
    "V": "UNIT-V",
}


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()


def _derive_subject(filename: str) -> str:
    stem = filename
    while stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    return stem.strip().lower().replace(" ", "_")


def _normalize_unit(match_value: Optional[str], fallback: str = "UNIT-I") -> str:
    if not match_value:
        return fallback
    key = match_value.strip().upper()
    return UNIT_ALIASES.get(key, fallback)


def ingest() -> None:
    ingest_start = perf_counter()
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    documents = []

    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        raise ValueError("No PDF files found in data/ folder.")

    for pdf_file in pdf_files:
        file_start = perf_counter()
        subject = _derive_subject(pdf_file)
        pdf_path = DATA_DIR / pdf_file

        logger.info("Processing %s (subject=%s)", pdf_file, subject)

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        current_unit = "UNIT-I"

        for page in pages:
            match = UNIT_REGEX.search(page.page_content or "")
            if match:
                current_unit = _normalize_unit(match.group(1), fallback=current_unit)

            base_meta = page.metadata or {}
            page.metadata = {
                **base_meta,
                "subject": subject,
                "unit": current_unit,
                "source": pdf_file,
            }

            documents.append(page)

        logger.info(
            "Finished %s pages=%d in %.2fs",
            pdf_file,
            len(pages),
            perf_counter() - file_start,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = LocalEmbeddings()

    VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(str(VECTOR_DB_PATH))

    total_sec = perf_counter() - ingest_start
    logger.info(
        "Vector DB created successfully. chunks=%d chunk_size=%d overlap=%d total_time=%.2fs",
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        total_sec,
    )
    print("Vector DB created successfully")
    print(f"Total chunks stored: {len(chunks)}")


if __name__ == "__main__":
    ingest()
