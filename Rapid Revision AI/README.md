# Rapid Revision AI

Rapid Revision AI is an offline, unit-wise exam assistant built with Streamlit, FAISS, and Ollama.
It answers questions from your own PDFs and can generate structured revision notes for a selected subject and unit.

## Features
- Offline RAG pipeline (local embeddings + local LLM via Ollama).
- Subject/unit-filtered retrieval from indexed PDFs.
- "Ask Question" tab with source citations (file name + page).
- "Generate Notes" tab for exam-focused, sectioned notes.
- FAISS vector index generation from PDFs in `data/`.

## Project structure
- `app.py`: Streamlit UI (subject/unit selection, Q&A, notes).
- `ingest.py`: PDF loading, unit tagging, chunking, FAISS index build.
- `chatbot_core.py`: model selection, retrieval, prompting, answer/notes generation.
- `logging_utils.py`: rotating logger setup.
- `data/`: input PDFs.
- `vector_store/faiss_index/`: generated FAISS index.
- `logs/chatbot.log`: runtime logs (created at runtime).

## Prerequisites
- Python 3.10+ recommended
- Ollama installed and running: https://ollama.com

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Pull at least one Ollama model (example):
   ```bash
   ollama pull mistral:latest
   ```
3. Put your PDFs in `data/`.
4. Build the index:
   ```bash
   python ingest.py
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## PDF and metadata conventions
- Subject is derived from the PDF filename, lowercased with spaces replaced by `_`.
  - Example: `Machine Learning.pdf` -> `machine_learning`
- The UI currently offers these subjects:
  - `machine_learning`, `deep_learning`, `compiler_design`, `data_structures`, `environmental_science`
- Unit is inferred while ingesting using headings like `UNIT-I`, `UNIT II`, `UNIT 3`, etc.
- If no unit marker is found yet, pages default to `UNIT-I`.

## Environment variables
Optional runtime tuning:
- `OLLAMA_MODEL` (default: auto-pick from local models)
- `QUESTION_TOP_K` (default: `2`)
- `NOTES_TOP_K` (default: `4`)
- `NOTES_MAX_DOCS` (default: `5`)
- `OLLAMA_NUM_CTX` (default: `2048`)
- `OLLAMA_NUM_THREAD` (default: `cpu_count - 1`, minimum `1`)
- `QA_NUM_PREDICT` (default: `120`)
- `NOTES_NUM_PREDICT` (default: `420`)
- `QA_TIMEOUT_SEC` (default: `120`)
- `NOTES_TIMEOUT_SEC` (default: `300`)
- `OLLAMA_KEEP_ALIVE` (default: `1h`)
- `INGEST_CHUNK_SIZE` (default: `850`)
- `INGEST_CHUNK_OVERLAP` (default: `120`)

Model auto-pick order (if `OLLAMA_MODEL` is unset):
`llama3.2:3b-instruct-q4_K_M` -> `phi3:mini` -> `qwen2.5:3b` -> `mistral:latest` -> `llama3:8b-instruct-q4_0` -> `llama3:latest` -> first local model found -> `mistral:latest`.

Windows example:
```bash
set QUESTION_TOP_K=3
set OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
streamlit run app.py
```

## Troubleshooting
- `Vector DB not found`: run `python ingest.py` first.
- No answers for a subject/unit: ensure PDF filename matches one of the UI subject names and re-ingest.
- Weak retrieval: confirm PDFs contain selectable text (not scanned images), then rebuild index.
- Ollama errors: ensure Ollama service is running and the selected model is pulled locally.
