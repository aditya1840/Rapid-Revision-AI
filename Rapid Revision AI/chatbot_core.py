import os
from collections import defaultdict
from typing import Dict, List, Tuple

from langchain.embeddings.base import Embeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

VECTOR_DB_PATH = "vector_store/faiss_index"


def _read_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _pick_model() -> str:
    env_model = os.getenv("OLLAMA_MODEL", "").strip()
    if env_model:
        return env_model

    try:
        import ollama

        names = [m.get("name", "") for m in ollama.list().get("models", []) if m.get("name")]
    except Exception:
        names = []

    preferred = [
        "llama3.2:3b-instruct-q4_K_M",
        "phi3:mini",
        "qwen2.5:3b",
        "mistral:latest",
        "llama3:8b-instruct-q4_0",
        "llama3:latest",
    ]
    for model in preferred:
        if model in names:
            return model
    return names[0] if names else "mistral:latest"


MODEL_NAME = _pick_model()
QUESTION_TOP_K = _read_int("QUESTION_TOP_K", 2)
NOTES_TOP_K = _read_int("NOTES_TOP_K", 4)
NOTES_MAX_DOCS = _read_int("NOTES_MAX_DOCS", 5)
OLLAMA_NUM_CTX = _read_int("OLLAMA_NUM_CTX", 2048)
OLLAMA_NUM_THREAD = _read_int("OLLAMA_NUM_THREAD", max(1, (os.cpu_count() or 4) - 1))
QA_NUM_PREDICT = _read_int("QA_NUM_PREDICT", 120)
NOTES_NUM_PREDICT = _read_int("NOTES_NUM_PREDICT", 420)
QA_TIMEOUT_SEC = _read_int("QA_TIMEOUT_SEC", 120)
NOTES_TIMEOUT_SEC = _read_int("NOTES_TIMEOUT_SEC", 300)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "1h")

_notes_cache: Dict[str, Dict[str, object]] = {}


def _normalize_subject(subject: str) -> str:
    return (subject or "").strip().lower().replace(" ", "_")


def _normalize_unit(unit: str) -> str:
    value = (unit or "UNIT-I").strip().upper().replace(" ", "").replace("_", "-")
    aliases = {
        "UNIT1": "UNIT-I",
        "UNIT2": "UNIT-II",
        "UNIT3": "UNIT-III",
        "UNIT4": "UNIT-IV",
        "UNIT5": "UNIT-V",
    }
    return aliases.get(value, value)


def _error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    lower = message.lower()
    if "requires more system memory" in lower:
        return (
            f"Model `{MODEL_NAME}` needs more RAM. Install a smaller model, for example: "
            "`ollama pull llama3.2:3b-instruct-q4_K_M` and set `OLLAMA_MODEL`."
        )
    if "model not found" in lower or "unknown model" in lower:
        return f"Model `{MODEL_NAME}` not found. Run `ollama pull {MODEL_NAME}`."
    if "timed out" in lower or "read timeout" in lower:
        return "Model request timed out. Reduce context/predict tokens or increase timeout."
    return f"Model call failed: {message or exc.__class__.__name__}"


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()


def _build_llm(num_predict: int, timeout: int) -> Ollama:
    return Ollama(
        model=MODEL_NAME,
        temperature=0.15,
        num_ctx=OLLAMA_NUM_CTX,
        num_predict=num_predict,
        num_thread=OLLAMA_NUM_THREAD,
        timeout=timeout,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )


if not os.path.exists(VECTOR_DB_PATH):
    raise FileNotFoundError("Vector DB not found. Run `python ingest.py` first.")

db = FAISS.load_local(
    VECTOR_DB_PATH,
    LocalEmbeddings(),
    allow_dangerous_deserialization=True,
)
qa_llm = _build_llm(num_predict=QA_NUM_PREDICT, timeout=QA_TIMEOUT_SEC)
notes_llm = _build_llm(num_predict=NOTES_NUM_PREDICT, timeout=NOTES_TIMEOUT_SEC)
_subject_unit_docs: Dict[Tuple[str, str], List] = defaultdict(list)
for _doc in db.docstore._dict.values():
    _meta = _doc.metadata or {}
    _subject_unit_docs[(_meta.get("subject"), _meta.get("unit"))].append(_doc)
for _docs in _subject_unit_docs.values():
    _docs.sort(key=lambda d: d.metadata.get("page", 0))


def _retrieve(subject: str, unit: str, query: str, k: int) -> List:
    return db.similarity_search(
        query,
        k=max(1, k),
        filter={"subject": _normalize_subject(subject), "unit": _normalize_unit(unit)},
    )


def _doc_key(doc) -> tuple:
    meta = doc.metadata or {}
    return (meta.get("source"), meta.get("page"))


def _dedupe_docs(docs: List, max_docs: int) -> List:
    seen = set()
    unique: List = []
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= max_docs:
            break
    return unique


def _sample_unit_docs(subject: str, unit: str, max_docs: int) -> List:
    key = (_normalize_subject(subject), _normalize_unit(unit))
    docs = _subject_unit_docs.get(key, [])
    if len(docs) <= max_docs:
        return docs
    step = len(docs) / float(max_docs)
    return [docs[int(i * step)] for i in range(max_docs)]


def _retrieve_notes_docs(subject: str, unit: str, max_docs: int) -> List:
    seed_queries = [
        f"{subject} {unit} overview",
        f"{subject} {unit} core concepts definitions",
        f"{subject} {unit} mechanism architecture process",
        f"{subject} {unit} formulas terms exam questions",
    ]
    gathered: List = []
    for query in seed_queries:
        gathered.extend(_retrieve(subject, unit, query, k=NOTES_TOP_K))
    sampled = _sample_unit_docs(subject, unit, max_docs=max(2, max_docs // 2))
    return _dedupe_docs(gathered + sampled, max_docs=max_docs)


def _context(docs: List, max_chars: int, per_doc_chars: int) -> str:
    parts: List[str] = []
    used = 0
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown_source")
        page = meta.get("page")
        page_label = f"{page + 1}" if isinstance(page, int) else "?"
        text = " ".join((doc.page_content or "").split())
        if not text:
            continue
        chunk = text[:per_doc_chars]
        block = f"[{source}, page {page_label}]\n{chunk}"
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _sources(docs: List) -> List[str]:
    seen = set()
    sources: List[str] = []
    for doc in docs:
        name = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page")
        label = f"{name} (page {page + 1})" if isinstance(page, int) else name
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def _is_complete_notes(text: str) -> bool:
    answer = (text or "").strip().lower()
    needed = [
        "unit summary",
        "core concepts",
        "important terms",
        "likely exam questions",
        "2-minute revision",
    ]
    return all(part in answer for part in needed)


def ask_question(
    subject: str,
    unit: str,
    question: str,
    session_id: str = "default",
    use_history: bool = False,
    k: int = QUESTION_TOP_K,
) -> Dict[str, object]:
    del session_id, use_history
    question = (question or "").strip()
    if not question:
        return {"answer": "Enter a question first.", "sources": []}

    docs = _retrieve(subject, unit, question, k=k)
    if not docs:
        return {"answer": "No relevant content found for this subject/unit.", "sources": []}

    prompt = f"""
Use only the context to answer.
If missing, reply: Not enough information in provided notes.
Keep answer under 120 words.
Do not leave incomplete sentences.
Do not mention page numbers.
End with a complete final sentence.

Context:
{_context(docs, max_chars=1200, per_doc_chars=450)}

Question:
{question}
"""
    try:
        answer = qa_llm.invoke(prompt).strip()
    except Exception as exc:
        answer = _error_message(exc)
    return {"answer": answer, "sources": _sources(docs)}


def generate_notes(
    subject: str,
    unit: str,
    session_id: str = "default",
    use_history: bool = False,
) -> Dict[str, object]:
    del session_id, use_history
    cache_key = f"{_normalize_subject(subject)}|{_normalize_unit(unit)}|{MODEL_NAME}"
    cached = _notes_cache.get(cache_key)
    if cached and _is_complete_notes(cached.get("answer", "")):
        return {"answer": cached["answer"], "sources": list(cached["sources"])}
    if cached:
        _notes_cache.pop(cache_key, None)

    docs = _retrieve_notes_docs(subject, unit, max_docs=NOTES_MAX_DOCS)
    if not docs:
        return {"answer": "No indexed content found for this subject/unit.", "sources": []}

    ctx = _context(docs, max_chars=1700, per_doc_chars=360)
    prompt = f"""
You are a professional university teacher creating exam-ready notes.
Write clear, structured classroom notes using ONLY the given context.
Do not add facts not present in context.
If any important topic is missing, write exactly: not present in provided notes.
Cover all topics in the exact order below.
Target length: 180 to 240 words.
Each bullet should be one short line.

Output format (use this exact order):
### Unit Summary
- 2 to 3 bullets

### Core Concepts
- 4 to 5 bullets
- Each bullet should include what it is + why it matters

### Important Terms / Formulae
- 3 to 4 bullets
- Include symbols/formula names where available in context

### Likely Exam Questions
1. short question
2. short question
3. medium question
4. medium question
5. long question

### 2-Minute Revision
- 4 to 5 one-line quick-recall bullets

Context:
{ctx}
"""
    try:
        answer = notes_llm.invoke(prompt).strip()
        if not _is_complete_notes(answer):
            retry_prompt = f"""
Rewrite the notes in complete form.
Use all five sections exactly as listed.
Keep total length 180 to 220 words.
Do not stop in mid section.

### Unit Summary
### Core Concepts
### Important Terms / Formulae
### Likely Exam Questions
### 2-Minute Revision

Context:
{ctx}
"""
            answer = notes_llm.invoke(retry_prompt).strip()
    except Exception as exc:
        answer = _error_message(exc)
    result = {"answer": answer, "sources": _sources(docs)}
    if _is_complete_notes(answer):
        _notes_cache[cache_key] = {"answer": answer, "sources": list(result["sources"])}
    return result
