import os
import io
import json
import pickle
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF extraction
from pypdf import PdfReader

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# FAISS
import faiss

# Text generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Config
# -----------------------------
STORAGE_DIR = "storage"
EMB_PATH = os.path.join(STORAGE_DIR, "embeddings.pkl")
INDEX_PATH = os.path.join(STORAGE_DIR, "index.faiss")
CONFIG_PATH = os.path.join(STORAGE_DIR, "config.json")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-base"


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def read_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
    return pages


def clean_text(txt: str) -> str:
    return "\n".join(line.strip() for line in txt.splitlines() if line.strip())


def chunk_text(text: str, page: int, max_words: int = 220, overlap: int = 40) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        chunks.append({"text": " ".join(chunk_words), "page": page})
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# -----------------------------
# Embedding Store
# -----------------------------
class EmbeddingStore:
    def __init__(self, embed_model_name: str):
        self.embedder = SentenceTransformer(embed_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.texts: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(self.dim)
        self.is_trained = True

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def add(self, chunk_texts: List[str], metas: List[Dict[str, Any]]):
        vecs = self.encode(chunk_texts)
        self.index.add(vecs)
        self.texts.extend(chunk_texts)
        self.meta.extend(metas)

    def search(self, query: str, k: int = 5):
        q = self.encode([query])
        scores, idxs = self.index.search(q, min(k, len(self.texts) or 1))
        results = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append({"text": self.texts[i], "score": float(s), "meta": self.meta[i]})
        return results

    def save(self):
        ensure_dirs()
        faiss.write_index(self.index, INDEX_PATH)
        with open(EMB_PATH, "wb") as f:
            pickle.dump({"texts": self.texts, "meta": self.meta}, f)
        with open(CONFIG_PATH, "w") as f:
            json.dump({"dim": self.dim, "embed_model": EMBED_MODEL_NAME}, f, indent=2)

    @classmethod
    def load(cls, embed_model_name: str):
        ensure_dirs()
        store = cls(embed_model_name)
        if os.path.exists(INDEX_PATH) and os.path.exists(EMB_PATH):
            store.index = faiss.read_index(INDEX_PATH)
            with open(EMB_PATH, "rb") as f:
                data = pickle.load(f)
            store.texts = data.get("texts", [])
            store.meta = data.get("meta", [])
        return store


# -----------------------------
# Generator
# -----------------------------
class Generator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device >= 0:
            self.model.to("cuda")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if self.device >= 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join([f"[Page {c['meta']['page']}] {c['text']}" for c in contexts])
    instruction = (
        "You are a helpful assistant that answers ONLY from the provided context.\n"
        "If the answer cannot be found, say 'I couldn't find that in the document.'\n"
        "Be concise. Provide page citations like (p. X).\n"
    )
    return f"{instruction}\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:"


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="PDF Q&A Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: Optional[EmbeddingStore] = None
_gen: Optional[Generator] = None


def get_store() -> EmbeddingStore:
    global _store
    if _store is None:
        _store = EmbeddingStore.load(EMBED_MODEL_NAME)
    return _store


def get_generator() -> Generator:
    global _gen
    if _gen is None:
        _gen = Generator(GENERATOR_MODEL)
    return _gen


class AskRequest(BaseModel):
    question: str
    k: int = 5


@app.get("/")
async def root():
    return {"status": "ok", "message": "PDF Q&A Bot is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    content = await file.read()
    pages = read_pdf(content)
    chunk_texts, metas = [], []

    for p in pages:
        txt = clean_text(p["text"])
        if not txt:
            continue
        chunks = chunk_text(txt, page=p["page"])
        for ch in chunks:
            chunk_texts.append(ch["text"])
            metas.append({"page": ch["page"], "source": file.filename})

    if not chunk_texts:
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF")

    store = get_store()
    store.add(chunk_texts, metas)
    store.save()

    return {"ok": True, "file": file.filename, "pages": len(pages), "chunks_added": len(chunk_texts)}


@app.post("/ask")
async def ask(payload: AskRequest):
    store = get_store()
    if len(store.texts) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    results = store.search(payload.question, k=payload.k)
    if not results:
        return {"answer": "I couldn't find that in the document.", "contexts": []}

    prompt = build_prompt(payload.question, results)
    gen = get_generator()
    answer = gen.generate(prompt)

    contexts = [
        {"page": r["meta"]["page"], "source": r["meta"].get("source", ""), "score": round(r["score"], 4),
         "snippet": r["text"][:300] + ("…" if len(r["text"]) > 300 else "")}
        for r in results
    ]

    return {"answer": answer, "contexts": contexts}
