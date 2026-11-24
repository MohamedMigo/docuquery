"""FastAPI service: POST /ask {"question": "..."} -> answer + citations."""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from .engine import DocuQuery

app = FastAPI(title="DocuQuery")
dq = DocuQuery()
dq.add_directory(os.environ.get("DOCS_DIR", "data"))


class Q(BaseModel):
    question: str
    k: int = 3


@app.get("/health")
def health():
    return {"chunks": len(dq.chunks)}


@app.post("/ask")
def ask(q: Q):
    return dq.ask(q.question, q.k)
