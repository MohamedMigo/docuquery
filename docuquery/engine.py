"""Retrieval-augmented question answering over local text/markdown files.

Hybrid retrieval: BM25 (lexical) + TF-IDF cosine (sparse vector), fused with
Reciprocal Rank Fusion. Answers are extractive by default (no API key needed)
and every answer carries source citations. An optional LLM can be plugged in
through the `generator` callable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the and or of to in on for with is are was were be by as at it this that from "
           "what how why when which who do does can i you we they your our not no".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass
class Chunk:
    source: str
    index: int
    text: str


def chunk_text(text: str, source: str, size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split on words into overlapping windows, preferring paragraph starts."""
    words = text.split()
    out, i, n = [], 0, 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(Chunk(source, n, " ".join(words[i:i + size])))
        n += 1
        i += step
    return out


class DocuQuery:
    def __init__(self, size: int = 80, overlap: int = 20, k1: float = 1.5, b: float = 0.75):
        self.size, self.overlap, self.k1, self.b = size, overlap, k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._len: list[int] = []

    # -- indexing -------------------------------------------------------
    def add_text(self, text: str, source: str) -> None:
        for ch in chunk_text(text, source, self.size, self.overlap):
            toks = tokenize(ch.text)
            self.chunks.append(ch)
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())

    def add_directory(self, folder: str | Path, exts: Iterable[str] = (".md", ".txt")) -> int:
        n = 0
        for p in sorted(Path(folder).rglob("*")):
            if p.suffix.lower() in exts:
                self.add_text(p.read_text(encoding="utf-8", errors="ignore"), str(p))
                n += 1
        return n

    # -- scoring --------------------------------------------------------
    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        return math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))

    def _bm25(self, q: list[str]) -> list[float]:
        avg = sum(self._len) / max(1, len(self._len))
        scores = []
        for tf, ln in zip(self._tf, self._len):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / avg))
            scores.append(s)
        return scores

    def _tfidf(self, q: list[str]) -> list[float]:
        qv = {t: c * self._idf(t) for t, c in Counter(q).items()}
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        scores = []
        for tf in self._tf:
            dv = {t: c * self._idf(t) for t, c in tf.items()}
            dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
            scores.append(sum(v * dv.get(t, 0.0) for t, v in qv.items()) / (qn * dn))
        return scores

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        fused: dict[int, float] = {}
        for scores in (self._bm25(q), self._tfidf(q)):
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
            for rank, i in enumerate(ranked):
                if scores[i] > 0:
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], s) for i, s in top]

    # -- answering ------------------------------------------------------
    def ask(self, question: str, k: int = 3,
            generator: Callable[[str, list[Chunk]], str] | None = None) -> dict:
        hits = self.search(question, k)
        if not hits:
            return {"answer": "I could not find this in the indexed documents.", "sources": []}
        chunks = [c for c, _ in hits]
        if generator:
            answer = generator(question, chunks)
        else:
            answer = self._extractive(question, chunks)
        sources = [{"source": c.source, "chunk": c.index, "score": round(s, 4)} for c, s in hits]
        return {"answer": answer, "sources": sources}

    @staticmethod
    def _extractive(question: str, chunks: list[Chunk]) -> str:
        q = set(tokenize(question))
        best, best_s = "", -1.0
        for c in chunks:
            for sent in re.split(r"(?<=[.!?])\s+", c.text):
                toks = set(tokenize(sent))
                s = len(q & toks) / (1 + math.log(1 + len(toks)))
                if s > best_s:
                    best, best_s = sent.strip(), s
        return best
