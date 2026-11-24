# DocuQuery

Retrieval-augmented question answering over your own documents, with **source citations**, a CLI and a REST API. Pure Python, no API key required.

## How it works
1. Documents are split into overlapping word windows.
2. Two retrievers score each chunk: **BM25** (lexical) and **TF-IDF cosine**.
3. Rankings are merged with **Reciprocal Rank Fusion**, which is more robust than either alone.
4. The best sentence is returned as an extractive answer together with the chunk sources. Pass `generator=` to `ask()` to have any LLM write the final answer from the retrieved chunks.

## Run
```bash
pip install -r requirements.txt
python -m docuquery.cli data "How many charge cycles do LFP batteries last?"
DOCS_DIR=data uvicorn docuquery.api:app   # POST /ask {"question": "..."}
pytest -q
```

Example output:
```json
{"answer": "...last for 3000 to 6000 charge cycles.", "sources": [{"source": "data/batteries.md", "chunk": 0, "score": 0.0328}]}
```

## Layout
- `docuquery/engine.py` chunking, BM25, TF-IDF, fusion, answering
- `docuquery/api.py` FastAPI app, `docuquery/cli.py` command line
- `tests/` retrieval, citation, and fallback tests

## Ideas to extend
Swap TF-IDF for dense embeddings, add PDF loaders, or plug an LLM into `generator`.
