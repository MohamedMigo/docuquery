import argparse
import json

from .engine import DocuQuery


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask questions about a folder of documents")
    ap.add_argument("folder")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=3)
    a = ap.parse_args()
    dq = DocuQuery()
    n = dq.add_directory(a.folder)
    print(f"[indexed {n} files, {len(dq.chunks)} chunks]")
    print(json.dumps(dq.ask(a.question, a.k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
