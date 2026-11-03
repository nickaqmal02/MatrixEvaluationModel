"""Lexical-miss classification probe: for every dense-retrieval miss in the
RAG eval, check whether the question shares exact keywords with its gold chunk.

Verdict semantics:
  overlap non-empty -> LEXICAL miss   (BM25 / hybrid could have caught it)
  overlap empty     -> SEMANTIC miss  (BM25 is blind here; dense's failure mode)
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import chromadb

EVAL_DIR: Path = Path("/Users/apple/degree/ModelEvaluationFinalYear")
CHROMA_DIR: Path = Path("/Users/apple/degree/finalyearengineredo/data/vector_store")
COLLECTION_NAME: str = "treatment_documents"

STOPWORDS: set[str] = {
    "kenapa", "saya", "anak", "autism", "bila", "nak", "apa", "untuk", "tak",
    "yang", "aku", "ko", "ke", "ni", "di", "dan", "dengan", "itu", "ini",
    "ada", "the", "a", "an", "is", "how", "what", "why", "my", "child",
}


def tokens(text: str) -> set[str]:
    """Lowercase word tokens, minus stopwords/short tokens."""
    return {
        tok
        for tok in re.findall(r"[\w']+", text.lower())
        if tok not in STOPWORDS and len(tok) > 2
    }


def main() -> None:
    results: dict = json.loads((EVAL_DIR / "rag_eval_results.json").read_text())
    records: list[dict] = results["records"]
    misses: list[dict] = [r for r in records if r.get("hit@5") == 0]

    gold_ids: set[str] = {cid for m in misses for cid in m["gold_chunk_ids"]}

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)
    print(f"collection size: {col.count()}, misses: {len(misses)}")

    got = col.get(ids=list(gold_ids), include=["documents"])
    docs: dict[str, str] = dict(zip(got["ids"], got["documents"]))

    tally: dict[str, int] = defaultdict(int)
    for m in misses:
        q_tokens: set[str] = tokens(m["question"])
        print(f"\n--- miss id {m['id']}: {m['question']}")
        for cid in m["gold_chunk_ids"]:
            shared: set[str] = q_tokens & tokens(docs[cid])
            kind = "LEXICAL" if shared else "SEMANTIC"
            tally[kind] += 1
            print(f"   gold {cid}: {kind} | overlap={sorted(shared)}")
            if not shared:
                print(f"      doc: {docs[cid][:180]!r}")

    print(f"\nVERDICT: {dict(tally)}")


if __name__ == "__main__":
    main()
