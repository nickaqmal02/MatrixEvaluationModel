"""re-anchor gold chunk ids after corpus re-ingest.

old ids were <doc_hash>_<chunk_index> keyed to the pre-alignment docs.
re-ingest changed every doc hash, and two chunk boundaries moved.
"""
import json
from pathlib import Path

OLD_HASH_TO_NEW: dict[str, str] = {
    "ab083244": "7a2ee7cf",   # terapi pertuturan autism
    "5eba3fda": "7b6d878e",   # tingkah laku cabaran autism
    "e21fed4e": "a9aab0e9",   # terapi pemakanan autism
}

# chunk-boundary drift — cannot be fixed by hash swap alone
OVERRIDES: dict[int, list[str]] = {
    3: ["7a2ee7cf_1"],                  # PECS definition moved _2 -> _1
    8: ["7a2ee7cf_0", "7a2ee7cf_3"],    # "artikulasi" in _0, "sebutan" in _0/_3
}


def remap(gold_id: str) -> str:
    """swap the doc-hash prefix, keep the chunk index."""
    old_hash, idx = gold_id.split("_")
    return f"{OLD_HASH_TO_NEW[old_hash]}_{idx}"


def main() -> None:
    path: Path = Path("gold_set_rag.json")
    gold: list[dict] = json.loads(path.read_text())
    for item in gold:
        if item["id"] in OVERRIDES:
            before = item["gold_chunk_ids"]
            item["gold_chunk_ids"] = OVERRIDES[item["id"]]
            print(f"Q{item['id']:>2} override {before} -> {item['gold_chunk_ids']}")
        else:
            item["gold_chunk_ids"] = [remap(g) for g in item["gold_chunk_ids"]]
    path.write_text(json.dumps(gold, indent=2, ensure_ascii=False))
    print("rewritten:", path)


if __name__ == "__main__":
    main()
