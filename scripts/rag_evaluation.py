""""
rag retrieval evaluation for sentiri
"""

import os
import sys
import json
from datetime import datetime
from typing import Literal
from statistics import mean


# declaring sentiri root
SENTIRI_ROOT = "/Users/apple/degree/finalyearengineredo"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finalyearengineredo.settings")
sys.path.insert(0, SENTIRI_ROOT)

import django
django.setup()

from chat_analyzer.services.rag_service import retrieve, rerank, get_collection

# load gold set 
with open("gold_set_rag.json") as f:
    gold_set = json.load(f)

# 3. metric functions
def hit_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> Literal[0,1]:
    # TODO: 1 if any gold id appears in the first k, else 0
    top_k = set(ranked_ids[:k])
    return 1 if set(gold_ids) & top_k else 0
    
def reciprocal_rank(ranked_ids: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0/rank
    return 0.0

def precision_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) ->float:
    """
    what fractions are actually gold :)
    the formula: 
        precision@k = | top_k n gold | / k 
    """
    # the improve one O(k+g) to input
    top_k = set(ranked_ids[:k])
    gold = set(gold_ids)
    # this is so good dude :)
    return len(top_k & gold) / k 
    

records = [] # NOTES: THIS IS A LIST 
for i, item in enumerate(gold_set):
    question = item["question"]
    print(f"[{i+1}/{len(gold_set)}] {question[:60]}")
    # Stage 1 — dense retrieval (top 20)
    dense = retrieve(question, top_k=20)          # list of dicts with "id" using top 20 k nearest neighbourhood
    dense_ids = [d["id"] for d in dense] # we store list of ids 
    # Stage 2 — rerank (top 5)
    reranked = rerank(question, dense)            # sorted dicts with "id"
    rerank_ids = [d["id"] for d in reranked]

    rec = {
        "id": item["id"], "question": question,
        "lang": item["lang"], "difficulty": item["difficulty"],
        "adversarial": item["adversarial"],
        "gold_chunk_ids": item["gold_chunk_ids"],
        "dense_ranked_ids": dense_ids,
        "rerank_ranked_ids": rerank_ids,
        "dense_scores": [d["similarity_score"] for d in dense],
        "rerank_scores": [d["rerank_score"] for d in reranked],
    }

    if not item["adversarial"] and item["gold_chunk_ids"]:
        gold = item["gold_chunk_ids"]
        rec["hit@5"] = hit_at_k(rerank_ids, gold, 5)
        rec["hit@20"] = hit_at_k(dense_ids, gold, 20)
        rec["mrr_rerank"] = reciprocal_rank(rerank_ids, gold)
        rec["mrr_dense"] = reciprocal_rank(dense_ids, gold)
        rec["precision@5"] = precision_at_k(rerank_ids, gold, 5)

    records.append(rec)

# 5. aggregate 
# TODO: COMPUTE MEANS FOR: HIT@5, HIT@20, MRR_RERANK, MRR_DENSE, PRECISION@5
real_records: list[dict] = [r for r in records if not r["adversarial"]] # so we only fetch the adversarial: false

aggregate: dict = {
    "n_real_questions": len(real_records),
    "hit@5": mean([r["hit@5"] for r in real_records]),
    "hit@20": mean([r["hit@20"] for r in real_records]),
    "mrr_dense": mean([r["mrr_dense"] for r in real_records]),
    "mrr_rerank": mean([r["mrr_rerank"] for r in real_records]),
    "precision@5": mean([r["precision@5"] for r in real_records]),
}

# 6. Save + print summary 
# TODO: OUTPUT DICT (META + RECORDS + AGGREGATE)
output: dict = {
    "meta": {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_questions": len(records),
        "n_real": len(real_records),
    },
    "records": records,
    "aggregate": aggregate,
}

with open("rag_eval_results.json", "w") as f:
   json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n=== RAG EVAL ({len(real_records)} real / {len(records)} total) ===")
for key, value in aggregate.items():
    print(f"{key:18s} {value}")





