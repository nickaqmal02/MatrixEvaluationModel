"""
Topic coherence evaluation for Sentiri's BERTopic layer.

Computes C_v coherence per topic using the Röder et al. (2015) recipe:
  - sliding window of 110 tokens (NOT 10)
  - NPMI with epsilon smoothing (NOT hard -1.0 for zero co-occurrence)
  - indirect cosine similarity (the semantic rescue layer)

Two modes:
  1. gensim (preferred) — the reference implementation, exact Röder recipe
  2. hand-rolled fallback — if gensim isn't installed, fixes the 3 bugs
     from the original manual computation but skips the cosine layer

Output: JSON file with per-topic scores + mean, for the Quarto notebook
        and decisions.md.

Usage:
  cd /Users/apple/degree/finalyearengineredo
  venv/bin/python scripts/topic_coherence_eval.py

Author: Nik
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path("/Users/apple/degree/finalyearengineredo")
OUTPUT_ROOT: Path = Path("/Users/apple/degree/ModelEvaluationFinalYear/EvalResults")
EVAL_DIR: Path = PROJECT_ROOT / "scripts"
MODEL_DIR: Path = PROJECT_ROOT / "chat_analyzer" / "ml_models" / "topic_model"
OUTPUT_PATH: Path = OUTPUT_ROOT / "data" / "topic_coherence_results.json"

# ---------------------------------------------------------------------------
# Config — Röder et al. (2015) spec
# ---------------------------------------------------------------------------
WINDOW_SIZE: int = 110          # was 10 in the buggy version — Röder uses 110
EPSILON: float = 1e-12          # smoothing for zero co-occurrence (was -1.0)
TOP_N_WORDS: int = 10           # words per topic to score


# ---------------------------------------------------------------------------
# Data loading — pull topics + corpus from the saved BERTopic model
# ---------------------------------------------------------------------------
def load_topic_data() -> dict[str, Any]:
    """Load the saved BERTopic model and extract topics + corpus texts.

    Returns dict with:
      - topics: list of {name, top_words: list[str]}
      - corpus_texts: list[str] (the cleaned documents the model was trained on)
    """
    from bertopic import BERTopic

    # The model was saved via BERTopic.save("...topic_model.pkl")
    # which uses joblib/pickle internally. BERTopic.load() handles
    # both .pkl files and directory formats — use it directly.
    pkl_path: Path = MODEL_DIR / "topic_model.pkl"
    if pkl_path.exists():
        print(f"Loading BERTopic from pickle: {pkl_path} ...")
        topic_model = BERTopic.load(str(pkl_path))
    else:
        print(f"Loading BERTopic from directory: {MODEL_DIR} ...")
        topic_model = BERTopic.load(str(MODEL_DIR))

    # Get topic info (topic IDs, names, counts)
    topic_info = topic_model.get_topic_info()
    print(f"  topics found: {len(topic_info)}")

    # Get top words per topic (skip topic -1 = outliers)
    topics: list[dict[str, Any]] = []
    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue  # outlier cluster, not a real topic
        top_words_raw = topic_model.get_topic(topic_id)
        # get_topic returns list of (word, score) tuples
        top_words = [w for w, _ in top_words_raw[:TOP_N_WORDS]]
        if len(top_words) < 2:
            continue
        name = topic_info[topic_info.Topic == topic_id].Name.values[0]
        topics.append({"id": int(topic_id), "name": str(name), "top_words": top_words})

    print(f"  real topics (excl outliers): {len(topics)}")

    # Corpus texts — we need the original documents to build windows
    # BERTopic doesn't store the training corpus by default.
    # Option A: re-extract from the database (preferred)
    # Option B: use the topic model's documents_ attribute if available
    corpus_texts: list[str] = []
    try:
        # BERTopic stores docs if calculate_documents=True was set
        docs = topic_model.documents_  # type: ignore
        if docs is not None:
            corpus_texts = list(docs)
    except AttributeError:
        pass

    if not corpus_texts:
        # Fallback: pull from Django DB
        print("  model has no stored documents — pulling from DB ...")
        import os
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finalyearengineredo.settings")
        sys.path.insert(0, str(PROJECT_ROOT))
        import django
        django.setup()
        from chat_analyzer.models import MessageTopic, Conversation
        from chat_analyzer.services.text_cleaner import clean_for_topic_modeling

        # Pull all message texts — Conversation has the text directly
        # (each row IS a message, no separate Message model)
        from chat_analyzer.models import Conversation
        from chat_analyzer.services.text_cleaner import clean_for_topic_modeling

        raw_messages: list[str] = list(
            Conversation.objects.values_list("message", flat=True)
        )
        corpus_texts: list[str] = [
            clean_for_topic_modeling(m)
            for m in raw_messages
            if m and len(m.split()) >= 2
        ]
        corpus_texts = list(dict.fromkeys(corpus_texts))  # dedupe

    print(f"  corpus docs: {len(corpus_texts)}")
    return {"topics": topics, "corpus_texts": corpus_texts}


# ---------------------------------------------------------------------------
# Mode 1: gensim CoherenceModel (preferred — exact Röder recipe)
# ---------------------------------------------------------------------------
def compute_cv_gensim(
    topics: list[dict[str, Any]],
    corpus_texts: list[str],
) -> dict[str, Any]:
    """Compute C_v using gensim's CoherenceModel — the reference implementation."""
    from gensim.models import CoherenceModel

    tokenized_docs: list[list[str]] = [text.split() for text in corpus_texts]
    topic_word_lists: list[list[str]] = [t["top_words"] for t in topics]

    # gensim requires a Dictionary when using `topics` (not a trained model)
    from gensim.corpora import Dictionary
    dictionary: Dictionary = Dictionary(tokenized_docs)

    cm = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",   # NPMI + indirect cosine + window=110, all handled
    )

    per_topic: list[float] = cm.get_coherence_per_topic()
    mean: float = cm.get_coherence()

    rows: list[dict[str, Any]] = []
    for t, score in zip(topics, per_topic):
        rows.append({
            "id": t["id"],
            "name": t["name"],
            "c_v": round(score, 4),
            "top_words": ", ".join(t["top_words"][:5]),
        })

    return {
        "method": "gensim CoherenceModel (c_v)",
        "per_topic": rows,
        "mean_c_v": round(mean, 4),
        "n_topics": len(topics),
        "n_docs": len(corpus_texts),
    }


# ---------------------------------------------------------------------------
# Mode 2: hand-rolled fallback (fixes 3 bugs, no cosine layer)
# ---------------------------------------------------------------------------
def build_windows(tokenized_docs: list[list[str]], window_size: int) -> list[set[str]]:
    """Build sliding windows of `window_size` tokens from tokenized docs.

    Each window is a SET (membership only — order discarded).
    Step size = 1 (overlapping windows, per Röder spec).
    """
    windows: list[set[str]] = []
    for tokens in tokenized_docs:
        if len(tokens) < 2:
            continue
        step = 1
        for i in range(0, len(tokens) - 1, step):
            window_tokens = tokens[i : i + window_size]
            if len(window_tokens) >= 2:
                windows.append(set(window_tokens))
    return windows


def npmi_score(w1: str, w2: str, windows: list[set[str]], n_total: int) -> float:
    """NPMI with epsilon smoothing (NOT hard -1.0 for zero co-occurrence).

    NPMI = PMI / (-log P(w1,w2))
    PMI  = log( P(w1,w2) / (P(w1) * P(w2)) )
    """
    n_w1: int = sum(1 for w in windows if w1 in w)
    n_w2: int = sum(1 for w in windows if w2 in w)
    n_w1_w2: int = sum(1 for w in windows if w1 in w and w2 in w)

    if n_w1 == 0 or n_w2 == 0:
        return 0.0  # word not in corpus at all → no signal, not -1.0

    # epsilon smoothing — the key fix (was: if n_w1_w2 == 0: return -1.0)
    p_w1: float = n_w1 / n_total
    p_w2: float = n_w2 / n_total
    p_w1_w2: float = (n_w1_w2 + EPSILON) / n_total

    pmi: float = math.log(p_w1_w2 / (p_w1 * p_w2))
    npmi: float = pmi / (-math.log(p_w1_w2))
    return float(npmi)


def compute_cv_handrolled(
    topics: list[dict[str, Any]],
    corpus_texts: list[str],
) -> dict[str, Any]:
    """Hand-rolled C_v (NPMI only, no cosine layer).

    Fixes vs the original buggy computation:
      1. WINDOW_SIZE = 110 (was 10)
      2. epsilon smoothing for zero co-occurrence (was -1.0)
      3. returns 0.0 for words not in corpus (was -1.0)

    NOTE: missing indirect cosine — scores will be slightly lower than gensim's
    because semantically-related non-co-occurring pairs get no rescue.
    """
    tokenized_docs: list[list[str]] = [text.split() for text in corpus_texts]
    windows: list[set[str]] = build_windows(tokenized_docs, WINDOW_SIZE)
    n_total: int = len(windows)

    print(f"  windows built: {n_total} (size={WINDOW_SIZE})")

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for t in topics:
        words: list[str] = t["top_words"][:TOP_N_WORDS]
        if len(words) < 2:
            rows.append({
                "id": t["id"], "name": t["name"],
                "c_v": float("nan"), "top_words": ", ".join(words[:5]),
            })
            continue

        pair_scores: list[float] = [
            npmi_score(w1, w2, windows, n_total)
            for w1, w2 in combinations(words, 2)
        ]
        mean: float = float(np.mean(pair_scores))
        scores.append(mean)
        rows.append({
            "id": t["id"],
            "name": t["name"],
            "c_v": round(mean, 4),
            "top_words": ", ".join(words[:5]),
            "n_pairs": len(pair_scores),
            "n_zero_cooccur": sum(
                1 for w1, w2 in combinations(words, 2)
                if sum(1 for w in windows if w1 in w and w2 in w) == 0
            ),
        })

    overall: float = float(np.mean(scores)) if scores else float("nan")
    return {
        "method": f"hand-rolled NPMI (window={WINDOW_SIZE}, eps-smoothed, no cosine)",
        "per_topic": rows,
        "mean_c_v": round(overall, 4),
        "n_topics": len(topics),
        "n_docs": len(corpus_texts),
        "n_windows": n_total,
        "note": "missing indirect cosine layer — gensim version is preferred",
    }


# ---------------------------------------------------------------------------
# Mode 1b: gensim C_v for DEFINED topics (seed_topics.py keyword lists)
# ---------------------------------------------------------------------------
DEFINED_TOPICS: list[dict[str, Any]] = [
    {
        "name": "Progress and Sessions",
        "top_words": ["perkembangan", "sesi", "mula", "anak", "perubahan",
                       "therapy", "terapi", "rawatan", "improvement", "maju",
                       "proses", "konsisten", "fasa"],
    },
    {
        "name": "Behaviour and Transitions",
        "top_words": ["anak", "makan", "perubahan", "mula", "nak",
                       "tantrum", "mengamuk", "menangis", "marah", "lari",
                       "perangai", "meraung", "mengadu", "kawan"],
    },
    {
        "name": "Sleep and Routine",
        "top_words": ["tidur", "malam", "lena", "nyenyak", "balut",
                       "kepala", "rehat", "bangun", "mata", "terjaga", "buai"],
    },
    {
        "name": "Speech and Emotional Feedback",
        "top_words": ["perkembangan", "positif", "nampak", "sikit", "doktor",
                       "cakap", "bercakap", "sebut", "perkataan", "komunikasi",
                       "syukur", "gembira", "happy", "respond"],
    },
    {
        "name": "School and Play Interaction",
        "top_words": ["cikgu", "minggu", "sekolah", "tantrum", "rawatan",
                       "main", "bermain", "play", "kawan", "rakan",
                       "belajar", "tulis", "baca", "kelas"],
    },
]


def compute_cv_defined(
    corpus_texts: list[str],
) -> dict[str, Any]:
    """Compute C_v for the 5 manually-defined therapy topics (seed_topics.py).

    This measures whether the keywords YOU chose by domain knowledge
    actually co-occur in the corpus — i.e., are your predefined categories
    statistically grounded, or just clinically sensible?
    """
    from gensim.models import CoherenceModel
    from gensim.corpora import Dictionary

    tokenized_docs: list[list[str]] = [text.split() for text in corpus_texts]
    dictionary: Dictionary = Dictionary(tokenized_docs)
    topic_word_lists: list[list[str]] = [t["top_words"] for t in DEFINED_TOPICS]

    cm = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_v",
    )

    per_topic: list[float] = cm.get_coherence_per_topic()
    mean: float = cm.get_coherence()

    rows: list[dict[str, Any]] = []
    for t, score in zip(DEFINED_TOPICS, per_topic):
        rows.append({
            "name": t["name"],
            "c_v": round(score, 4),
            "top_words": ", ".join(t["top_words"][:5]),
            "n_keywords": len(t["top_words"]),
        })

    return {
        "method": "gensim CoherenceModel (c_v) — DEFINED topics (seed_topics.py)",
        "per_topic": rows,
        "mean_c_v": round(mean, 4),
        "n_topics": len(DEFINED_TOPICS),
        "n_docs": len(corpus_texts),
        "note": "measures whether manually-defined keywords co-occur in corpus",
    }
def main() -> None:
    print("=" * 60)
    print("Sentiri — Topic Coherence Evaluation")
    print("=" * 60)

    # Load data
    topic_data: dict[str, Any] = load_topic_data()
    topics: list[dict[str, Any]] = topic_data["topics"]
    corpus_texts: list[str] = topic_data["corpus_texts"]

    if not topics:
        print("ERROR: no topics found — check that the BERTopic model is saved.")
        sys.exit(1)

    # Try gensim first (preferred), fall back to hand-rolled
    result: dict[str, Any]
    try:
        import gensim  # noqa: F401
        print("\nUsing gensim CoherenceModel (preferred) ...")
        result = compute_cv_gensim(topics, corpus_texts)
    except ImportError:
        print("\ngensim not installed — using hand-rolled fallback ...")
        print("(install gensim for the full C_v recipe with indirect cosine)")
        result = compute_cv_handrolled(topics, corpus_texts)

    # Also compute C_v for the 5 DEFINED topics (seed_topics.py keywords)
    print("\n" + "=" * 60)
    print("Computing C_v for DEFINED topics (seed_topics.py) ...")
    print("=" * 60)
    try:
        import gensim  # noqa: F401
        defined_result: dict[str, Any] = compute_cv_defined(corpus_texts)
    except ImportError:
        defined_result = {
            "method": "skipped — gensim required for defined-topic C_v",
            "note": "install gensim to compute",
        }

    # Save JSON — both discovered + defined results
    output: dict[str, Any] = {
        "discovered_topics": result,
        "defined_topics": defined_result,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary — discovered topics
    print("\n" + "=" * 60)
    print(f"DISCOVERED TOPICS — {result['method']}")
    print(f"Topics: {result['n_topics']} | Docs: {result['n_docs']}")
    print("=" * 60)
    print(f"{'Topic':<40} {'C_v':>8} {'Top words'}")
    print("-" * 80)
    for row in result["per_topic"]:
        name: str = str(row["name"])[:38]
        cv: Any = row["c_v"]
        cv_str: str = f"{cv:>8.4f}" if isinstance(cv, float) and not math.isnan(cv) else "     nan"
        print(f"{name:<40} {cv_str}   {row['top_words']}")
    print("-" * 80)
    print(f"{'MEAN':<40} {result['mean_c_v']:>8.4f}")

    # Print summary — defined topics
    if "per_topic" in defined_result:
        print("\n" + "=" * 60)
        print(f"DEFINED TOPICS — {defined_result['method']}")
        print(f"Topics: {defined_result['n_topics']} | Docs: {defined_result['n_docs']}")
        print("=" * 60)
        print(f"{'Topic':<40} {'C_v':>8} {'Keywords (first 5)'}")
        print("-" * 80)
        for row in defined_result["per_topic"]:
            name_d: str = str(row["name"])[:38]
            cv_d: Any = row["c_v"]
            cv_str_d: str = f"{cv_d:>8.4f}" if isinstance(cv_d, float) and not math.isnan(cv_d) else "     nan"
            print(f"{name_d:<40} {cv_str_d}   {row['top_words']}")
        print("-" * 80)
        print(f"{'MEAN':<40} {defined_result['mean_c_v']:>8.4f}")
    else:
        print(f"\nDEFINED TOPICS: {defined_result.get('note', 'skipped')}")

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
