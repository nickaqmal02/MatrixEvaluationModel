"""Evaluate all trained checkpoints on the shared test split (inference only).
Mirrors the notebook's protocol: same split, cleaned text, same metrics.
Writes eval_results.json for the notebook's metrics-matrix cell."""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SENTIRI = Path("/Users/apple/degree/finalyearengineredo")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finalyearengineredo.settings")
sys.path.insert(0, str(SENTIRI))
import django; django.setup()
from chat_analyzer.services.text_cleaner import clean_for_sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA = SENTIRI / "chat_analyzer/data/labeled_data/labeledsentimentdatatwo_balanced.csv"
df = pd.read_csv(DATA)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

texts = test_df["text"].astype(str).map(clean_for_sentiment)
keep = texts.str.strip().ne("")
test_texts = texts[keep].tolist()
y_test = test_df.loc[keep, "label"].values

device = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {"xlmr": "xlm-roberta-base", "mbert": "google-bert/bert-base-multilingual-cased",
          "indobert": "indobenchmark/indobert-base-p1",
          "distilbert": "distilbert/distilbert-base-multilingual-cased"}

results = {}
for run_id, base in MODELS.items():
    ckpt = Path(f"runs/{run_id}_best")
    if not ckpt.exists():
        print(f"SKIP {run_id}: no checkpoint"); continue
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            enc = tok(test_texts[i:i+32], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(device)
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    results[run_id] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "neg_recall": float(recall_score(y_test, y_pred, labels=[0], average="macro")),
        "pos_recall": float(recall_score(y_test, y_pred, labels=[2], average="macro")),
        "neu_recall": float(recall_score(y_test, y_pred, labels=[1], average="macro")),
        "params_m": round(sum(p.numel() for p in model.parameters())/1e6, 1),
        "y_pred": [int(v) for v in y_pred],
        "y_test": [int(v) for v in y_test],
        "texts_kept": len(y_pred),
    }
    print(run_id, {k: v for k, v in results[run_id].items() if k not in ("y_pred", "y_test")})

json.dump(results, open("eval_results.json", "w"), indent=2)
print("saved eval_results.json")
