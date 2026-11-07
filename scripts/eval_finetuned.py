#!/usr/bin/env python3
"""Run fine-tuned XLM-R on the eval test split — produce predictions + per-class metrics."""
import os, sys, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EVAL_ROOT = Path("/Users/apple/degree/ModelEvaluationFinalYear")
SENTIRI_ROOT = Path("/Users/apple/degree/finalyearengineredo")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finalyearengineredo.settings")
sys.path.insert(0, str(SENTIRI_ROOT))
import django; django.setup()
from chat_analyzer.services.text_cleaner import clean_for_sentiment

# Same data + split as the eval (seed=42, stratified)
df = pd.read_csv(SENTIRI_ROOT / "chat_analyzer/data/labeled_data/labeledsentimentdatatwo_balanced.csv")
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Clean test texts through production pipeline
test_texts = test_df["text"].astype(str).map(clean_for_sentiment)
keep = test_texts.str.strip().ne("")
test_texts = test_texts[keep].tolist()
test_labels = test_df.loc[keep, "label"].tolist()

print(f"Test split: {len(test_texts)} messages (after cleaning)")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load FINE-TUNED model (trained with corrected labels)
finetuned_path = SENTIRI_ROOT / "chat_analyzer/ml_models/xlm_roberta_malay"
print(f"Loading fine-tuned model from: {finetuned_path}")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(str(finetuned_path)).to(device).eval()

preds = []
with torch.no_grad():
    for i in range(0, len(test_texts), 32):
        enc = tokenizer(test_texts[i:i+32], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        p = model(**enc).logits.argmax(-1).cpu().numpy()
        preds.extend(p.tolist())

y_pred = np.array(preds)
y_true = np.array(test_labels)

# Compute metrics
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
neg_recall = recall_score(y_true, y_pred, labels=[0], average="macro")
neu_recall = recall_score(y_true, y_pred, labels=[1], average="macro")
pos_recall = recall_score(y_true, y_pred, labels=[2], average="macro")

print(f"\n{'='*60}")
print(f"FINE-TUNED XLM-R RESULTS (corrected labels)")
print(f"{'='*60}")
print(f"Accuracy:   {acc:.3f}")
print(f"Macro-F1:   {f1_macro:.3f}")
print(f"Neg recall: {neg_recall:.3f}")
print(f"Neu recall: {neu_recall:.3f}")
print(f"Pos recall: {pos_recall:.3f}")
print(f"\n{classification_report(y_true, y_pred, target_names=['negative','neutral','positive'])}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(f"Confusion matrix:")
print(f"  (rows=true, cols=pred)")
print(f"  {cm}")

# Save results
results = {
    "xlmr_finetuned": {
        "accuracy": round(acc, 3),
        "macro_f1": round(f1_macro, 3),
        "neg_recall": round(neg_recall, 3),
        "neu_recall": round(neu_recall, 3),
        "pos_recall": round(pos_recall, 3),
        "params_m": 278.0,
        "y_pred": y_pred.tolist(),
        "y_test": y_true.tolist(),
        "texts_kept": len(test_texts),
    }
}

outpath = EVAL_ROOT / "finetune_eval_results.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {outpath}")

# Also update eval_results.json with fine-tuned entry
with open(EVAL_ROOT / "eval_results.json") as f:
    all_results = json.load(f)
all_results["xlmr_finetuned"] = results["xlmr_finetuned"]
with open(EVAL_ROOT / "eval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Updated eval_results.json with xlmr_finetuned entry")
