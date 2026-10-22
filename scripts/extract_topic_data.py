"""Extract topic data from Sentiri DB for the evaluation notebook.
Saves topic_clusters.json with per-topic texts, keywords, and stats."""
import os, sys, json
from pathlib import Path
from collections import Counter

SENTIRI = Path("/Users/apple/degree/finalyearengineredo")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "finalyearengineredo.settings")
sys.path.insert(0, str(SENTIRI))
import django; django.setup()

from chat_analyzer.models import Conversation, MessageTopic, Topic
from django.db.models import Count

# --- Extract topic-level data ---
data = {
    "total_conversations": Conversation.objects.count(),
    "total_with_cleaned_topic": Conversation.objects.filter(
        cleaned_text_topic__isnull=False
    ).exclude(cleaned_text_topic="").count(),
    "total_messagetopic_rows": MessageTopic.objects.count(),
    "conversations_multi_topic": Conversation.objects.annotate(
        n=Count("topics")
    ).filter(n__gt=1).count(),
    "topics": [],
}

for t in Topic.objects.all():
    # Get all texts assigned to this topic
    texts = list(Conversation.objects.filter(
        topics__topic=t,
        cleaned_text_topic__isnull=False
    ).exclude(cleaned_text_topic="").values_list("cleaned_text_topic", flat=True))

    # Top words across all texts for this topic (for coherence)
    word_freq = Counter()
    for text in texts:
        for w in text.split():
            word_freq[w] += 1
    top_words = [w for w, _ in word_freq.most_common(10)]

    data["topics"].append({
        "name": t.name,
        "status": t.status,
        "message_count": len(texts),
        "primary_count": MessageTopic.objects.filter(topic=t, is_primary=True).count(),
        "top_words": top_words,
        "texts_sample": texts[:10],
    })

# --- All cleaned texts (for coherence corpus) ---
all_texts = list(Conversation.objects.filter(
    cleaned_text_topic__isnull=False
).exclude(cleaned_text_topic="").values_list("cleaned_text_topic", flat=True))
data["corpus_texts"] = all_texts

out = Path("/Users/apple/degree/ModelEvaluationFinalYear/topic_clusters.json")
json.dump(data, open(out, "w"), indent=2, ensure_ascii=False)
print(f"Saved {out} ({out.stat().st_size // 1024}KB)")
print(f"Topics: {len(data['topics'])}")
print(f"Corpus texts: {len(all_texts)}")
for t in data["topics"]:
    print(f"  {t['name']:40s} msgs={t['message_count']:4d} primary={t['primary_count']:4d} top_words={t['top_words'][:5]}")
