
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ds = load_dataset("neuralchemy/Prompt-injection-dataset", "core")

splits = {
    name: [str(x).strip() for x in ds[name]["text"]]
    for name in ("train", "validation", "test")
}

# Fit one shared representation across all partitions.
all_texts = (
    splits["train"] + splits["validation"] + splits["test"]
)
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=150_000,
    dtype="float32",
)
X = vectorizer.fit_transform(all_texts)

sizes = {name: len(texts) for name, texts in splits.items()}
offsets = {}
start = 0
for name in splits:
    offsets[name] = (start, start + sizes[name])
    start += sizes[name]

results = []
pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

for left, right in pairs:
    a0, a1 = offsets[left]
    b0, b1 = offsets[right]
    similarities = cosine_similarity(X[a0:a1], X[b0:b1], dense_output=True)

    # Keep only high-similarity pairs for manual review.
    rows, cols = (similarities >= 0.85).nonzero()
    for i, j in zip(rows, cols):
        results.append({
            "split_a": left,
            "index_a": int(i),
            "label_a": int(ds[left][int(i)]["label"]),
            "source_a": ds[left][int(i)]["source"],
            "text_a": splits[left][int(i)],
            "split_b": right,
            "index_b": int(j),
            "label_b": int(ds[right][int(j)]["label"]),
            "source_b": ds[right][int(j)]["source"],
            "text_b": splits[right][int(j)],
            "similarity": float(similarities[i, j]),
        })

out = pd.DataFrame(results)
if not out.empty:
    out = out.sort_values("similarity", ascending=False)

out.to_csv("results/near_duplicate_pairs.csv", index=False)
print("High-similarity cross-split pairs:", len(out))
if not out.empty:
    print(out[["similarity", "split_a", "source_a",
               "split_b", "source_b", "label_a", "label_b"]]
          .head(20).to_string(index=False))
    print("Saved to results/near_duplicate_pairs.csv")