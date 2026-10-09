
import time
import pandas as pd

from aegis.qgate.embed import TextEmbedder
from aegis.qgate.kernel import gram

DATA = "data/hf_prompt_injection/train.csv"


def main():
    df = pd.read_csv(DATA)
    texts = df["text"].astype(str).tolist()

    print(f"Fitting text embedder on {len(texts)} training samples...")
    embedder = TextEmbedder()
    X = embedder.fit_transform(texts)

    print(f"Feature matrix shape: {X.shape}")

    # Small, progressively larger benchmarks.
    for n in (8, 16, 32):
        start = time.perf_counter()
        K = gram(X[:n])
        elapsed = time.perf_counter() - start

        print(
            f"n={n:>2}: Gram matrix={K.shape}, "
            f"time={elapsed:.2f}s, "
            f"finite={bool(__import__('numpy').isfinite(K).all())}"
        )


if __name__ == "__main__":
    main()