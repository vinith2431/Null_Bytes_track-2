from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SemanticEmbedder:
    def __init__(self, dimensions=8):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components=dimensions,
            random_state=42
        )

    def fit_transform(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = self.scaler.fit_transform(
            embeddings
        )

        return self.pca.fit_transform(
            embeddings
        )

    def transform(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        embeddings = self.scaler.transform(
            embeddings
        )

        return self.pca.transform(
            embeddings
        )

class SemanticAngleEmbedder:
    """[M1] Semantic features as qubit angles: MiniLM sentence embedding -> standardise -> PCA(8) -> [0, pi].
    The 8-qubit feature map needs angles in [0, pi]; unseen text outside the training range is clipped."""

    def __init__(self, dimensions=8, model=None):
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=dimensions, random_state=42)
        self.lo = self.hi = None

    def _encode(self, texts):
        return self.model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)

    def fit_embeddings(self, emb):
        z = self.pca.fit_transform(self.scaler.fit_transform(emb))
        self.lo, self.hi = z.min(axis=0), z.max(axis=0)
        return self

    def from_embeddings(self, emb):
        z = self.pca.transform(self.scaler.transform(emb))
        u = (np.asarray(z, dtype=np.float64) - self.lo) / np.maximum(self.hi - self.lo, 1e-9)
        return np.clip(u * np.pi, 0.0, np.pi)                # clip after scaling: float32 rounding can exceed pi

    def fit(self, texts):
        return self.fit_embeddings(self._encode(texts))

    def transform(self, texts):
        return self.from_embeddings(self._encode(texts))

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def __getstate__(self):                 # do not pickle the 90 MB transformer; reload it by name
        return {k: v for k, v in self.__dict__.items() if k != "model"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
