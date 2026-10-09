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