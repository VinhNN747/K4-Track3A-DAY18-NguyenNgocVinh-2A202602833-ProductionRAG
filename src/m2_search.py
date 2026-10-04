from __future__ import annotations

"""Module 2: Vietnamese BM25, dense retrieval, and reciprocal-rank fusion."""

import os
import re
import sys
import hashlib
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_DIM,
                    EMBEDDING_MODEL, HYBRID_TOP_K, QDRANT_HOST, QDRANT_PORT)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text while keeping compound words searchable by BM25."""
    if os.getenv("RAG_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return " ".join(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return " ".join(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(chunk.get("text", "")).split() for chunk in self.documents]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or top_k <= 0:
            return []
        tokens = segment_vietnamese(query).split()
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)[:top_k]
        return [
            SearchResult(self.documents[index].get("text", ""), float(scores[index]),
                         dict(self.documents[index].get("metadata", {})), "bm25")
            for index in indices if scores[index] > 0
        ]


class DenseSearch:
    """Qdrant-backed vector search with an in-process fallback for no-Docker runs."""

    def __init__(self):
        self.client = None
        if os.getenv("RAG_OFFLINE", "").lower() not in {"1", "true", "yes"}:
            try:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
                self.client.get_collections()
            except Exception:
                try:
                    from qdrant_client import QdrantClient
                    self.client = QdrantClient(":memory:")
                except Exception:
                    self.client = None
        self._encoder = None
        self._chunks: list[dict] = []
        self._vectors = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("RAG_OFFLINE", "").lower() in {"1", "true", "yes"}:
                self._encoder = _HashingEncoder()
                return self._encoder
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:
                print(f"  Embedding model unavailable; using hashing fallback ({exc})")
                self._encoder = _HashingEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        self._chunks = list(chunks)
        if not chunks:
            return
        vectors = self._get_encoder().encode([chunk.get("text", "") for chunk in chunks], show_progress_bar=False)
        self._vectors = vectors
        if self.client is None:
            return
        from qdrant_client.models import Distance, PointStruct, VectorParams
        vector_size = len(vectors[0]) if len(vectors) else EMBEDDING_DIM
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection_name=collection,
                                      vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
        points = [
            PointStruct(id=index, vector=vector.tolist(),
                        payload={**chunk.get("metadata", {}), "text": chunk.get("text", "")})
            for index, (vector, chunk) in enumerate(zip(vectors, chunks))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        if top_k <= 0 or not self._chunks:
            return []
        query_vector = self._get_encoder().encode(query).tolist()
        if self.client is not None:
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            return [
                SearchResult(str(point.payload.get("text", "")), float(point.score),
                             {key: value for key, value in point.payload.items() if key != "text"}, "dense")
                for point in response.points
            ]
        import numpy as np
        vectors = np.asarray(self._vectors)
        scores = vectors @ np.asarray(query_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector) + 1e-9)
        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self._chunks[index].get("text", ""), float(scores[index]),
                             dict(self._chunks[index].get("metadata", {})), "dense") for index in indices]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using score(d) = sum(1 / (k + rank + 1))."""
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, tuple[float, SearchResult]] = {}
    for ranked_results in results_list:
        for rank, result in enumerate(ranked_results):
            score, representative = scores.get(result.text, (0.0, result))
            scores[result.text] = (score + 1.0 / (k + rank + 1), representative)
    return [SearchResult(result.text, score, dict(result.metadata), "hybrid")
            for score, result in sorted(scores.values(), key=lambda item: item[0], reverse=True)[:top_k]]


class HybridSearch:
    """Combines BM25 and dense candidates with reciprocal-rank fusion."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        return reciprocal_rank_fusion([
            self.bm25.search(query, top_k=BM25_TOP_K),
            self.dense.search(query, top_k=DENSE_TOP_K),
        ], top_k=top_k)


class _HashingEncoder:
    """Small deterministic encoder used only when the configured model is unavailable."""

    dimension = 384

    def encode(self, values, show_progress_bar: bool = False):
        import numpy as np
        is_single = isinstance(values, str)
        texts = [values] if is_single else values
        vectors = np.zeros((len(texts), self.dimension), dtype=float)
        for row, text in enumerate(texts):
            for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE):
                index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                vectors[row, index] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors[0] if is_single else vectors
