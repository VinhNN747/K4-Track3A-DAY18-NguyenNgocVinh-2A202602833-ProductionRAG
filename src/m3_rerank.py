from __future__ import annotations

"""Module 3: cross-encoder reranking of hybrid-search candidates."""

import os
import re
import sys
import time
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_load_attempted = False

    def _load_model(self):
        """Load the multilingual BGE cross encoder once per reranker instance."""
        if self._model is None and not self._model_load_attempted:
            self._model_load_attempted = True
            if os.getenv("RAG_OFFLINE", "").lower() in {"1", "true", "yes"}:
                raise RuntimeError("offline mode")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _fallback_scores(query: str, documents: list[dict]) -> list[float]:
        """A deterministic offline fallback; production uses the cross encoder."""
        query_terms = set(re.findall(r"[\wÀ-ỹ]+", query.lower(), flags=re.UNICODE))
        scores = []
        for document in documents:
            terms = set(re.findall(r"[\wÀ-ỹ]+", document.get("text", "").lower(), flags=re.UNICODE))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            scores.append(overlap + float(document.get("score", 0.0)) * 0.001)
        return scores

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Score query-document pairs, sort descending, and return only ``top_k``."""
        if not documents or top_k <= 0:
            return []
        try:
            model = self._load_model()
            scores = (model.predict([(query, document.get("text", "")) for document in documents])
                      if model is not None else self._fallback_scores(query, documents))
        except Exception as exc:
            print(f"  Cross-encoder unavailable; using lexical fallback ({exc})")
            scores = self._fallback_scores(query, documents)
        if isinstance(scores, (int, float)):
            scores = [scores]
        ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[:top_k])
        ]


class FlashrankReranker:
    """Optional low-latency reranker that shares the same return type."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            response = self._model.rerank(RerankRequest(query=query, passages=[{"text": d.get("text", "")} for d in documents]))
            by_text = {document.get("text", ""): document for document in documents}
            return [RerankResult(item["text"], float(by_text[item["text"]].get("score", 0.0)),
                                 float(item["score"]), dict(by_text[item["text"]].get("metadata", {})), rank)
                    for rank, item in enumerate(response[:top_k])]
        except Exception:
            return CrossEncoderReranker._fallback_to_results(query, documents, top_k)


def _fallback_to_results(query: str, documents: list[dict], top_k: int) -> list[RerankResult]:
    scores = CrossEncoderReranker._fallback_scores(query, documents)
    ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
    return [RerankResult(doc.get("text", ""), float(doc.get("score", 0.0)), float(score),
                         dict(doc.get("metadata", {})), index)
            for index, (score, doc) in enumerate(ranked[:top_k])]


# Keep the fallback available to Flashrank without creating a second scoring rule.
CrossEncoderReranker._fallback_to_results = staticmethod(_fallback_to_results)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}
