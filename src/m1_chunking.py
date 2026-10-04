from __future__ import annotations

"""Advanced chunking strategies for the production RAG pipeline."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Return a PDF text layer, or an empty string for an image-only PDF."""
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load Markdown files and PDFs which contain an extractable text layer."""
    documents = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, encoding="utf-8") as file:
            documents.append({"text": file.read(), "metadata": {"source": os.path.basename(path)}})
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(path)
        if text:
            documents.append({"text": text, "metadata": {"source": os.path.basename(path)}})
        else:
            print(f"  Bỏ qua {os.path.basename(path)}: PDF scan, cần OCR.")
    return documents


def _split_to_limit(text: str, limit: int) -> list[str]:
    """Split text into character-bounded chunks, preferring paragraph/sentence boundaries."""
    if limit <= 0:
        raise ValueError("chunk size must be positive")
    value = text.strip()
    if not value:
        return []
    pieces: list[str] = []
    remaining = value
    while len(remaining) > limit:
        candidates = [
            remaining.rfind("\n\n", 0, limit + 1),
            remaining.rfind("\n", 0, limit + 1),
            max(remaining.rfind(mark, 0, limit + 1) for mark in (". ", "! ", "? ")),
            remaining.rfind(" ", 0, limit + 1),
        ]
        cut = next((point for point in candidates if point >= limit // 2), limit)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Paragraph-based baseline chunker used for comparison."""
    metadata = metadata or {}
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > chunk_size:
            chunks.append(Chunk(current, {**metadata, "chunk_index": len(chunks), "strategy": "basic"}))
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(Chunk(current, {**metadata, "chunk_index": len(chunks), "strategy": "basic"}))
    return chunks


def _sentences(text: str) -> list[str]:
    """Retain Markdown headings while splitting natural-language sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [part.strip() for part in parts if part.strip()]


@lru_cache(maxsize=1)
def _semantic_model():
    """Use the pre-downloaded MiniLM model without turning a test run into a download."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.lower(), flags=re.UNICODE))
    right_tokens = set(re.findall(r"\w+", right.lower(), flags=re.UNICODE))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """Group adjacent sentences when their cosine similarity exceeds ``threshold``.

    The intended implementation uses MiniLM embeddings. Set
    ``RAG_USE_SEMANTIC_MODEL=1`` after pre-downloading the model to enable it.
    The lexical fallback keeps tests and offline runs fast and marks the backend
    in chunk metadata.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    metadata = metadata or {}
    sentences = _sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "similarity_backend": "none", "chunk_index": 0})]

    backend = "minilm"
    try:
        if os.getenv("RAG_USE_SEMANTIC_MODEL", "").lower() not in {"1", "true", "yes"}:
            raise RuntimeError("semantic model disabled; set RAG_USE_SEMANTIC_MODEL=1")
        import numpy as np
        embeddings = _semantic_model().encode(sentences, show_progress_bar=False)
        similarities = [
            float(np.dot(previous, current) / (np.linalg.norm(previous) * np.linalg.norm(current) + 1e-12))
            for previous, current in zip(embeddings, embeddings[1:])
        ]
    except Exception as exc:
        backend = "lexical_fallback"
        print(f"  Semantic model unavailable; using lexical fallback ({exc})")
        similarities = [_lexical_similarity(previous, current) for previous, current in zip(sentences, sentences[1:])]
        # Jaccard similarities have a much smaller typical range than cosine.
        threshold = min(threshold, 0.15)

    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    return [
        Chunk(" ".join(group), {**metadata, "strategy": "semantic", "similarity_backend": backend,
                                 "chunk_index": index})
        for index, group in enumerate(groups)
    ]


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent context chunks and smaller children linked by ``parent_id``."""
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    metadata = metadata or {}
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if not paragraphs:
        return [], []

    parent_texts: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for portion in _split_to_limit(paragraph, parent_size):
            candidate = f"{current}\n\n{portion}".strip() if current else portion
            if current and len(candidate) > parent_size:
                parent_texts.append(current)
                current = portion
            else:
                current = candidate
    if current:
        parent_texts.append(current)

    parents: list[Chunk] = []
    children: list[Chunk] = []
    source_id = re.sub(r"\W+", "_", str(metadata.get("source", "document"))).strip("_") or "document"
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{source_id}_parent_{parent_index}"
        child_texts = _split_to_limit(parent_text, child_size)
        parents.append(Chunk(parent_text, {**metadata, "chunk_type": "parent", "parent_id": parent_id,
                                           "chunk_index": parent_index, "child_count": len(child_texts)}))
        for child_index, child_text in enumerate(child_texts):
            children.append(Chunk(child_text, {**metadata, "chunk_type": "child", "child_index": child_index}, parent_id))
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk a Markdown document by heading while retaining heading and section path."""
    metadata = metadata or {}
    header_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if not matches:
        value = text.strip()
        return [Chunk(value, {**metadata, "strategy": "structure", "section": "Document", "chunk_index": 0})] if value else []

    chunks: list[Chunk] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(preamble, {**metadata, "strategy": "structure", "section": "Preamble", "chunk_index": 0}))

    heading_stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading))
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end():content_end].strip()
        # Keep a heading-only section too: it preserves top-level document context.
        section_text = f"{match.group(0).strip()}\n\n{content}".strip()
        section_path = " > ".join(item[1] for item in heading_stack)
        chunks.append(Chunk(section_text, {**metadata, "strategy": "structure", "section": section_path,
                                            "heading_level": level, "chunk_index": len(chunks)}))
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run every strategy across the supplied documents and return summary statistics."""
    def stats(chunks: list[Chunk]) -> dict:
        lengths = [len(chunk.text) for chunk in chunks]
        return ({"count": len(lengths), "avg_len": round(sum(lengths) / len(lengths)),
                 "min_len": min(lengths), "max_len": max(lengths)} if lengths
                else {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0})

    all_text = "\n\n".join(document["text"] for document in documents)
    metadata = {"source": "all"}
    basic = chunk_basic(all_text, metadata=metadata)
    semantic = chunk_semantic(all_text, metadata=metadata)
    parents, children = chunk_hierarchical(all_text, metadata=metadata)
    structure = chunk_structure_aware(all_text, metadata=metadata)
    results = {"basic": stats(basic), "semantic": stats(semantic),
               "hierarchical": {**stats(children), "parents": len(parents)}, "structure": stats(structure)}
    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, result in results.items():
        print(f"{name:<15} {result['count']:>7} {result['avg_len']:>5} {result['min_len']:>5} {result['max_len']:>5}")
    return results


if __name__ == "__main__":
    compare_strategies(load_documents())
