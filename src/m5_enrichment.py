from __future__ import annotations

"""Module 5: enrich chunks before indexing, with an API-free fallback."""

import json
import os
import re
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _chat(system: str, user: str, max_tokens: int) -> str | None:
    if not OPENAI_API_KEY or os.getenv("RAG_OFFLINE", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        from openai import OpenAI
        response = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  Enrichment API unavailable; using local fallback ({exc})")
        return None


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def summarize_chunk(text: str) -> str:
    response = _chat("Tóm tắt đoạn văn sau bằng tiếng Việt trong tối đa hai câu, không thêm thông tin.", text, 150)
    if response:
        return response
    sentences = _sentences(text)
    return " ".join(sentences[:2]) if sentences else text.strip()


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    response = _chat(
        f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. Mỗi câu một dòng, không đánh số.",
        text, 200,
    )
    if response:
        questions = [line.strip().lstrip("0123456789. -)") for line in response.splitlines() if line.strip()]
        return questions[:n_questions]
    sentences = _sentences(text)
    if not sentences:
        return []
    keywords = re.findall(r"[A-Za-zÀ-ỹ][\wÀ-ỹ-]*", sentences[0])
    subject = " ".join(keywords[:6]) or "nội dung này"
    return [f"{subject} quy định điều gì?", "Điều kiện hoặc con số quan trọng là gì?"][:n_questions]


def contextual_prepend(text: str, document_title: str = "") -> str:
    response = _chat(
        "Viết đúng một câu tiếng Việt đặt bối cảnh cho đoạn trích trong tài liệu; không suy đoán.",
        f"Tài liệu: {document_title}\n\nĐoạn trích:\n{text}", 80,
    )
    if not response:
        title = document_title or "tài liệu nội bộ"
        response = f"Đoạn trích từ {title} nói về {summarize_chunk(text).rstrip('.')} ."
    return f"{response}\n\n{text}"


def extract_metadata(text: str) -> dict:
    response = _chat(
        'Trích xuất JSON hợp lệ, không markdown: {"topic":"...","entities":["..."],"category":"policy|hr|it|finance|safety|training","language":"vi|en"}.',
        text, 160,
    )
    if response:
        try:
            parsed = json.loads(response.removeprefix("```json").removesuffix("```").strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    category = "policy"
    for candidate, markers in {
        "it": ("mật khẩu", "vpn", "mfa", "bảo mật"),
        "finance": ("lương", "chi phí", "tạm ứng", "thưởng"),
        "training": ("đào tạo", "mentor", "buddy"),
        "safety": ("an toàn", "sự cố", "pccc"),
        "hr": ("nghỉ phép", "thử việc", "phụ cấp", "nhân viên"),
    }.items():
        if any(marker in lowered for marker in markers):
            category = candidate
            break
    entities = re.findall(r"\b(?:MFA|VPN|HR|PCCC|AES-?256|\d{4})\b", text, flags=re.IGNORECASE)
    return {"topic": summarize_chunk(text)[:120], "entities": entities, "category": category,
            "language": "vi" if re.search(r"[À-ỹ]", text) else "en"}


def _enrich_single_call(text: str, source: str) -> dict:
    response = _chat(
        """Phân tích đoạn trích và trả về JSON hợp lệ, không markdown:
{"summary":"tối đa 2 câu","questions":["..."],"context":"một câu bối cảnh","metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance|safety|training","language":"vi|en"}}""",
        f"Tài liệu: {source}\n\nĐoạn trích:\n{text}", 400,
    )
    if response:
        try:
            parsed = json.loads(response.removeprefix("```json").removesuffix("```").strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Đoạn trích từ {source or 'tài liệu nội bộ'}.",
        "metadata": extract_metadata(text),
    }


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    methods = methods or ["combined"]
    valid_methods = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - valid_methods
    if unknown:
        raise ValueError(f"Unknown enrichment methods: {', '.join(sorted(unknown))}")
    enriched = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        source = str(chunk.get("metadata", {}).get("source", ""))
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", ""))
            questions = list(result.get("questions", []))
            context_line = str(result.get("context", ""))
            auto_metadata = dict(result.get("metadata", {}))
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            context_line = contextual_prepend(text, source).removesuffix(text).strip() if "contextual" in methods else ""
            auto_metadata = extract_metadata(text) if "metadata" in methods else {}
        enriched_text = f"{context_line}\n\n{text}" if context_line else text
        enriched.append(EnrichedChunk(text, enriched_text, summary, questions,
                                      {**chunk.get("metadata", {}), **auto_metadata}, "+".join(methods)))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)
    return enriched
