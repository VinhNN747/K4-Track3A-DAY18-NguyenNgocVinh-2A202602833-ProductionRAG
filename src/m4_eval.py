from __future__ import annotations

"""Module 4: RAGAS evaluation with a transparent offline fallback."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", value.lower(), flags=re.UNICODE))


def _overlap(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & _tokens(candidate)) / len(reference_tokens)


def _offline_results(questions: list[str], answers: list[str], contexts: list[list[str]],
                     ground_truths: list[str]) -> list[EvalResult]:
    """Produce clearly bounded lexical proxies when RAGAS cannot call an LLM."""
    results = []
    for question, answer, context_list, ground_truth in zip(questions, answers, contexts, ground_truths):
        joined_context = " ".join(context_list)
        relevant_contexts = [context for context in context_list if _overlap(ground_truth, context) > 0]
        results.append(EvalResult(
            question=question,
            answer=answer,
            contexts=context_list,
            ground_truth=ground_truth,
            faithfulness=_overlap(answer, joined_context),
            answer_relevancy=_overlap(question, answer),
            context_precision=len(relevant_contexts) / len(context_list) if context_list else 0.0,
            context_recall=_overlap(ground_truth, joined_context),
        ))
    return results


def _aggregate(per_question: list[EvalResult]) -> dict:
    count = len(per_question)
    return {
        metric: (sum(getattr(result, metric) for result in per_question) / count if count else 0.0)
        for metric in METRICS
    }


def evaluate_ragas(questions: list[str], answers: list[str], contexts: list[list[str]],
                   ground_truths: list[str]) -> dict:
    """Evaluate all four RAGAS metrics, falling back to lexical diagnostics offline."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts, and ground_truths must have the same length")
    if not questions:
        return {**{metric: 0.0 for metric in METRICS}, "per_question": []}

    if OPENAI_API_KEY and os.getenv("RAG_OFFLINE", "").lower() not in {"1", "true", "yes"}:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                         "contexts": contexts, "ground_truth": ground_truths})
            dataframe = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                                    context_precision, context_recall]).to_pandas()
            per_question = [EvalResult(
                question=row["question"], answer=row["answer"], contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            ) for _, row in dataframe.iterrows()]
            return {**_aggregate(per_question), "per_question": per_question}
        except Exception as exc:
            print(f"  RAGAS failed; using offline lexical diagnostics ({exc})")
    else:
        print("  Offline mode or no OPENAI_API_KEY; using lexical diagnostics")

    per_question = _offline_results(questions, answers, contexts, ground_truths)
    return {**_aggregate(per_question), "per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Return the lowest-scoring questions and a diagnosis from the error tree."""
    diagnostic_tree = {
        "faithfulness": ("Câu trả lời có nội dung không được hỗ trợ bởi context.",
                           "Siết prompt chỉ dùng context và giảm temperature."),
        "context_recall": ("Retriever thiếu chunk chứa thông tin cần thiết.",
                           "Điều chỉnh chunking và tăng recall bằng BM25 + dense search."),
        "context_precision": ("Context chứa quá nhiều chunk không liên quan.",
                              "Tăng chất lượng reranking hoặc lọc theo metadata."),
        "answer_relevancy": ("Câu trả lời chưa bám sát câu hỏi.",
                              "Cải thiện answer prompt và query rewrite."),
    }
    analyzed = []
    for result in eval_results:
        scores = {metric: float(getattr(result, metric)) for metric in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": round(sum(scores.values()) / len(scores), 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyzed, key=lambda item: item["score"])[:max(0, bottom_n)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {key: value for key, value in results.items() if key != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(item) for item in results.get("per_question", [])],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")
