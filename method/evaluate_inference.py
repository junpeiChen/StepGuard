import argparse
import json
import os
import re
import string
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inference.py outputs.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to .json or .jsonl result file")
    parser.add_argument("--output", type=str, default=None, help="Optional metrics output path")
    return parser.parse_args()


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if path.lower().endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def get_first_nonempty(record, keys):
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return ""


def normalize_text(s):
    if not s:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def clean_extracted_answer(text):
    if not text:
        return ""

    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    text = re.sub(
        r"(?i)^(the answer is|the extracted answer is|extracted answer:|final answer:|answer:|conclusion:)\s*",
        "",
        text,
    ).strip()
    text = text.replace("**", "").replace('"', "").replace("'", "").strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    concise = lines[0]
    if len(lines) > 1 and len(lines[-1].split()) <= len(concise.split()):
        concise = lines[-1]
    return concise.strip()


def extract_marked_answer(text):
    if not text:
        return ""

    patterns = [
        r"(?im)^\s*Final Answer\s*:\s*(.+?)\s*$",
        r"(?im)^\s*Answer\s*:\s*(.+?)\s*$",
        r"(?im)^\s*Final Response\s*:\s*(.+?)\s*$",
        r"(?im)^\s*Conclusion\s*:\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return clean_extracted_answer(matches[-1])
    return ""


def extract_prediction_text(record):
    explicit_answer = clean_extracted_answer(record.get("final_answer", ""))
    if explicit_answer:
        return explicit_answer

    text = get_first_nonempty(
        record,
        [
            "response",
            "model_output",
            "output",
            "prediction",
            "pred",
            "generated_text",
            "generated_response",
        ],
    )

    marked_answer = extract_marked_answer(text)
    if marked_answer:
        return marked_answer

    steps = record.get("step_verifier_report") or []
    if steps:
        last_execution = steps[-1].get("execution", "").strip()
        marked_step_answer = extract_marked_answer(last_execution)
        if marked_step_answer:
            return marked_step_answer
        cleaned_step_answer = clean_extracted_answer(last_execution)
        if cleaned_step_answer:
            return cleaned_step_answer

    split_steps = re.findall(r"<\|execution\|>(.*?)(?=<\|instruction\|>|$)", text, flags=re.DOTALL)
    if split_steps:
        marked_chunk_answer = extract_marked_answer(split_steps[-1])
        if marked_chunk_answer:
            return marked_chunk_answer
        cleaned_chunk_answer = clean_extracted_answer(split_steps[-1])
        if cleaned_chunk_answer:
            return cleaned_chunk_answer

    return clean_extracted_answer(text)


def extract_ground_truth_text(record):
    gold = get_first_nonempty(
        record,
        [
            "right_answer",
            "ground_truth",
            "best_answer",
            "Best Answer",
            "gold",
            "reference",
            "answer",
        ],
    )
    return str(gold).strip()


def exact_match(prediction, ground_truth):
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    return overlap / len(pred_tokens)


def lcs_length(a, b):
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(records):
    summary = {
        "count": 0,
        "em": 0.0,
        "f1": 0.0,
        "bleu1": 0.0,
        "rouge_l_f1": 0.0,
        "avg_intervention_count": 0.0,
        "avg_rollback_count": 0.0,
        "avg_attempt_count": 0.0,
        "avg_step_accept_rate": 0.0,
        "trajectory_accept_rate": 0.0,
    }

    per_example = []
    accepted_trajectory = 0

    for record in records:
        gold = extract_ground_truth_text(record)
        pred = extract_prediction_text(record)
        step_summary = record.get("step_verifier_summary") or {}

        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        bleu1 = bleu1_score(pred, gold)
        rouge_l = rouge_l_f1(pred, gold)

        summary["count"] += 1
        summary["em"] += em
        summary["f1"] += f1
        summary["bleu1"] += bleu1
        summary["rouge_l_f1"] += rouge_l
        summary["avg_intervention_count"] += record.get("intervention_count", 0)
        summary["avg_rollback_count"] += record.get("rollback_count", 0)
        summary["avg_attempt_count"] += record.get("attempt_count", 1)
        summary["avg_step_accept_rate"] += step_summary.get("step_accept_rate", 0.0)
        accepted = 1 if step_summary.get("trajectory_accepted", False) else 0
        accepted_trajectory += accepted

        per_example.append(
            {
                "id": record.get("id"),
                "question": record.get("question"),
                "gold": gold,
                "prediction": pred,
                "em": round(em, 4),
                "f1": round(f1, 4),
                "bleu1": round(bleu1, 4),
                "rouge_l_f1": round(rouge_l, 4),
                "trajectory_accepted": bool(accepted),
            }
        )

    if summary["count"] > 0:
        count = summary["count"]
        summary["em"] = round(summary["em"] / count, 4)
        summary["f1"] = round(summary["f1"] / count, 4)
        summary["bleu1"] = round(summary["bleu1"] / count, 4)
        summary["rouge_l_f1"] = round(summary["rouge_l_f1"] / count, 4)
        summary["avg_intervention_count"] = round(summary["avg_intervention_count"] / count, 4)
        summary["avg_rollback_count"] = round(summary["avg_rollback_count"] / count, 4)
        summary["avg_attempt_count"] = round(summary["avg_attempt_count"] / count, 4)
        summary["avg_step_accept_rate"] = round(summary["avg_step_accept_rate"] / count, 4)
        summary["trajectory_accept_rate"] = round(accepted_trajectory / count, 4)

    return {"summary": summary, "per_example": per_example}


def default_output_path(predictions_path):
    base, _ = os.path.splitext(predictions_path)
    return base + ".metrics.json"


def main():
    args = parse_args()
    records = load_records(args.predictions)
    report = evaluate(records)

    output_path = args.output or default_output_path(args.predictions)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = report["summary"]
    print(f"Count: {summary['count']}")
    print(f"EM: {summary['em']:.4f}")
    print(f"F1: {summary['f1']:.4f}")
    print(f"BLEU-1: {summary['bleu1']:.4f}")
    print(f"ROUGE-L-F1: {summary['rouge_l_f1']:.4f}")
    print(f"Avg Interventions: {summary['avg_intervention_count']:.4f}")
    print(f"Avg Rollbacks: {summary['avg_rollback_count']:.4f}")
    print(f"Avg Attempts: {summary['avg_attempt_count']:.4f}")
    print(f"Avg Step Accept Rate: {summary['avg_step_accept_rate']:.4f}")
    print(f"Trajectory Accept Rate: {summary['trajectory_accept_rate']:.4f}")
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
