import importlib.util
import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "advance" / "halueval" / "results" / "Halueval_stepguard_results.json"
DEFAULT_OFFICIAL_REPO = REPO_ROOT / "Token-Guard-main"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved StepGuard results with official metrics.")
    parser.add_argument("--result_path", type=str, default=str(RESULT_PATH))
    parser.add_argument("--official_repo", type=str, default=str(DEFAULT_OFFICIAL_REPO))
    return parser.parse_args()


def ensure_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "unknown package"
        raise RuntimeError(
            f"Failed to import dependency '{missing_name}' required by {file_path}.\n"
            f"Please install the official evaluation dependencies first, for example:\n"
            f"  pip install -r {DEFAULT_OFFICIAL_REPO}/requirements.txt"
        ) from exc
    return module


def load_records(path: Path):
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_answer_from_text(extract_simple_answer, text: str):
    return extract_simple_answer(text or "").strip()


def clean_generation_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("</s>", "")
    text = text.strip()
    text = re.sub(r"<\|[^>\n]+\|>", "", text)
    return text.strip()


def convert_records(records, extract_simple_answer):
    processed = []
    for item in records:
        response = clean_generation_text(item.get("response", item.get("model_output", "")))
        answer = extract_answer_from_text(extract_simple_answer, response)
        if "ground_truth" in item and item.get("ground_truth") is not None:
            ground_truth = item.get("ground_truth")
        elif "right_answer" in item and item.get("right_answer") is not None:
            ground_truth = item.get("right_answer")
        elif "answer" in item and item.get("answer") is not None:
            ground_truth = item.get("answer")
        else:
            ground_truth = ""

        processed.append(
            {
                "id": item.get("id"),
                "passage": item.get("passage", item.get("knowledge", "")),
                "question": item.get("question", ""),
                "ground_truth": ground_truth,
                "answer": answer,
            }
        )
    return processed


def print_diagnostics(records, response_records):
    response_blank = sum(1 for item in response_records if not str(item.get("answer", "")).strip())

    print("\n=== Extraction Diagnostics ===")
    print(f"Raw records: {len(records)}")
    print(f"Blank answers from response extraction: {response_blank}")


def evaluate_processed_records(records, get_metrics):
    em_total = 0.0
    f1_total = 0.0
    bleu_total = 0.0
    rouge_total = 0.0
    count = 0

    for item in records:
        pred = item.get("answer", "")
        gold = item.get("ground_truth", "")
        em, f1, bleu, rouge_l = get_metrics(pred, gold)
        em_total += em
        f1_total += f1
        bleu_total += bleu
        rouge_total += rouge_l
        count += 1

    if count == 0:
        return {"count": 0, "EM": 0.0, "F1": 0.0, "BLEU": 0.0, "ROUGE-L": 0.0}

    return {
        "count": count,
        "EM": round(em_total / count, 4),
        "F1": round(f1_total / count, 4),
        "BLEU": round(bleu_total / count, 4),
        "ROUGE-L": round(rouge_total / count, 4),
    }


def print_summary(name: str, summary: dict):
    print(
        f"{name:<18} "
        f"EM={summary['EM']:.4f} "
        f"F1={summary['F1']:.4f} "
        f"BLEU={summary['BLEU']:.4f} "
        f"ROUGE-L={summary['ROUGE-L']:.4f} "
        f"N={summary['count']}"
    )


def main():
    args = parse_args()
    result_path = Path(args.result_path)
    official_repo = Path(args.official_repo)
    process_py = official_repo / "eval" / "processed_answer" / "process.py"
    eval_py = official_repo / "eval" / "eval.py"

    ensure_exists(result_path, "Result file")
    ensure_exists(process_py, "Official process.py")
    ensure_exists(eval_py, "Official eval.py")

    process_module = load_module("official_process_module_eval_only", process_py)
    eval_module = load_module("official_eval_module_eval_only", eval_py)
    extract_simple_answer = process_module.extract_simple_answer
    get_metrics = eval_module.get_metrics

    raw_records = load_records(result_path)
    response_records = convert_records(raw_records, extract_simple_answer)

    out_dir = SCRIPT_DIR / "outputs"
    response_processed_path = out_dir / "processed_response_extract.jsonl"

    write_jsonl(response_processed_path, response_records)

    response_metrics = evaluate_processed_records(response_records, get_metrics)
    summary = {
        "paths": {
            "result_path": str(result_path),
            "response_processed_path": str(response_processed_path),
        },
        "metrics": {
            "response_extract": response_metrics,
        },
    }

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nResult path: {result_path}")
    print_diagnostics(raw_records, response_records)

    print("\n=== Official Evaluation ===")
    print_summary("ResponseExtract", response_metrics)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
