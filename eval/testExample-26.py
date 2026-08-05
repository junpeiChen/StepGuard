import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
STEPGUARD_ROOT = SCRIPT_DIR.parent
DEFAULT_TOKEN_GUARD_ROOT = STEPGUARD_ROOT / "Token-Guard-main"
RUN_STEPGUARD_PATH = STEPGUARD_ROOT / "method" / "run_stepguard.py"
OFFICIAL_EVAL_PATH = SCRIPT_DIR / "run_official_eval.py"

TEST_NAME = "testExample-26"
MODEL_PATH = os.environ.get("STEPGUARD_MODEL_PATH", "")
MAX_EXAMPLES = "26"
MAX_NEW_TOKENS = "3000"
MAX_INPUT_TOKENS = "4096"
OUTPUT_DIR = SCRIPT_DIR / TEST_NAME


DATASETS = [
    {
        "display_name": "CovidQA",
        "dataset_name": "covidQA",
        "input_file": "CovidQA.json",
        "output_file": "CovidQA_testExample-26_results.json",
    },
    {
        "display_name": "DROP_History",
        "dataset_name": "history",
        "input_file": "DROP_History.json",
        "output_file": "DROP_History_testExample-26_results.json",
    },
    {
        "display_name": "DROP_Nfl",
        "dataset_name": "nfl",
        "input_file": "DROP_Nfl.json",
        "output_file": "DROP_Nfl_testExample-26_results.json",
    },
    {
        "display_name": "FinanceBench",
        "dataset_name": "financebench",
        "input_file": "FinanceBench.json",
        "output_file": "FinanceBench_testExample-26_results.json",
    },
    {
        "display_name": "Halueval",
        "dataset_name": "halueval",
        "input_file": "Halueval.json",
        "output_file": "Halueval_testExample-26_results.json",
    },
    {
        "display_name": "PubmedQA",
        "dataset_name": "pubmedqa",
        "input_file": "PubmedQA.json",
        "output_file": "PubmedQA_testExample-26_results.json",
    },
    {
        "display_name": "RAGTruth",
        "dataset_name": "ragtruth",
        "input_file": "RAGTruth.json",
        "output_file": "RAGTruth_testExample-26_results.json",
    },
]


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_command(dataset, output_path):
    return [
        sys.executable,
        str(RUN_STEPGUARD_PATH),
        "--dataset_name",
        dataset["dataset_name"],
        "--token_guard_root",
        str(DEFAULT_TOKEN_GUARD_ROOT),
        "--input_data",
        str(STEPGUARD_ROOT / "data" / dataset["input_file"]),
        "--output_path",
        str(output_path),
        "--model_path",
        MODEL_PATH,
        "--max_examples",
        MAX_EXAMPLES,
        "--max_new_tokens",
        MAX_NEW_TOKENS,
        "--max_input_tokens",
        MAX_INPUT_TOKENS,
    ]


def load_official_evaluator():
    eval_runner = load_module("stepguard_official_eval_runner_test_26", OFFICIAL_EVAL_PATH)
    process_py = DEFAULT_TOKEN_GUARD_ROOT / "eval" / "processed_answer" / "process.py"
    eval_py = DEFAULT_TOKEN_GUARD_ROOT / "eval" / "eval.py"
    process_module = eval_runner.load_module("official_process_module_test_26", process_py)
    eval_module = eval_runner.load_module("official_eval_module_test_26", eval_py)
    return eval_runner, process_module.extract_simple_answer, eval_module.get_metrics


def evaluate_result(eval_runner, extract_simple_answer, get_metrics, result_path):
    raw_records = eval_runner.load_records(result_path)
    processed_records = eval_runner.convert_records(raw_records, extract_simple_answer)
    return eval_runner.evaluate_processed_records(processed_records, get_metrics)


def save_summary(rows):
    summary_path = OUTPUT_DIR / f"{TEST_NAME}_summary.json"
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


def print_results_table(rows):
    print(f"\n=== {TEST_NAME} Results ===")
    print(f"{'Dataset':<18} {'EM':>8} {'F1':>8} {'BLEU':>8} {'ROUGE-L':>8} {'N':>6} {'Time(s)':>10} {'s/ex':>10}")
    print("-" * 95)
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['dataset']:<18} "
            f"{metrics['EM']:>8.4f} "
            f"{metrics['F1']:>8.4f} "
            f"{metrics['BLEU']:>8.4f} "
            f"{metrics['ROUGE-L']:>8.4f} "
            f"{metrics['count']:>6} "
            f"{row['runtime_seconds']:>10.2f} "
            f"{row['avg_time_per_example']:>10.4f}"
        )


def run_all_datasets():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_runner, extract_simple_answer, get_metrics = load_official_evaluator()
    rows = []

    for dataset in DATASETS:
        output_path = OUTPUT_DIR / dataset["output_file"]
        cmd = build_command(dataset, output_path)

        print(f"\n=== Running {dataset['display_name']} ({MAX_EXAMPLES} examples) ===")
        print(" ".join(cmd))

        start_time = time.perf_counter()
        subprocess.run(cmd, cwd=str(STEPGUARD_ROOT), check=True)
        elapsed = round(time.perf_counter() - start_time, 4)

        if not output_path.exists():
            print(f"[WARN] Missing result file, skipped evaluation: {output_path}")
            continue

        metrics = evaluate_result(eval_runner, extract_simple_answer, get_metrics, output_path)
        count = metrics.get("count", 0)
        avg_time = round(elapsed / count, 6) if count else 0.0
        rows.append(
            {
                "test_name": TEST_NAME,
                "dataset": dataset["display_name"],
                "dataset_name": dataset["dataset_name"],
                "input_path": str(STEPGUARD_ROOT / "data" / dataset["input_file"]),
                "result_path": str(output_path),
                "runtime_seconds": elapsed,
                "avg_time_per_example": avg_time,
                "metrics": metrics,
            }
        )

        save_summary(rows)
        print_results_table(rows)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not MODEL_PATH or not Path(MODEL_PATH).is_dir():
        raise SystemExit(
            "STEPGUARD_MODEL_PATH must point to a local model directory. "
            f"Current value: {MODEL_PATH!r}"
        )
    run_all_datasets()


if __name__ == "__main__":
    main()
