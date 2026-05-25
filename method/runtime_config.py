import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_GUARD_ROOT = str(REPO_ROOT / "Token-Guard-main")
DEFAULT_DATASET_NAME = "halueval"
DEFAULT_MODEL_PATH = os.environ.get("STEPGUARD_MODEL_PATH", "Qwen/Qwen3-8B")
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
)

TAU_C = 0.40
TAU_H = 1.55
TOP_K = 3
TRIGGER_MODE = "and"
STEP_ACCEPT_THRESHOLD = 0.55
COOLDOWN_STEPS = 10
MAX_NEW_TOKENS = 3000
MAX_INPUT_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_STEP_REGENS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hi-CoT StepGuard on local datasets or Token-Guard datasets."
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--token_guard_root", type=str, default=DEFAULT_TOKEN_GUARD_ROOT)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--input_data",
        type=str,
        default=None,
        help="Optional explicit dataset path. If omitted, use token_guard_root/data/<dataset_name>.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional explicit output path. Supports .json and .jsonl",
    )
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max_input_tokens", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--tau_c", type=float, default=TAU_C)
    parser.add_argument("--tau_h", type=float, default=TAU_H)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--trigger_mode", type=str, default=TRIGGER_MODE, choices=["and", "or"])
    parser.add_argument("--step_accept_threshold", type=float, default=STEP_ACCEPT_THRESHOLD)
    parser.add_argument("--cooldown_steps", type=int, default=COOLDOWN_STEPS)
    parser.add_argument("--max_step_regens", type=int, default=MAX_STEP_REGENS)
    parser.add_argument("--disable_eadp", action="store_true")
    parser.add_argument("--disable_step_verifier", action="store_true")
    parser.add_argument("--disable_rollback", action="store_true")
    return parser.parse_args()


def resolve_input_path(args):
    if args.input_data:
        return args.input_data
    return os.path.join(args.token_guard_root, "data", f"{args.dataset_name}.json")


def resolve_output_path(args):
    if args.output_path:
        return args.output_path
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    file_name = f"{args.dataset_name}_stepguard_results.json"
    return os.path.join(DEFAULT_OUTPUT_DIR, file_name)


def resolve_token_guard_imports(token_guard_root):
    guard_dir = os.path.join(token_guard_root, "guard")
    if not os.path.isdir(guard_dir):
        raise FileNotFoundError(f"Token-Guard guard directory not found: {guard_dir}")
    if guard_dir not in sys.path:
        sys.path.insert(0, guard_dir)


def build_token_guard_args(model_path, data_path, dataset_name):
    class _Args:
        pass

    args = _Args()
    args.model_id = "shared_hf"
    args.model_path = model_path
    args.gpus = 1
    dataset_key = (dataset_name or "").strip().lower()
    args.datasets = "covidQA" if dataset_key == "covidqa" else dataset_key
    args.data_path = data_path
    args.output_dir = "./results/"
    args.step_beam_size = 1
    args.num_rollout = 1
    args.num_foresight = 4
    args.strategy = "cluster"
    args.width_pruning_strategy = "low_sigma"
    args.depth_pruning_strategy = "cluster"
    args.cluster_num = 2
    args.threshold = 0.75
    args.least_foresight_num = 1
    args.sigma_rate = 0.8
    args.record_process = False
    args.file_name = "stepguard"
    args.time_path = "./results/time/"
    args.seed = 0
    args.max_examples = -1
    args.shot_mode = "zeroshot"
    args.tau_global = None
    return args
