import importlib.util
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_TOKEN_GUARD_ROOT = os.path.join(REPO_ROOT, "Token-Guard-main")
DEFAULT_INPUT_DATA = os.path.join(DEFAULT_TOKEN_GUARD_ROOT, "data", "DROP_Nfl.json")
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "advance", "drop_nfl", "results", "DROP_Nfl_stepguard_results.json")
INFERENCE_PATH = os.path.join(REPO_ROOT, "advance", "method", "run_stepguard.py")


def load_inference_module():
    method_dir = os.path.dirname(INFERENCE_PATH)
    if method_dir not in sys.path:
        sys.path.insert(0, method_dir)
    spec = importlib.util.spec_from_file_location("drop_nfl_inference_core", INFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load inference module from: {INFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_default_flag(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main():
    ensure_default_flag("--dataset_name", "nfl")
    ensure_default_flag("--token_guard_root", DEFAULT_TOKEN_GUARD_ROOT)
    ensure_default_flag("--input_data", DEFAULT_INPUT_DATA)
    ensure_default_flag("--output_path", DEFAULT_OUTPUT_PATH)
    ensure_default_flag("--max_new_tokens", "3000")
    ensure_default_flag("--max_input_tokens", "4096")
    inference = load_inference_module()
    inference.main()


if __name__ == "__main__":
    main()
