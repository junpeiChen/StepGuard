import importlib.util
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEPGUARD_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_TOKEN_GUARD_ROOT = os.path.join(STEPGUARD_ROOT, "Token-Guard-main")
DEFAULT_INPUT_DATA = os.path.join(STEPGUARD_ROOT, "data", "DROP_History.json")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "results", "DROP_History_stepguard_results.json")
INTERFERE_PATH = os.path.join(STEPGUARD_ROOT, "method", "run_stepguard.py")


def load_inference_module():
    method_dir = os.path.dirname(INTERFERE_PATH)
    if method_dir not in sys.path:
        sys.path.insert(0, method_dir)
    spec = importlib.util.spec_from_file_location("drop_history_inference_core", INTERFERE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load inference module from: {INTERFERE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_default_flag(flag, value):
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


def main():
    ensure_default_flag("--dataset_name", "history")
    ensure_default_flag("--token_guard_root", DEFAULT_TOKEN_GUARD_ROOT)
    ensure_default_flag("--input_data", DEFAULT_INPUT_DATA)
    ensure_default_flag("--output_path", DEFAULT_OUTPUT_PATH)
    ensure_default_flag("--max_new_tokens", "3000")
    ensure_default_flag("--max_input_tokens", "4096")
    inference = load_inference_module()
    inference.main()


if __name__ == "__main__":
    main()
