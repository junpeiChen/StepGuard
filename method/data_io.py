import json
import os


def load_dataset(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    if input_path.lower().endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Unsupported dataset structure in {input_path}")


def first_present_value(example, keys):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def normalize_example(example, idx, dataset_name):
    knowledge = example.get("knowledge")
    if knowledge is None:
        knowledge = example.get("passage")
    if knowledge is None:
        knowledge = example.get("context", "")

    question = example.get("question", "")
    right_answer = first_present_value(
        example,
        ("right_answer", "answer", "ground_truth", "best_answer", "Best Answer"),
    )

    normalized = {
        "id": example.get("id", idx),
        "knowledge": knowledge,
        "question": question,
        "right_answer": right_answer,
        "raw_example": example,
    }
    return normalized


def load_and_normalize_dataset(input_path, dataset_name, max_examples):
    raw_data = load_dataset(input_path)
    if max_examples >= 0:
        raw_data = raw_data[:max_examples]
    return [normalize_example(example, idx, dataset_name) for idx, example in enumerate(raw_data)]


def load_existing_results(output_path):
    if not output_path or not os.path.exists(output_path):
        return []

    with open(output_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    if output_path.lower().endswith(".jsonl"):
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def save_results(output_path, results):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.lower().endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
