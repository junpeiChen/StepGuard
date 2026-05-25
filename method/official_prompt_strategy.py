"""Official-style prompt templates for StepGuard.

This module keeps dataset-specific prompt wording aligned with Token-Guard's
official PromptBuilder while appending only the Hi-CoT instruction/execution
scaffold required by StepGuard. It does not change StepGuard's verification,
dynamic pruning, or local rollback mechanisms.
"""


HICOT_SCAFFOLD = """

Please reason hierarchically by alternating between <|instruction|> and <|execution|>.
Use at most 2 step pairs unless absolutely necessary.
Keep each execution concise and grounded in the given information.

You MUST start with an instruction and follow this format:
<|instruction|> Step 1: [your plan]
<|execution|> Step 1: [your grounded analysis]
  <|instruction|> Step 2: [your next plan]
  <|execution|> Step 2: [your grounded analysis]

The last line must use this exact format:
Answer: [your answer here]
"""

DATASET_ALIASES = {
    "drop_history": "history",
    "history": "history",
    "drop_nfl": "nfl",
    "nfl": "nfl",
    "financebench": "financebench",
    "halueval": "halueval",
    "pubmedqa": "pubmedqa",
    "covidqa": "covidQA",
    "covidQA": "covidQA",
    "ragtruth": "ragtruth",
}


def _load_few_shot_examples():
    try:
        from logic_example import (
            HISTORY_8_FEW_SHOT,
            NFL_8_FEW_SHOT,
            covidQA_4_FEW_SHOT,
            financebench_5_FEW_SHOT,
            halueval_6_FEW_SHOT,
            pubmedQA_4_FEW_SHOT,
            RAGTruth_5_FEW_SHOT,
        )
    except Exception:
        return {}

    return {
        "history": HISTORY_8_FEW_SHOT,
        "nfl": NFL_8_FEW_SHOT,
        "covidQA": covidQA_4_FEW_SHOT,
        "financebench": financebench_5_FEW_SHOT,
        "halueval": halueval_6_FEW_SHOT,
        "pubmedqa": pubmedQA_4_FEW_SHOT,
        "ragtruth": RAGTruth_5_FEW_SHOT,
    }


def canonical_dataset_name(dataset_name=""):
    key = (dataset_name or "").strip()
    return DATASET_ALIASES.get(key, DATASET_ALIASES.get(key.lower(), key.lower()))


def official_zero_shot_prompt(dataset_name=""):
    dataset_key = canonical_dataset_name(dataset_name)
    prompts = {
        "pubmedqa": (
            "You will be given a PubMed-style passage and a Yes/No/Maybe question.\n"
            "Answer rules:\n"
            "1. Begin with exactly one of: \"Yes.\" / \"No.\" / \"Maybe.\"\n"
            "2. Summarize the main conclusion from the passage in exactly ONE short sentence (<=25 words).\n"
            "3. Preserve key phrases and medical terms from the passage; do not replace them with synonyms.\n"
            "4. Always include explicitly stated conditions, subgroups, or limitations if they appear in the conclusion.\n"
            "5. Do NOT add recommendations, explanations, or new information.\n"
            "6. The final output must be exactly ONE LINE:\n"
            "Answer:[Yes./No./Maybe. + short sentence]"
        ),
        "financebench": (
            "You are an equity research analyst. Answer the question using **only the data provided**. Follow these instructions carefully:\n\n"
            "1. Always produce a single-line final answer.\n"
            "2. Do not show calculations, reasoning, or commentary.\n"
            "3. Match the exact format of the ground truth:\n"
            "   - \"$360000.00\" for USD thousands\n"
            "   - \"$7223.00\" for USD millions\n"
            "   - \"$4.90\" for USD billions\n"
            "   - \"34.7%\" for percentages\n"
            "   - \"1.08\" for ratios\n"
            "4. If the answer is not directly available from the statements, output:\n"
            "   \"Unable to answer based on given data.\"\n\n"
            "**Example:**\n"
            "Q: How much was Boeing's FY2017 interest expense (USD thousands)?\n"
            "A: Answer: $360000.00\n"
            "At the end, output: Answer:[your answer here]."
        ),
        "halueval": (
            "You will be presented with a question.\n"
            "Answer the user's question strictly based on the given information.\n"
            "Do not make up information.\n"
            "At the end, output: Answer:[your answer here]."
        ),
        "history": (
            "You will be presented with a question.\n"
            "Answer the user's question strictly based on the given information.\n"
            "Do not make up information.\n"
            "At the end, output: Answer:[your answer here]."
        ),
        "nfl": (
            "You will be presented with a question.\n"
            "Answer the user's question strictly based on the given information.\n"
            "Do not make up information.\n"
            "At the end, output: Answer:[your answer here]."
        ),
        "ragtruth": (
            "You are given passages and a question. Follow these steps:\n"
            "Answer the question using only the information from the given passages.\n"
            " - Include specific examples, numbers, or comparisons if mentioned.\n"
            " - Include all details that support the answer.\n"
            " - Do not add external information.\n"
            " - If the passages do not contain sufficient information, answer: \"Unable to answer based on given passages.\"\n"
            "At the end, output: Answer:[your answer here]"
        ),
        "covidQA": (
            "You are a biomedical QA assistant. Answer using ONLY the provided passage.\n"
            "Rules:\n"
            "1. Copy the EXACT PHRASE or SENTENCE from the passage that answers the question.\n"
            "2. NEVER answer yes/no questions with just 'Yes.' or 'No.' - always include the specific text.\n"
            "   Wrong: 'No.'  Correct: 'the current strategy... is not sufficient because...'\n"
            "3. For list questions (viruses, methods, etc.): include ALL items from the passage.\n"
            "4. Use the FULL NAME as written (e.g., 'Human metapneumovirus (HMPV)' not 'HMPV').\n"
            "5. Do NOT paraphrase or summarize - copy exact wording.\n"
            "6. Your FINAL output line must be EXACTLY: Answer:[copied text]"
        ),
    }

    return prompts.get(
        dataset_key,
        "You will be presented with a question.\n"
        "Answer the user's question strictly based on the given information.\n"
        "Do not make up information.\n"
        "At the end, output: Answer:[your answer here].",
    )


def official_prompt(dataset_name="", shot_mode="fewshot", include_hicot=True):
    """Return official-style dataset prompt, optionally with StepGuard Hi-CoT scaffold.

    CovidQA follows Token-Guard's official behavior and always uses zeroshot,
    because few-shot examples consume too much context for long medical passages.
    """
    dataset_key = canonical_dataset_name(dataset_name)
    base_prompt = official_zero_shot_prompt(dataset_key)

    if dataset_key != "covidQA" and shot_mode == "fewshot":
        few_shot = _load_few_shot_examples().get(dataset_key, "")
        if few_shot:
            base_prompt = (
                f"{base_prompt}\n\n"
                "I will give you some examples for reference:\n"
                f"{few_shot}"
            )

    if include_hicot:
        scaffold = HICOT_SCAFFOLD
        return base_prompt + scaffold
    return base_prompt


def build_system_prompt(dataset_name="", shot_mode="fewshot"):
    return official_prompt(dataset_name=dataset_name, shot_mode=shot_mode, include_hicot=True)
