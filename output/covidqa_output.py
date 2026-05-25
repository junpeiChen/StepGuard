import re

from answer_canonicalizer import AnswerNormalizer


def clean_response_text(text):
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(<\|im_end\|>|<\|eot_id\|>|</s>)", "", cleaned)
    cleaned = re.sub(r"<\|[^>\n]+\|>", "", cleaned)
    return cleaned.strip()


def clean_answer_text(text):
    cleaned = clean_response_text(text)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().strip('"').strip("'").strip()


class LightweightAnswerNormalizer(AnswerNormalizer):
    """Official-prompt-compatible extraction only.

    This normalizer intentionally avoids dataset-specific semantic rewriting.
    It only extracts the final Answer span and removes generation artifacts.
    """

    def extract_answer(self, response):
        cleaned = clean_response_text(response)
        matches = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.*?)\s*$", cleaned)
        if matches:
            return clean_answer_text(matches[-1])

        loose_matches = list(re.finditer(r"(?i)(?:final\s+)?answer\s*:\s*", cleaned))
        if loose_matches:
            start = loose_matches[-1].end()
            tail = cleaned[start:].strip()
            first_line = next((line.strip() for line in tail.splitlines() if line.strip()), "")
            return clean_answer_text(first_line)

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return clean_answer_text(lines[-1]) if lines else ""

    def normalize_surface_form(self, answer):
        return clean_answer_text(answer)

    def refine_answer(self, question, passage, raw_answer):
        return self.normalize_surface_form(raw_answer)

    def finalize_response(self, question, passage, response):
        cleaned = clean_response_text(response)
        answer = self.extract_answer(cleaned)
        if not answer:
            return cleaned.rstrip()

        lines = cleaned.rstrip().splitlines()
        while lines and re.match(r"(?im)^\s*(?:final\s+)?answer\s*:\s*.*$", lines[-1].strip()):
            lines.pop()
        base = "\n".join(lines).rstrip()
        if base:
            return f"{base}\nAnswer: {answer}"
        return f"Answer: {answer}"


def build_answer_normalizer():
    return LightweightAnswerNormalizer()
