import re
import importlib.util
import os


def normalize_text(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def clean_response_text(text):
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"<\|[^>\n]+\|>", "", cleaned)
    cleaned = cleaned.replace("</s>", "")
    return cleaned.strip()


def clean_answer_text(text):
    cleaned = clean_response_text(text)
    cleaned = cleaned.strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class AnswerNormalizer:
    """Normalize final answers into dataset-friendly surface forms.

    This module keeps the reasoning trace intact and only standardizes the
    final `Answer:` line so semantically correct outputs are not penalized by
    superficial formatting differences.
    """

    NULL_LIKE_ANSWERS = {
        "null",
        "none",
        "not mentioned",
        "not provided",
        "not stated",
        "not specified",
        "unknown",
        "cannot be determined",
        "cannot determine",
        "unable to determine",
        "unable to answer",
        "not enough information",
    }
    INVALID_PREFIXES = (
        "<|instruction|>",
        "<|execution|>",
        "step ",
        "the reasoning steps are",
        "background knowledge:",
        "question:",
    )
    PREFIXED_SPAN_HINTS = ("via", "by", "for", "to", "from", "with")
    DESCRIPTIVE_PREFIXES = ("what ", "why ", "how ")
    ABSOLUTE_NUMERIC_TRIGGERS = (
        "how many more",
        "by how many",
        "difference",
        "how much more",
        "how much less",
        "increase",
        "decrease",
    )
    NUMBER_WORD_ONES = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    NUMBER_WORD_TENS = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    def extract_answer(self, response):
        cleaned = clean_response_text(response)
        matches = re.findall(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$", cleaned)
        if matches:
            return matches[-1].strip()

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return ""

        candidate = lines[-1]
        if self.is_invalid_candidate(candidate):
            return ""
        return candidate

    def is_invalid_candidate(self, text):
        cleaned = clean_answer_text(text)
        if not cleaned:
            return True
        lowered = cleaned.lower()
        if lowered.startswith(self.INVALID_PREFIXES):
            return True
        if re.fullmatch(r"step\s+\d+\s*:\s*.*", lowered):
            return True
        if cleaned.endswith(":"):
            return True
        return False

    def normalize_surface_form(self, answer):
        cleaned = clean_answer_text(answer)
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered in self.NULL_LIKE_ANSWERS:
            return "null"
        if re.fullmatch(r"the\s+[A-Z].*", cleaned):
            return cleaned.split(" ", 1)[1].strip()
        if re.fullmatch(r"[-+]?\d+(?:\.0+)", cleaned):
            return cleaned.split(".", 1)[0]
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%", cleaned):
            return cleaned[:-1]
        if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", cleaned):
            return cleaned.replace(",", "")
        return cleaned

    def compress_binary_answer(self, answer):
        cleaned = clean_answer_text(answer).lower()
        if re.fullmatch(r"yes[.!]?", cleaned):
            return "yes"
        if re.fullmatch(r"no[.!]?", cleaned):
            return "no"
        return ""

    def find_case_insensitive_span(self, text, query):
        if not text or not query:
            return ""
        start = text.lower().find(query.lower())
        if start < 0:
            return ""
        return text[start : start + len(query)]

    def find_prefixed_span(self, text, query):
        if not text or not query:
            return ""
        for prefix in self.PREFIXED_SPAN_HINTS:
            candidate = f"{prefix} {query}"
            span = self.find_case_insensitive_span(text, candidate)
            if span:
                return span.strip()
        return ""

    def map_event_answer_to_question_option(self, question, answer):
        if not question or not answer:
            return ""

        q = question.strip()
        m = re.match(r"(?is)^(?:what|which)\s+happened\s+first,\s*(.+?),\s+or\s+(.+?)\?\s*$", q)
        if not m:
            return ""

        options = [m.group(1).strip(), m.group(2).strip()]
        answer_tokens = set(normalize_text(answer))
        if not answer_tokens:
            return ""

        best_option = ""
        best_score = 0
        for option in options:
            option_tokens = set(normalize_text(option))
            score = len(answer_tokens & option_tokens)
            if score > best_score:
                best_score = score
                best_option = option

        return best_option if best_score > 0 else ""

    def convert_number_words_to_numeric(self, answer):
        text = clean_answer_text(answer).lower().replace("-", " ")
        if not text or not re.fullmatch(r"[a-z\s]+", text):
            return ""

        tokens = [tok for tok in text.split() if tok != "and"]
        if not tokens:
            return ""

        total = 0
        current = 0
        for token in tokens:
            if token in self.NUMBER_WORD_ONES:
                current += self.NUMBER_WORD_ONES[token]
            elif token in self.NUMBER_WORD_TENS:
                current += self.NUMBER_WORD_TENS[token]
            elif token == "hundred":
                current = max(current, 1) * 100
            elif token == "thousand":
                total += max(current, 1) * 1000
                current = 0
            else:
                return ""
        total += current
        return str(total)

    def normalize_scaled_numeric_answer(self, question, answer):
        cleaned = clean_answer_text(answer)
        lowered_question = clean_answer_text(question).lower()
        match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*(thousand|million|billion)", cleaned.lower())
        if not match:
            return ""

        value = float(match.group(1))
        unit = match.group(2)
        absolute_scale = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}[unit]

        if any(trigger in lowered_question for trigger in self.ABSOLUTE_NUMERIC_TRIGGERS):
            absolute = value * absolute_scale
            if absolute.is_integer():
                return str(int(absolute))
            return format(absolute, "f").rstrip("0").rstrip(".")

        if value.is_integer():
            return str(int(value))
        return format(value, "f").rstrip("0").rstrip(".")

    def refine_answer(self, question, passage, raw_answer):
        answer = clean_answer_text(raw_answer)
        if not answer or self.is_invalid_candidate(answer):
            return ""

        converted_number_word = self.convert_number_words_to_numeric(answer)
        if converted_number_word:
            return converted_number_word

        normalized_answer = self.normalize_surface_form(answer)
        if normalized_answer == "null":
            return "null"

        scaled_numeric_answer = self.normalize_scaled_numeric_answer(question, answer)
        if scaled_numeric_answer:
            return scaled_numeric_answer

        mapped_option = self.map_event_answer_to_question_option(question, answer)
        if mapped_option:
            return mapped_option

        binary_answer = self.compress_binary_answer(answer)
        if binary_answer:
            return binary_answer

        exact_span = self.find_case_insensitive_span(passage or "", answer)
        if exact_span:
            return exact_span.strip()

        answer_no_quotes = answer.strip('"').strip("'").strip()
        exact_span = self.find_case_insensitive_span(passage or "", answer_no_quotes)
        if exact_span:
            return exact_span.strip().strip('"').strip("'").strip()

        prefixed_span = self.find_prefixed_span(passage or "", answer_no_quotes)
        if prefixed_span:
            return prefixed_span.strip().strip('"').strip("'").strip()

        lowered_question = (question or "").strip().lower()
        if lowered_question.startswith(self.DESCRIPTIVE_PREFIXES) and passage:
            for chunk in re.split(r"(?<=[.!?;])\s+|\n+", passage):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if answer_no_quotes.lower() in chunk.lower():
                    chunk_tokens = normalize_text(chunk)
                    answer_tokens = normalize_text(answer_no_quotes)
                    if answer_tokens and len(chunk_tokens) <= max(len(answer_tokens) + 6, 12):
                        return chunk.strip().strip('"').strip("'").strip()

        return self.normalize_surface_form(answer_no_quotes)

    def finalize_response(self, question, passage, response):
        raw_answer = self.extract_answer(response)
        refined_answer = self.refine_answer(question, passage, raw_answer)
        if not refined_answer:
            return response.rstrip()

        cleaned_response = response.rstrip()
        lines = cleaned_response.splitlines()
        while lines and re.match(r"(?im)^\s*Answer\s*:\s*.*$", lines[-1].strip()):
            lines.pop()

        base = "\n".join(lines).rstrip()
        if base:
            return f"{base}\nAnswer: {refined_answer}"
        return f"Answer: {refined_answer}"


def load_dataset_normalizer(dataset_name, repo_root=None):
    dataset_key = (dataset_name or "").strip().lower()
    strategy_map = {
        "financebench": os.path.join("advance", "output", "financebench_output.py"),
        "history": os.path.join("advance", "output", "drop_history_output.py"),
        "drop_history": os.path.join("advance", "output", "drop_history_output.py"),
        "nfl": os.path.join("advance", "output", "drop_nfl_output.py"),
        "drop_nfl": os.path.join("advance", "output", "drop_nfl_output.py"),
        "halueval": os.path.join("advance", "output", "halueval_output.py"),
        "pubmedqa": os.path.join("advance", "output", "pubmedqa_output.py"),
        "ragtruth": os.path.join("advance", "output", "ragtruth_output.py"),
        "covidqa": os.path.join("advance", "output", "covidqa_output.py"),
    }

    relative_path = strategy_map.get(dataset_key)
    if not relative_path:
        return AnswerNormalizer()

    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    strategy_path = os.path.join(repo_root, relative_path)
    if not os.path.exists(strategy_path):
        return AnswerNormalizer()

    spec = importlib.util.spec_from_file_location(f"{dataset_key}_output", strategy_path)
    if spec is None or spec.loader is None:
        return AnswerNormalizer()

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_answer_normalizer"):
        return module.build_answer_normalizer()
    return AnswerNormalizer()
