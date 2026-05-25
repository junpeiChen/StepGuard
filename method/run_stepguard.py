import gc
import copy
import math
import os
import re
import time

import torch
from transformers import LogitsProcessor
from answer_canonicalizer import (
    AnswerNormalizer,
    clean_response_text,
    load_dataset_normalizer,
    normalize_text,
)
from runtime_config import (
    build_token_guard_args,
    parse_args,
    resolve_input_path,
    resolve_output_path,
    resolve_token_guard_imports,
)
from data_io import load_and_normalize_dataset, load_existing_results, save_results
from official_prompt_strategy import build_system_prompt


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def parse_hicot_steps(text):
    pattern = re.compile(r"(<\|instruction\|>|<\|execution\|>)")
    chunks = pattern.split(text or "")
    current_tag = None
    pending_instruction = ""
    steps = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk in {"<|instruction|>", "<|execution|>"}:
            current_tag = chunk
            continue
        if current_tag == "<|instruction|>":
            pending_instruction = chunk
        elif current_tag == "<|execution|>":
            steps.append({"instruction": pending_instruction, "execution": chunk})
            pending_instruction = ""

    return steps


ANSWER_NORMALIZER = AnswerNormalizer()


def configure_answer_normalizer(dataset_name):
    global ANSWER_NORMALIZER
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ANSWER_NORMALIZER = load_dataset_normalizer(dataset_name, repo_root=repo_root)


def extract_answer_from_response(text):
    return ANSWER_NORMALIZER.extract_answer(text)


def is_invalid_answer_candidate(text):
    return ANSWER_NORMALIZER.is_invalid_candidate(text)


def normalize_final_answer_text(answer):
    return ANSWER_NORMALIZER.normalize_surface_form(answer)


def compress_binary_answer(answer):
    return ANSWER_NORMALIZER.compress_binary_answer(answer)


def find_case_insensitive_span(text, query):
    return ANSWER_NORMALIZER.find_case_insensitive_span(text, query)


def find_prefixed_span(text, query, prefixes):
    if prefixes != AnswerNormalizer.PREFIXED_SPAN_HINTS:
        for prefix in prefixes:
            candidate = f"{prefix} {query}"
            span = ANSWER_NORMALIZER.find_case_insensitive_span(text, candidate)
            if span:
                return span.strip()
        return ""
    return ANSWER_NORMALIZER.find_prefixed_span(text, query)


def map_event_answer_to_question_option(question, answer):
    return ANSWER_NORMALIZER.map_event_answer_to_question_option(question, answer)


def convert_number_words_to_numeric(answer):
    return ANSWER_NORMALIZER.convert_number_words_to_numeric(answer)


def normalize_scaled_numeric_answer(question, answer):
    return ANSWER_NORMALIZER.normalize_scaled_numeric_answer(question, answer)


def refine_answer_with_passage(question, passage, raw_answer):
    return ANSWER_NORMALIZER.refine_answer(question, passage, raw_answer)


def finalize_response(question, passage, response):
    return ANSWER_NORMALIZER.finalize_response(question, passage, response)


def lexical_overlap_score(instruction_text, execution_text):
    instruction_tokens = set(normalize_text(instruction_text))
    execution_tokens = set(normalize_text(execution_text))
    if not instruction_tokens or not execution_tokens:
        return 0.0
    return len(instruction_tokens & execution_tokens) / len(instruction_tokens | execution_tokens)


def get_system_prompt(dataset_name=""):
    """Compatibility wrapper for older scripts that import this function."""
    return build_system_prompt(dataset_name)


def preprocess_covidqa_passage(passage, question="", max_chars=12000):
    passage = passage or ""
    if len(passage) <= max_chars:
        return passage

    abstract = ""
    abs_idx = passage.find("Abstract:")
    text_idx = passage.find("Text:")
    if abs_idx >= 0:
        end = text_idx if text_idx > abs_idx else min(len(passage), abs_idx + 2000)
        abstract = passage[abs_idx:end].strip()

    text_body = passage[text_idx:] if text_idx >= 0 else passage
    budget = max_chars - len(abstract) - 10
    if budget <= 1000:
        return passage[:max_chars]

    anchor = text_body[:600]
    remaining_budget = budget - len(anchor) - 2
    question_words = set(re.findall(r"\b\w{4,}\b", (question or "").lower()))
    rest = text_body[600:]
    paragraphs = [
        para.strip()
        for para in re.split(r"\n{2,}|\n(?=[A-Z\-\d])", rest)
        if para.strip() and len(para.strip()) > 40
    ]
    if not paragraphs:
        paragraphs = [
            sent.strip()
            for sent in re.split(r"(?<=[.!?])\s+", rest)
            if sent.strip() and len(sent.strip()) > 30
        ]

    scored_indices = sorted(
        range(len(paragraphs)),
        key=lambda i: len(question_words & set(re.findall(r"\b\w{4,}\b", paragraphs[i].lower()))),
        reverse=True,
    )
    selected = []
    used = 0
    for idx in scored_indices:
        para = paragraphs[idx]
        if used + len(para) + 2 <= remaining_budget:
            selected.append((idx, para))
            used += len(para) + 2
        if used >= remaining_budget:
            break

    selected.sort(key=lambda x: x[0])
    relevant_text = anchor + "\n\n" + "\n\n".join(para for _, para in selected)
    return (abstract + "\n\n" + relevant_text if abstract else relevant_text).strip()


def build_user_prompt(item, dataset_name=""):
    # Keep the question before the long passage so right-side truncation cannot
    # silently remove the task and make different examples look identical.
    knowledge = item["knowledge"]
    if (dataset_name or "").strip().lower() == "covidqa":
        knowledge = preprocess_covidqa_passage(knowledge, item["question"])
    return (
        f"Question: {item['question']}\n"
        f"Passage: {knowledge}\n"
        f"Question: {item['question']}"
    )


def clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None

    cloned = []
    for layer in past_key_values:
        if layer is None:
            cloned.append(None)
            continue

        if isinstance(layer, (tuple, list)):
            cloned_layer = []
            for t in layer:
                if t is None:
                    cloned_layer.append(None)
                else:
                    cloned_layer.append(t.detach().clone())
            cloned.append(tuple(cloned_layer))
        else:
            cloned.append(layer.detach().clone())

    return tuple(cloned)


class HiCoTStepStateMachine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.inst_token_ids = tokenizer.encode("<|instruction|>", add_special_tokens=False)
        self.exec_token_ids = tokenizer.encode("<|execution|>", add_special_tokens=False)
        self.think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
        self.think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.reset()

    def reset(self):
        self.inside_think = False
        self.mode = None
        self.current_instruction_start = None
        self.current_instruction_text = ""
        self.current_execution_start = None
        self.accepted_steps = []
        self.pending_snapshot = False

    def snapshot(self):
        return {
            "inside_think": self.inside_think,
            "mode": self.mode,
            "current_instruction_start": self.current_instruction_start,
            "current_instruction_text": self.current_instruction_text,
            "current_execution_start": self.current_execution_start,
            "accepted_steps": copy.deepcopy(self.accepted_steps),
            "pending_snapshot": self.pending_snapshot,
        }

    def restore(self, state):
        self.inside_think = state["inside_think"]
        self.mode = state["mode"]
        self.current_instruction_start = state["current_instruction_start"]
        self.current_instruction_text = state["current_instruction_text"]
        self.current_execution_start = state["current_execution_start"]
        self.accepted_steps = copy.deepcopy(state["accepted_steps"])
        self.pending_snapshot = state["pending_snapshot"]

    @staticmethod
    def _ends_with_token_pattern(buffer_ids, pattern_ids):
        if not pattern_ids or len(buffer_ids) < len(pattern_ids):
            return False
        return buffer_ids[-len(pattern_ids):] == pattern_ids

    def on_prompt_ready(self, sequence_ids):
        self._replay_sequence(sequence_ids)

    def _replay_sequence(self, sequence_ids):
        self.reset()
        buffer_ids = []
        for pos, token_id in enumerate(sequence_ids):
            buffer_ids.append(token_id)
            if len(buffer_ids) > 64:
                buffer_ids = buffer_ids[-64:]

            if self._ends_with_token_pattern(buffer_ids, self.think_start_ids):
                self.inside_think = True
                self.mode = None
                continue

            if self._ends_with_token_pattern(buffer_ids, self.think_end_ids):
                self.inside_think = False
                continue

            if self._ends_with_token_pattern(buffer_ids, self.inst_token_ids):
                self.mode = "instruction"
                self.current_instruction_start = pos + 1
                self.current_execution_start = None
                continue

            if self._ends_with_token_pattern(buffer_ids, self.exec_token_ids):
                if self.current_instruction_start is not None:
                    end_idx = pos + 1 - len(self.exec_token_ids)
                    inst_ids = sequence_ids[self.current_instruction_start:end_idx]
                    self.current_instruction_text = self.tokenizer.decode(
                        inst_ids,
                        skip_special_tokens=False,
                    ).strip()
                self.mode = "execution"
                self.current_execution_start = pos + 1
                self.pending_snapshot = True

    def on_token_appended(self, sequence_ids):
        token_id = sequence_ids[-1]
        buffer_ids = sequence_ids[-64:]
        event = None

        if self._ends_with_token_pattern(buffer_ids, self.think_start_ids):
            self.inside_think = True
            self.mode = None
            return None

        if self._ends_with_token_pattern(buffer_ids, self.think_end_ids):
            self.inside_think = False
            return None

        if self._ends_with_token_pattern(buffer_ids, self.exec_token_ids):
            if self.current_instruction_start is not None:
                end_idx = len(sequence_ids) - len(self.exec_token_ids)
                inst_ids = sequence_ids[self.current_instruction_start:end_idx]
                self.current_instruction_text = self.tokenizer.decode(
                    inst_ids,
                    skip_special_tokens=False,
                ).strip()
            self.mode = "execution"
            self.current_execution_start = len(sequence_ids)
            self.pending_snapshot = True
            return {"type": "execution_started"}

        if self._ends_with_token_pattern(buffer_ids, self.inst_token_ids):
            if self.mode == "execution" and self.current_execution_start is not None:
                end_idx = len(sequence_ids) - len(self.inst_token_ids)
                exec_ids = sequence_ids[self.current_execution_start:end_idx]
                execution_text = self.tokenizer.decode(
                    exec_ids,
                    skip_special_tokens=False,
                ).strip()
                completed_step = {
                    "instruction": self.current_instruction_text,
                    "execution": execution_text,
                }
                event = {"type": "step_completed", "step": completed_step}
            self.mode = "instruction"
            self.current_instruction_start = len(sequence_ids)
            self.current_execution_start = None
            self.pending_snapshot = False
            return event

        return None

    def finalize_last_step(self, sequence_ids):
        if self.mode == "execution" and self.current_execution_start is not None:
            exec_ids = sequence_ids[self.current_execution_start:]
            execution_text = self.tokenizer.decode(
                exec_ids,
                skip_special_tokens=False,
            ).strip()
            if execution_text:
                return {
                    "instruction": self.current_instruction_text,
                    "execution": execution_text,
                }
        return None


class EntropyAwareDynamicPruningLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        prompt_lengths,
        threshold_c,
        threshold_h,
        top_k,
        trigger_mode,
        cooldown_steps,
    ):
        self.tokenizer = tokenizer
        self.prompt_lengths = [int(length) for length in prompt_lengths]
        self.tau_c = threshold_c
        self.tau_h = threshold_h
        self.top_k = top_k
        self.trigger_mode = trigger_mode.lower()
        self.cooldown_steps = cooldown_steps

        self.inst_token_ids = tokenizer.encode("<|instruction|>", add_special_tokens=False)
        self.exec_token_ids = tokenizer.encode("<|execution|>", add_special_tokens=False)
        self.think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
        self.think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.max_entropy = math.log(max(2, tokenizer.vocab_size))

        self.batch_size = None
        self.processed_lengths = None
        self.is_executing = None
        self.inside_think = None
        self.cooldown_counter = None
        self.intervention_count = None
        self.current_step_stats = None
        self.completed_step_stats = None

    @staticmethod
    def _ends_with_token_pattern(buffer_ids, pattern_ids):
        if not pattern_ids or len(buffer_ids) < len(pattern_ids):
            return False
        return buffer_ids[-len(pattern_ids):] == pattern_ids

    @staticmethod
    def _new_step_stats():
        return {
            "token_count": 0,
            "intervention_count": 0,
            "entropy_sum": 0.0,
            "prob_diff_sum": 0.0,
            "max_prob_sum": 0.0,
            "entropy_norm_sum": 0.0,
        }

    def _ensure_batch_state(self, bsz):
        if self.batch_size == bsz:
            return
        self.batch_size = bsz
        self.processed_lengths = list(self.prompt_lengths[:bsz])
        self.is_executing = [False] * bsz
        self.inside_think = [False] * bsz
        self.cooldown_counter = [0] * bsz
        self.intervention_count = [0] * bsz
        self.current_step_stats = [self._new_step_stats() for _ in range(bsz)]
        self.completed_step_stats = [[] for _ in range(bsz)]

    def _finalize_step(self, sample_idx):
        stats = self.current_step_stats[sample_idx]
        if stats["token_count"] == 0:
            self.current_step_stats[sample_idx] = self._new_step_stats()
            return

        token_count = stats["token_count"]
        self.completed_step_stats[sample_idx].append(
            {
                "token_count": token_count,
                "intervention_count": stats["intervention_count"],
                "avg_entropy": stats["entropy_sum"] / token_count,
                "avg_entropy_norm": stats["entropy_norm_sum"] / token_count,
                "avg_prob_diff": stats["prob_diff_sum"] / token_count,
                "avg_max_prob": stats["max_prob_sum"] / token_count,
                "intervention_rate": stats["intervention_count"] / token_count,
            }
        )
        self.current_step_stats[sample_idx] = self._new_step_stats()

    def _advance_state_machine(self, sample_idx, input_ids):
        start = min(self.processed_lengths[sample_idx], input_ids.shape[0])
        if start >= input_ids.shape[0]:
            return

        new_tokens = input_ids[start:].tolist()
        rolling = input_ids[max(self.prompt_lengths[sample_idx], input_ids.shape[0] - 32):].tolist()
        buffer_ids = rolling[:-len(new_tokens)] if len(new_tokens) < len(rolling) else []

        for token_id in new_tokens:
            buffer_ids.append(token_id)
            if len(buffer_ids) > 32:
                buffer_ids = buffer_ids[-32:]

            if self._ends_with_token_pattern(buffer_ids, self.think_start_ids):
                self.inside_think[sample_idx] = True
                self.is_executing[sample_idx] = False
                continue

            if self._ends_with_token_pattern(buffer_ids, self.think_end_ids):
                self.inside_think[sample_idx] = False
                continue

            if self._ends_with_token_pattern(buffer_ids, self.exec_token_ids):
                self.is_executing[sample_idx] = not self.inside_think[sample_idx]
                continue

            if self._ends_with_token_pattern(buffer_ids, self.inst_token_ids):
                if self.is_executing[sample_idx]:
                    self._finalize_step(sample_idx)
                self.is_executing[sample_idx] = False

        self.processed_lengths[sample_idx] = input_ids.shape[0]

    def finalize_open_steps(self):
        if self.batch_size is None:
            return
        for sample_idx in range(self.batch_size):
            self._finalize_step(sample_idx)

    def get_sample_step_stats(self, sample_idx):
        return list(self.completed_step_stats[sample_idx]) if self.completed_step_stats else []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        bsz = input_ids.shape[0]
        self._ensure_batch_state(bsz)

        probs = torch.softmax(scores, dim=-1)
        top_probs, _ = torch.topk(probs, k=2, dim=-1)
        max_prob = top_probs[:, 0]
        prob_diff = top_probs[:, 0] - top_probs[:, 1]
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        entropy_norm = entropy / self.max_entropy

        for i in range(bsz):
            self._advance_state_machine(i, input_ids[i])

            if self.cooldown_counter[i] > 0:
                self.cooldown_counter[i] -= 1

            if not self.is_executing[i] or self.inside_think[i]:
                continue

            stats = self.current_step_stats[i]
            stats["token_count"] += 1
            stats["entropy_sum"] += entropy[i].item()
            stats["prob_diff_sum"] += prob_diff[i].item()
            stats["max_prob_sum"] += max_prob[i].item()
            stats["entropy_norm_sum"] += entropy_norm[i].item()

            if self.trigger_mode == "or":
                should_prune = (max_prob[i] < self.tau_c) or (entropy[i] > self.tau_h)
            else:
                should_prune = (max_prob[i] < self.tau_c) and (entropy[i] > self.tau_h)

            if self.cooldown_counter[i] > 0 or not should_prune:
                continue

            self.intervention_count[i] += 1
            stats["intervention_count"] += 1
            self.cooldown_counter[i] = self.cooldown_steps

            seq_scores = scores[i : i + 1, :]
            top_values, top_indices = torch.topk(seq_scores, self.top_k, dim=-1)
            pruned_scores = torch.full_like(seq_scores, float("-inf"))
            pruned_scores.scatter_(1, top_indices, top_values)
            scores[i] = pruned_scores.squeeze(0)

        return scores
def evaluate_step_with_token_guard(
    step,
    item,
    prompt_builder,
    tg_scorer,
    system_prompt,
    accepted_steps,
    accept_threshold,
):
    if accept_threshold <= 0:
        return {
            "accepted": True,
            "step_report": {
                "instruction": step["instruction"],
                "execution": step["execution"],
                "alignment_score": 1.0,
                "token_guard_segment_score": 1.0,
                "token_guard_token_mean": 1.0,
                "step_score": 1.0,
            },
        }

    raw_example = item["raw_example"]
    example = {
        "passage": raw_example.get("passage", item["knowledge"]),
        "question": item["question"],
    }
    base_chat = prompt_builder.prepare_chat_template(example, system_prompt)
    accepted_prefix_text = "The reasoning steps are:\n\n"
    for accepted in accepted_steps:
        accepted_prefix_text += (
            f"<|instruction|> {accepted['instruction']}\n"
            f"<|execution|> {accepted['execution']}\n"
        )
    context_chat = list(base_chat)
    context_chat[-1] = {
        "role": "assistant",
        "content": accepted_prefix_text + f"<|instruction|> {step['instruction']}\n<|execution|> ",
    }
    context_text = tg_scorer.tokenizer.apply_chat_template(
        context_chat, tokenize=False
    ).rstrip(tg_scorer.tokenizer.eos_token).rstrip()

    base_context = prompt_builder.preprocess_passage(
        example["passage"],
        prompt_builder.args.datasets,
        question=example["question"],
    )
    init_context = f"{system_prompt}\nPassage: {base_context}\nQuestion: {example['question']}\n"
    h_x = tg_scorer.initialize_anchor(init_context)

    artifacts = tg_scorer.verify_candidates(
        context_text=context_text,
        candidate_texts=[step["execution"]],
        h_x=h_x,
    )
    artifact = artifacts[0] if artifacts else None
    if artifact is None:
        return {
            "accepted": False,
            "step_report": {
                "instruction": step["instruction"],
                "execution": step["execution"],
                "alignment_score": 0.0,
                "token_guard_segment_score": 0.0,
                "token_guard_token_mean": 0.0,
                "step_score": 0.0,
            },
        }

    refined_artifact = tg_scorer.refine_segment(
        context_text=context_text,
        artifact=artifact,
        h_x=h_x,
        max_retries=1,
    )
    alignment_score = lexical_overlap_score(step["instruction"], step["execution"])
    step_score = clamp01(0.75 * refined_artifact.segment_score + 0.25 * alignment_score)
    return {
        "accepted": step_score >= accept_threshold,
        "step_report": {
            "instruction": step["instruction"],
            "execution": step["execution"],
            "alignment_score": round(alignment_score, 4),
            "token_guard_segment_score": round(refined_artifact.segment_score, 4),
            "token_guard_token_mean": round(
                sum(refined_artifact.token_scores) / max(len(refined_artifact.token_scores), 1),
                4,
            ),
            "step_score": round(step_score, 4),
        },
    }


def sample_next_token(logits, tokenizer, temperature, top_p):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs > top_p
    nucleus_mask[..., 1:] = nucleus_mask[..., :-1].clone()
    nucleus_mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(nucleus_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_idx)


def generate_once(model, tokenizer, messages, args, item, prompt_builder, tg_scorer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(model.device)
    sequence_ids = prompt_inputs.input_ids[0].tolist()

    state_machine = HiCoTStepStateMachine(tokenizer)
    state_machine.on_prompt_ready(sequence_ids)

    eadp_guard = EntropyAwareDynamicPruningLogitsProcessor(
        tokenizer=tokenizer,
        prompt_lengths=[len(sequence_ids)],
        threshold_c=args.tau_c,
        threshold_h=args.tau_h,
        top_k=(args.top_k if not args.disable_eadp else tokenizer.vocab_size),
        trigger_mode=args.trigger_mode,
        cooldown_steps=args.cooldown_steps,
    )

    start_time = time.time()
    current_step_regens = 0
    pending_snapshot = None

    with torch.no_grad():
        outputs = model(
            input_ids=prompt_inputs.input_ids,
            attention_mask=prompt_inputs.attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for _ in range(args.max_new_tokens):
            scores = next_logits.clone()
            scores = eadp_guard(
                torch.tensor([sequence_ids], dtype=torch.long, device=model.device),
                scores,
            )
            next_token = sample_next_token(
                scores,
                tokenizer=tokenizer,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            next_token_id = int(next_token.item())
            sequence_ids.append(next_token_id)

            token_tensor = next_token.to(model.device)
            outputs = model(
                input_ids=token_tensor,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            event = state_machine.on_token_appended(sequence_ids)

            if state_machine.pending_snapshot:
                pending_snapshot = {
                    "sequence_ids": list(sequence_ids),
                    "past_key_values": clone_past_key_values(past_key_values),
                    "next_logits": next_logits.detach().clone(),
                    "state_machine": state_machine.snapshot(),
                    "eadp_state": {
                        "batch_size": eadp_guard.batch_size,
                        "processed_lengths": copy.deepcopy(eadp_guard.processed_lengths),
                        "is_executing": copy.deepcopy(eadp_guard.is_executing),
                        "inside_think": copy.deepcopy(eadp_guard.inside_think),
                        "cooldown_counter": copy.deepcopy(eadp_guard.cooldown_counter),
                        "intervention_count": copy.deepcopy(eadp_guard.intervention_count),
                        "current_step_stats": copy.deepcopy(eadp_guard.current_step_stats),
                        "completed_step_stats": copy.deepcopy(eadp_guard.completed_step_stats),
                    },
                }
                state_machine.pending_snapshot = False
                current_step_regens = 0

            if event and event.get("type") == "step_completed":
                evaluation = evaluate_step_with_token_guard(
                    step=event["step"],
                    item=item,
                    prompt_builder=prompt_builder,
                    tg_scorer=tg_scorer,
                    system_prompt=build_system_prompt(args.dataset_name),
                    accepted_steps=state_machine.accepted_steps,
                    accept_threshold=(0.0 if args.disable_step_verifier else args.step_accept_threshold),
                )
                if evaluation["accepted"]:
                    step_to_store = dict(event["step"])
                    step_to_store.update(evaluation["step_report"])
                    state_machine.accepted_steps.append(step_to_store)
                    current_step_regens = 0
                elif (
                    pending_snapshot is not None
                    and not args.disable_rollback
                    and current_step_regens < args.max_step_regens
                ):
                    current_step_regens += 1
                    sequence_ids = list(pending_snapshot["sequence_ids"])
                    past_key_values = clone_past_key_values(pending_snapshot["past_key_values"])
                    next_logits = pending_snapshot["next_logits"].detach().clone()
                    state_machine.restore(pending_snapshot["state_machine"])

                    eadp_state = pending_snapshot["eadp_state"]
                    eadp_guard.batch_size = eadp_state["batch_size"]
                    eadp_guard.processed_lengths = copy.deepcopy(eadp_state["processed_lengths"])
                    eadp_guard.is_executing = copy.deepcopy(eadp_state["is_executing"])
                    eadp_guard.inside_think = copy.deepcopy(eadp_state["inside_think"])
                    eadp_guard.cooldown_counter = copy.deepcopy(eadp_state["cooldown_counter"])
                    eadp_guard.intervention_count = copy.deepcopy(eadp_state["intervention_count"])
                    eadp_guard.current_step_stats = copy.deepcopy(eadp_state["current_step_stats"])
                    eadp_guard.completed_step_stats = copy.deepcopy(eadp_state["completed_step_stats"])
                    continue
                else:
                    step_to_store = dict(event["step"])
                    step_to_store.update(evaluation["step_report"])
                    step_to_store["accepted"] = False
                    state_machine.accepted_steps.append(step_to_store)
                    current_step_regens = 0

            if next_token_id == tokenizer.eos_token_id:
                break

    latency = time.time() - start_time
    generated_part = sequence_ids[len(prompt_inputs.input_ids[0]):]
    response = tokenizer.decode(generated_part, skip_special_tokens=False)

    result = {
        "model_output": response,
        "latency_seconds": round(latency, 2),
    }

    del prompt_inputs, outputs, eadp_guard
    torch.cuda.empty_cache()
    gc.collect()
    return result


def solve_item_with_stepguard(model, tokenizer, item, args, prompt_builder, tg_scorer):
    system_prompt = build_system_prompt(args.dataset_name)
    user_prompt = build_user_prompt(item, args.dataset_name)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return generate_once(model, tokenizer, messages, args, item, prompt_builder, tg_scorer)


def get_record_ground_truth(record):
    for key in ("ground_truth", "right_answer", "answer"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def has_valid_response(record):
    response = str(record.get("response", "")).strip()
    if response:
        return True

    ground_truth = get_record_ground_truth(record)
    return ground_truth is not None and str(ground_truth).strip() == ""


def sync_existing_ground_truths(existing_results, data):
    ground_truth_by_id = {
        str(item["id"]): ("" if item.get("right_answer") is None else item.get("right_answer"))
        for item in data
    }
    synced = []
    for record in existing_results:
        record_id = str(record.get("id"))
        if record_id in ground_truth_by_id:
            ground_truth = ground_truth_by_id[record_id]
            record["ground_truth"] = ground_truth
            if str(ground_truth).strip() == "" and str(record.get("response", "")).strip().lower() == "null":
                record["response"] = ""
        synced.append(record)
    return synced


def build_output_record(item, solved, args):
    raw_example = item["raw_example"]
    passage = raw_example.get("passage", item["knowledge"])
    response = finalize_response(
        question=item["question"],
        passage=passage,
        response=solved["model_output"],
    )
    ground_truth = item["right_answer"]
    if ground_truth is None:
        ground_truth = ""
    if not str(response).strip() and str(ground_truth).strip():
        response = "null"
    return {
        "id": item["id"],
        "question": item["question"],
        "passage": passage,
        "ground_truth": ground_truth,
        "response": response,
    }


def main():
    args = parse_args()
    configure_answer_normalizer(args.dataset_name)
    input_path = resolve_input_path(args)
    output_path = resolve_output_path(args)

    resolve_token_guard_imports(args.token_guard_root)
    from token_guard_plugin import LatentEnvironment, TokenGuardConfig
    from prompt_builder import PromptBuilder

    print(f"Loading model from: {args.model_path}")
    print(f"Reading dataset from: {input_path}")
    print(f"Writing results to: {output_path}")

    tg_config = TokenGuardConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    tg_scorer = LatentEnvironment(
        model_path=args.model_path,
        config=tg_config,
    )
    model = tg_scorer.model
    tokenizer = tg_scorer.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    prompt_builder = PromptBuilder(
        build_token_guard_args(
            model_path=args.model_path,
            data_path=input_path,
            dataset_name=args.dataset_name,
        )
    )

    data = load_and_normalize_dataset(input_path, args.dataset_name, args.max_examples)
    if not data:
        print("No examples found.")
        return

    existing_results = sync_existing_ground_truths(load_existing_results(output_path), data)
    valid_existing_results = [item for item in existing_results if has_valid_response(item)]
    dropped_empty_count = len(existing_results) - len(valid_existing_results)
    completed_ids = {
        str(item.get("id"))
        for item in valid_existing_results
        if item.get("id") is not None
    }
    remaining_data = [item for item in data if str(item["id"]) not in completed_ids]
    results = list(valid_existing_results)

    print(f"Loaded existing records: {len(existing_results)}")
    if dropped_empty_count:
        print(f"Dropped empty existing records for rerun: {dropped_empty_count}")
    print(f"Starting StepGuard run, remaining items: {len(remaining_data)} / total {len(data)}")

    if not remaining_data:
        print("All items are already completed.")
        return

    for offset, item in enumerate(remaining_data):
        solved = solve_item_with_stepguard(model, tokenizer, item, args, prompt_builder, tg_scorer)
        output_record = build_output_record(item, solved, args)
        results.append(output_record)

        extracted_answer = refine_answer_with_passage(
            question=item["question"],
            passage=output_record["passage"],
            raw_answer=extract_answer_from_response(output_record["response"]),
        )
        current_idx = len(existing_results) + offset + 1
        print(f"[{current_idx}/{len(data)}] id={item['id']} Answer: {extracted_answer}")

        save_results(output_path, results)

    print("\nStep-aware Hi-CoT Guard run completed.")


if __name__ == "__main__":
    main()
