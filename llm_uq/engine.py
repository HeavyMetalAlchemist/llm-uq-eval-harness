from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig, PromptConfig

_final_re = re.compile(r"(?m)^\s*FINAL:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
_end_re = re.compile(r"(?m)^\s*END\s*$")


class ModelLoader:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tok = None
        self.model = None

    def load(self) -> "ModelLoader":
        cfg = self.cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        if cfg.quantize_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "quantize_4bit=true requires a CUDA GPU. "
                    "Set quantize_4bit: false in your config to run on CPU (slower, higher memory)."
                )
            try:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantization. "
                    "Install it with: pip install 'llm-uq[gpu]'"
                )
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.name,
                quantization_config=bnb_cfg,
                device_map=cfg.device_map,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.name,
                device_map=cfg.device_map,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        return self


@torch.no_grad()
def generate_response(
    question: str,
    model,
    tok,
    prompt_cfg: PromptConfig,
    model_cfg: ModelConfig,
    math_stopping: bool = False,
) -> dict:
    msgs = []
    system = prompt_cfg.get_system()
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})

    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt_text, return_tensors="pt")
    prompt_ids = enc.input_ids.to(model.device)
    attn_mask = enc.attention_mask.to(model.device)
    n_prompt_tokens = int(prompt_ids.shape[1])

    chunk_size = model_cfg.chunk_size
    max_total_new = model_cfg.max_new_tokens

    total_new = 0
    all_gen_ids: list[int] = []

    chosen_lp: list[float] = []
    entropies: list[float] = []
    margins: list[float] = []
    max_entropy: Optional[float] = None
    max_entropy_pos: Optional[int] = None
    min_margin: Optional[float] = None
    min_margin_pos: Optional[int] = None
    min_chosen_prob: Optional[float] = None
    min_chosen_prob_pos: Optional[int] = None

    gen_buffer = ""
    seen_final = False
    post_final_token_cap = 40
    tokens_after_final = 0

    while total_new < max_total_new:
        out = model.generate(
            input_ids=prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=min(chunk_size, max_total_new - total_new),
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

        full = out.sequences
        new_len = full.shape[1] - prompt_ids.shape[1]
        if new_len <= 0:
            break

        new_ids = full[:, -new_len:]
        all_gen_ids.extend(new_ids[0].tolist())
        total_new += int(new_len)

        for t, logits in enumerate(out.scores):
            tid = new_ids[0, t].item()
            if tid == tok.eos_token_id:
                break
            logp = F.log_softmax(logits[0], dim=-1)
            p = logp.exp()

            c_lp = float(logp[tid].item())
            chosen_lp.append(c_lp)

            H = float(-(p * logp).sum().item())
            entropies.append(H)

            top2, _ = torch.topk(p, k=2)
            margin = float(top2[0].item() - top2[1].item())
            margins.append(margin)

            pos = len(entropies) - 1
            if max_entropy is None or H > max_entropy:
                max_entropy, max_entropy_pos = H, pos
            if min_margin is None or margin < min_margin:
                min_margin, min_margin_pos = margin, pos
            cp = float(p[tid].item())
            if min_chosen_prob is None or cp < min_chosen_prob:
                min_chosen_prob, min_chosen_prob_pos = cp, pos

        if math_stopping:
            tail_text = tok.decode(new_ids[0], skip_special_tokens=True)
            gen_buffer = (gen_buffer + tail_text)[-256:]
            if _final_re.search(gen_buffer):
                seen_final = True
                post_final = gen_buffer[_final_re.search(gen_buffer).end():]
                if _end_re.search(post_final):
                    break
            if seen_final:
                tokens_after_final += int(new_len)
                if _end_re.search(gen_buffer) or tokens_after_final >= post_final_token_cap:
                    break
        else:
            # single-chunk: answers are short, stop after first generation
            break

        prompt_ids = torch.cat([prompt_ids, new_ids], dim=1)
        attn_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)

    gen_text = (
        tok.decode(torch.tensor(all_gen_ids, device=model.device), skip_special_tokens=True)
        if all_gen_ids else ""
    )

    conf = math.exp(sum(chosen_lp) / len(chosen_lp)) if chosen_lp else None
    mean_token_entropy = float(np.mean(entropies)) if entropies else None

    std_logp = None
    std_logp_last_k = None
    if chosen_lp:
        lp_arr = np.asarray(chosen_lp, dtype=float)
        std_logp = float(lp_arr.std())
        K = min(10 if math_stopping else 5, len(lp_arr))
        std_logp_last_k = float(lp_arr[-K:].std())

    return {
        "confidence": conf,
        "mean_token_entropy": mean_token_entropy,
        "max_token_entropy": float(max_entropy) if max_entropy is not None else None,
        "argmax_token_entropy": int(max_entropy_pos) if max_entropy_pos is not None else None,
        "min_token_margin": float(min_margin) if min_margin is not None else None,
        "argmin_token_margin": int(min_margin_pos) if min_margin_pos is not None else None,
        "min_chosen_prob": float(min_chosen_prob) if min_chosen_prob is not None else None,
        "argmin_chosen_prob": int(min_chosen_prob_pos) if min_chosen_prob_pos is not None else None,
        "std_logp": std_logp,
        "std_logp_last_k": std_logp_last_k,
        "n_prompt_tokens": n_prompt_tokens,
        "n_gen_tokens": total_new,
        "gen_text": gen_text,
    }
