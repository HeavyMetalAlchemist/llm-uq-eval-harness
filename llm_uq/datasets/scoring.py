from __future__ import annotations

import re
from typing import Any, Optional

_num_re = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_final_re = re.compile(r"(?m)^\s*FINAL:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def _extract_last_number(s: str) -> Optional[str]:
    last = None
    for m in _num_re.finditer(s):
        last = m.group(0)
    return last


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return " ".join(s.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def score(pred_text: str, gold: Any, mode: str, cosine_threshold: float = 0.7) -> int:
    """Return 1 (correct) or 0 (incorrect)."""
    if not pred_text:
        return 0

    if mode == "numeric":
        last_final = None
        for m in _final_re.finditer(pred_text):
            last_final = m
        pred_str = last_final.group(1) if last_final else _extract_last_number(pred_text)

        gold_str = None
        if isinstance(gold, str):
            gold_tail = gold.split("####")[-1].strip()
            gold_str = _extract_last_number(gold_tail)
        else:
            gold_str = str(gold)

        pred_num = _to_float(pred_str)
        gold_num = _to_float(gold_str)
        if pred_num is None or gold_num is None:
            return 0
        return int(abs(pred_num - gold_num) < 1e-9)

    if mode == "exact_match":
        golds = gold if isinstance(gold, list) else [gold]
        pred_norm = _normalize(pred_text)
        return int(any(_normalize(g) == pred_norm for g in golds))

    if mode == "contains":
        golds = gold if isinstance(gold, list) else [str(gold)]
        # handle TriviaQA-style: gold may be a dict with 'value' and 'aliases'
        if isinstance(gold, dict):
            golds = [gold.get("value", "")] + gold.get("aliases", [])
        pred_norm = _normalize(pred_text)
        pred_toks = pred_norm.split()
        for g in golds:
            gold_toks = _normalize(str(g)).split()
            if not gold_toks:
                continue
            for j in range(len(pred_toks) - len(gold_toks) + 1):
                if pred_toks[j : j + len(gold_toks)] == gold_toks:
                    return 1
        return 0

    if mode == "f1":
        golds = gold if isinstance(gold, list) else [str(gold)]
        if isinstance(gold, dict):
            golds = [gold.get("value", "")] + gold.get("aliases", [])
        best = max(_token_f1(pred_text, str(g)) for g in golds)
        return int(best >= 0.5)

    if mode == "cosine":
        from sentence_transformers import SentenceTransformer, util  # noqa: PLC0415
        import torch  # noqa: PLC0415

        _model = _get_embedder()
        golds = gold if isinstance(gold, list) else [str(gold)]
        emb_pred = _model.encode(pred_text, convert_to_tensor=True, normalize_embeddings=True)
        emb_golds = _model.encode([str(g) for g in golds], convert_to_tensor=True, normalize_embeddings=True)
        sim = float(torch.max(util.cos_sim(emb_pred, emb_golds)).item())
        return int(sim >= cosine_threshold)

    raise ValueError(f"Unknown scoring mode: {mode}")


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
