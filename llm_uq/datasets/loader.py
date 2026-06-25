from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from ..config import DatasetConfig


def iter_examples(cfg: DatasetConfig) -> Iterator[dict]:
    """Yield dicts with at minimum 'input' and 'gold' keys."""
    if cfg.source == "csv":
        df = pd.read_csv(cfg.name)
        rng = np.random.default_rng(cfg.seed)
        idxs = rng.permutation(len(df))[: cfg.n_samples]
        for i in idxs:
            row = df.iloc[int(i)]
            yield {"input": str(row[cfg.input_col]), "gold": str(row[cfg.output_col]), "_idx": int(i)}
    else:
        from datasets import load_dataset  # noqa: PLC0415

        load_kwargs: dict = {"split": cfg.split}
        if cfg.hf_config:
            ds = load_dataset(cfg.name, cfg.hf_config, **load_kwargs)
        else:
            ds = load_dataset(cfg.name, **load_kwargs)

        rng = np.random.default_rng(cfg.seed)
        idxs = rng.permutation(len(ds))[: cfg.n_samples]
        for i in idxs:
            ex = ds[int(i)]
            yield {"input": str(ex[cfg.input_col]), "gold": ex[cfg.output_col], "_idx": int(i), "_raw": ex}
