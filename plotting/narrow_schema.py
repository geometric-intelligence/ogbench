"""Expected seed count and canonical model names for plots."""

from __future__ import annotations

import math

EXPECTED_SEEDS = 3

MODEL_NAME_ALIASES: dict[str, str] = {
    "graph_sage": "sage",
    "graphsage": "sage",
    "mlagnn": "gatv4",
}


def canonical_model_name(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, float) and math.isnan(raw):
        return None
    s = str(raw).strip().lower()
    if not s or s in ("nan", "none", "<na>", "nat"):
        return None
    s = s.replace("-", "_").replace(" ", "_")
    return MODEL_NAME_ALIASES.get(s, s)
