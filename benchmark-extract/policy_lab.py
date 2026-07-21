"""Offline keep-policy experiments over the Rust paragraph dump.

Usage: from policy_lab import *; then eval_policy(fn) where fn(paras) -> [bool].
Each para dict: text, prob, verbatim, heading, depth, link_chars, tags,
nav/aside/header/footer/form/list/table/main/bq flags.
"""
import gzip
import json
import statistics
import sys

SP = "/private/tmp/claude-501/-Users-held-marin-resiliparse/c2512af5-1b64-4fb9-bde2-54e5279b7d33/scratchpad"
sys.path.insert(0, f"{SP}/jusText/benchmark/eval")
from metrics import score_pair  # noqa: E402

_GOLD = None
_PARAS = None


def load(split="train"):
    global _GOLD, _PARAS
    _GOLD = []
    with gzip.open(f"{SP}/jusText/benchmark/datasets/general/{split}.jsonl.gz", "rt") as f:
        for line in f:
            _GOLD.append(json.loads(line)["final_output"])
    _PARAS = [None] * len(_GOLD)
    with gzip.open(f"{SP}/rp-runs/{split}.paragraphs.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            _PARAS[d["idx"]] = d["paragraphs"]
    print(f"loaded {len(_GOLD)} docs ({split})")


def eval_policy(policy, sample=None, verbose=True):
    """policy(paras: list[dict]) -> list[bool]. Returns (mean_f1, mean_lev)."""
    idxs = range(len(_GOLD)) if sample is None else sample
    f1s, levs = [], []
    for i in idxs:
        keep = policy(_PARAS[i])
        pred = "\n\n".join(p["text"] for p, k in zip(_PARAS[i], keep) if k)
        s = score_pair(pred, _GOLD[i])
        f1s.append(s["rougeL_f"])
        levs.append(s["lev_similarity"])
    mf, ml = statistics.fmean(f1s), statistics.fmean(levs)
    if verbose:
        print(f"F1={mf:.4f} lev={ml:.4f} (n={len(f1s)})")
    return mf, ml


def per_doc(policy, sample=None):
    idxs = range(len(_GOLD)) if sample is None else sample
    out = []
    for i in idxs:
        keep = policy(_PARAS[i])
        pred = "\n\n".join(p["text"] for p, k in zip(_PARAS[i], keep) if k)
        s = score_pair(pred, _GOLD[i])
        out.append((i, s["rougeL_f"], s["rougeL_p"], s["rougeL_r"], s["lev_similarity"]))
    return out


def baseline(th):
    def policy(paras):
        return [p["verbatim"] or p["prob"] >= th for p in paras]
    return policy


SAMPLE = None  # set to a list of idxs for fast iteration


if __name__ == "__main__":
    load("train")
    import random
    random.seed(0)
    sample = random.sample(range(10000), 3000)
    for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        print(f"th={th:.2f}: ", end="")
        eval_policy(baseline(th), sample)
