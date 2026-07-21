"""Train a struct+fastText stacked RF on the paragraph dump (overlap labels),
evaluate end-to-end through the keep policy, export to JSON for Rust.
"""
import gzip
import json
import math
import random
import statistics
import sys

import numpy as np

SP = "/private/tmp/claude-501/-Users-held-marin-resiliparse/c2512af5-1b64-4fb9-bde2-54e5279b7d33/scratchpad"
sys.path.insert(0, f"{SP}/jusText/benchmark/eval")
from metrics import score_pair, tokenize  # noqa: E402

print("loading...", flush=True)
gold = []
with gzip.open(f"{SP}/jusText/benchmark/datasets/general/train.jsonl.gz", "rt") as f:
    for line in f:
        gold.append(json.loads(line)["final_output"])
paras = [None] * len(gold)
with gzip.open(f"{SP}/rp-runs/train.paragraphs.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        paras[d["idx"]] = d["paragraphs"]

FEATS = [
    "log_len", "log_words", "link_density", "heading", "position", "ends_sentence",
    "avg_word_len", "log_depth", "log_tags", "verbatim",
    "nav", "aside", "header", "footer", "form", "list", "table", "main", "bq",
    "ft", "prev_ft", "next_ft", "prev_log_len", "next_log_len",
    "prev_link", "next_link",
    "punct_ratio", "digit_ratio", "upper_ratio", "starts_bullet", "ends_colon",
    "log_n_paras", "ft_minus_docmean", "doc_max_ft", "ft2_prev", "ft2_next",
]

def base_feats(p, i, n):
    text = p["text"]
    words = text.split()
    ld = p["link_chars"] / max(1, len(text))
    return [
        math.log1p(len(text)), math.log1p(len(words)), ld,
        1.0 if p["heading"] else 0.0, i / max(1, n - 1),
        1.0 if text[-1:] in ".!?" else 0.0,
        len(text) / max(1, len(words)),
        math.log1p(p["depth"]), math.log1p(p["tags"]),
        1.0 if p["verbatim"] else 0.0,
        *[1.0 if p[k] else 0.0 for k in
          ["nav", "aside", "header", "footer", "form", "list", "table", "main", "bq"]],
    ]

LABEL_RE = __import__("re").compile(r"^[A-Za-z][\w /()&.-]{0,25}:\s")

def para_line_stats(p):
    lines = [l for l in p["text"].splitlines() if l.strip()]
    if not lines:
        return 0.0, 0.0, 0.0
    short = sum(1 for l in lines if len(l) < 60) / len(lines)
    label = sum(1 for l in lines if LABEL_RE.match(l.strip())) / len(lines)
    return float(len(lines)), short, label

def doc_features(ps):
    n = len(ps)
    base = [base_feats(p, i, n) for i, p in enumerate(ps)]
    probs = [p["prob"] for p in ps]
    stats = [para_line_stats(p) for p in ps]
    total_chars = sum(len(p["text"]) for p in ps) or 1
    all_lines = [l for p in ps for l in p["text"].splitlines() if l.strip()]
    doc_short = sum(1 for l in all_lines if len(l) < 60) / max(1, len(all_lines))
    doc_label = sum(1 for l in all_lines if LABEL_RE.match(l.strip())) / max(1, len(all_lines))
    doc_hi_ft = sum(1 for x in probs if x > 0.5) / max(1, n)
    rows = []
    for i in range(n):
        prev = base[i - 1] if i > 0 else None
        nxt = base[i + 1] if i < n - 1 else None
        p = ps[i]
        text = p["text"]
        nc = max(1, len(text))
        punct = sum(1 for c in text if not c.isalnum() and not c.isspace()) / nc
        digit = sum(1 for c in text if c.isdigit()) / nc
        upper = sum(1 for c in text if c.isupper()) / nc
        docmean = sum(probs) / max(1, len(probs))
        rows.append(base[i] + [
            probs[i],
            probs[i - 1] if i > 0 else 0.0,
            probs[i + 1] if i < n - 1 else 0.0,
            prev[0] if prev else 0.0, nxt[0] if nxt else 0.0,
            prev[2] if prev else 0.0, nxt[2] if nxt else 0.0,
            punct, digit, upper,
            1.0 if text.startswith(("- ", "1. ")) else 0.0,
            1.0 if text.rstrip().endswith(":") else 0.0,
            __import__("math").log1p(n),
            probs[i] - docmean, max(probs) if probs else 0.0,
            probs[i - 2] if i > 1 else 0.0,
            probs[i + 2] if i < n - 2 else 0.0,
            math.log1p(stats[i][0]), stats[i][1], stats[i][2],
            1.0 if LABEL_RE.match(text.strip()) else 0.0,
            doc_short, doc_label, doc_hi_ft,
            len(text) / total_chars,
            (n - 1 - i) / max(1, n - 1),
            math.log1p(p.get("cls_neg", 0)), math.log1p(p.get("cls_pos", 0)),
            1.0 if p.get("cls_comment") else 0.0,
            1.0 if p.get("cls_ad") else 0.0,
        ])
    return rows

def doc_labels(ps, g):
    gset = set(tokenize(g))
    labels = []
    for p in ps:
        toks = tokenize(p["text"])
        labels.append(1 if toks and sum(t in gset for t in toks) / len(toks) >= 0.6 else 0)
    return labels

print("featurizing...", flush=True)
X, y, doc_of = [], [], []
for i, ps in enumerate(paras):
    if not ps:
        continue
    X.extend(doc_features(ps))
    y.extend(doc_labels(ps, gold[i]))
    doc_of.extend([i] * len(ps))
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
print(f"{len(y)} paragraphs, positive rate {y.mean():.3f}", flush=True)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=60, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=0
)
rf.fit(X, y)
print("trained", flush=True)

# stash per-paragraph stacked probs back onto docs
proba = rf.predict_proba(X)[:, 1]
stack_prob = {}
k = 0
for i, ps in enumerate(paras):
    if not ps:
        stack_prob[i] = []
        continue
    stack_prob[i] = proba[k:k + len(ps)].tolist()
    k += len(ps)

def policy_keep(ps, probs, th):
    n = len(ps)
    keep = [p["verbatim"] or pr >= th for p, pr in zip(ps, probs)]
    for i, p in enumerate(ps):
        if keep[i] and not p["verbatim"]:
            words = len(p["text"].split())
            ld = p["link_chars"] / max(1, len(p["text"]))
            if words < 10 and ld > 0.8:
                keep[i] = False
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not keep[i] and probs[i] >= th * 0.625:
                if (keep[i - 1] if i > 0 else False) and (keep[i + 1] if i + 1 < n else False):
                    keep[i] = True
                    changed = True
    for i, p in enumerate(ps):
        if not keep[i] and p["heading"] and probs[i] >= th * 0.375 and i + 1 < n and keep[i + 1]:
            keep[i] = True
    if ps and not any(keep):
        keep[max(range(n), key=lambda i: probs[i])] = True
    return keep

random.seed(0)
sample = random.sample(range(10000), 3000)

def eval_stack(th):
    f1s, levs = [], []
    for i in sample:
        keep = policy_keep(paras[i], stack_prob[i], th)
        pred = "\n\n".join(p["text"] for p, kk in zip(paras[i], keep) if kk)
        s = score_pair(pred, gold[i])
        f1s.append(s["rougeL_f"])
        levs.append(s["lev_similarity"])
    print(f"stack th={th:.2f}: F1={statistics.fmean(f1s):.4f} lev={statistics.fmean(levs):.4f}")

for th in [0.35, 0.45, 0.5, 0.55, 0.65]:
    eval_stack(th)

# export RF to JSON for the Rust port
print("exporting...", flush=True)
trees = []
for est in rf.estimators_:
    t = est.tree_
    trees.append({
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": [round(float(v), 6) for v in t.threshold],
        "value": [round(float(v[0][1] / max(1e-9, v[0][0] + v[0][1])), 6) for v in t.value],
    })
with open(f"{SP}/rp-runs/stack_rf8.json", "w") as f:
    json.dump({"features": FEATS, "trees": trees}, f)
import joblib
joblib.dump(rf, f"{SP}/rp-runs/stack_rf8.joblib")
print("saved stack_rf8.json / .joblib")
