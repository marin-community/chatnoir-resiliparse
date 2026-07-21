"""Score rp-bench predictions against benchmark gold using the jusText harness metrics.

Usage: python3 score_rp.py <dataset.jsonl.gz> <predictions.jsonl> <out_prefix>
Writes <out_prefix>.metrics.jsonl and <out_prefix>.summary.json (same shape as the
jusText benchmark's committed runs, so numbers are directly comparable).
"""
import gzip
import json
import statistics
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-held-marin-resiliparse/c2512af5-1b64-4fb9-bde2-54e5279b7d33/scratchpad/jusText/benchmark/eval")
from metrics import score_pair  # noqa: E402

dataset_path, pred_path, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]

gold = []
with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            gold.append(json.loads(line))

preds = {}
with open(pred_path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        preds[row["idx"]] = row

assert len(preds) == len(gold), f"{len(preds)} predictions vs {len(gold)} gold docs"

metric_rows = []
elapsed = []
n_errors = 0
for i, rec in enumerate(gold):
    row = preds[i]
    if row.get("error"):
        n_errors += 1
    scores = score_pair(row["prediction"], rec.get("final_output", ""))
    metric_rows.append({
        "idx": i,
        "warc_record_id": rec.get("warc_record_id"),
        "url": rec.get("url"),
        "snapshot": rec.get("snapshot"),
        **scores,
    })
    elapsed.append(row["elapsed_ms"])

with open(f"{out_prefix}.metrics.jsonl", "w", encoding="utf-8") as f:
    for row in metric_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def agg(key):
    vals = [r[key] for r in metric_rows]
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }

summary = {
    "tag": out_prefix.rsplit("/", 1)[-1],
    "extractor": "resiliparse-rs body().inner_text()",
    "dataset": dataset_path,
    "n_docs": len(gold),
    "n_errors": n_errors,
    "timing": {
        "mean_ms_per_doc": statistics.fmean(elapsed),
        "median_ms_per_doc": statistics.median(elapsed),
    },
    "metrics": {k: agg(k) for k in ["rougeL_f", "rougeL_p", "rougeL_r", "lev_distance", "lev_similarity"]},
}
with open(f"{out_prefix}.summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

m = summary["metrics"]
print(f"{summary['tag']}: n={len(gold)} errors={n_errors} "
      f"rougeL_f={m['rougeL_f']['mean']:.4f} lev_sim={m['lev_similarity']['mean']:.4f} "
      f"{summary['timing']['mean_ms_per_doc']:.2f} ms/doc")
