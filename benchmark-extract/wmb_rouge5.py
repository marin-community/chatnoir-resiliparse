import gzip, json, statistics, sys
import jieba
from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams
SP = "/private/tmp/claude-501/-Users-held-marin-resiliparse/c2512af5-1b64-4fb9-bde2-54e5279b7d33/scratchpad"
preds = {}
for line in open(f"{SP}/rp-runs/wmb.predictions.jsonl"):
    r = json.loads(line)
    preds[r["idx"]] = r["prediction"]
f1s, by_level = [], {}
i = -1
with gzip.open(f"{SP}/rp-runs/wmb_en.jsonl.gz", "rt") as f:
    for line in f:
        i += 1
        d = json.loads(line)
        gold, pred = d["final_output"].strip(), preds[i].strip()
        if not gold and not pred:
            f1 = 1.0
        else:
            gt = _create_ngrams([x for x in jieba.lcut(d["final_output"])], 5)
            pt = _create_ngrams([x for x in jieba.lcut(preds[i])], 5)
            f1 = _score_ngrams(gt, pt).fmeasure
        f1s.append(f1)
        by_level.setdefault(d.get("snapshot"), []).append(f1)
print(f"ROUGE-5 F1 overall: {statistics.fmean(f1s):.4f} (n={len(f1s)})")
for lvl, v in sorted(by_level.items()):
    print(f"  {lvl}: {statistics.fmean(v):.4f} (n={len(v)})")
