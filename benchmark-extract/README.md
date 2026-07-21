# benchmark-extract — main-content extraction eval kit (untracked)

Artifacts from the overnight hill-climb of `resiliparse-rs`'s new
`parse/html/extract.rs` module against the Marin extractor benchmark
(gs://marin-us-central2/datasets/extractor_eval_set/medium_12k). See
`HILLCLIMB.md` for the full iteration log and data findings.

## Result (general split, mean ROUGE-L F1 / Levenshtein similarity)

| system | train (10k) | dev (1k) | speed |
|---|---|---|---|
| resiliparse-rs `inner_text` (before) | 0.703 / 0.558 | 0.707 / 0.561 | 0.4 ms/doc |
| **resiliparse-rs extract (after)** | **0.885 / 0.823** | **0.885 / 0.822** | ~2 ms/doc (1 thread) |
| jusText fork best committed dev run | — | 0.880 / 0.814 | ~350 ms/doc |
| stock jusText | 0.767 / — | 0.762 / 0.682 | 5.7 ms/doc |

Test split touched once (final table in HILLCLIMB.md); not re-run after the structured-page changes.

## Pieces

- `stack_rf8.json` — champion stacked classifier: RandomForest 60×d14 (sklearn export)
  over `extract::stack_features` (45 struct+text+neighbor+doc-structure features incl. the
  fastText keep-probability from MichaelR207/justext-classifier `general_ft.bin`).
  Use with `extract::DecisionForest` + `KeepPolicy { threshold: 0.60,
  neighbour_threshold: 0.375, heading_threshold: 0.225, nav_threshold: 0.0, .. }`.
- `rp-bench/` — the eval runner (parse → score → policy via
  `extract::extract_main_text`, plus dataset dump / debug modes). Path-depends on
  `../../resiliparse-rs`; needs vcpkg on PATH to build.
- `score_rp.py` — scores predictions with the jusText benchmark's exact metrics.
- `train_stack8.py` — retrains the champion from a paragraph dump (`rp-bench --mode dump`).
- `policy_lab.py` — offline keep-policy experiments over a dump.
- `wmb_convert.py` / `wmb_rouge5.py` — WebMainBench (opendatalab) external eval.

## Reproduce an eval

```bash
cd benchmark-extract/rp-bench && cargo build --release
FT=~/.cache/justext/models--MichaelR207--justext-classifier/snapshots/*/general_ft.bin
./target/release/rp-bench dev.jsonl.gz out.predictions.jsonl \
    --mode para --ft $FT --stack ../stack_rf8.json --threshold 0.60
python3 ../score_rp.py dev.jsonl.gz out.predictions.jsonl out
```
