# Rust resiliparse extraction hill-climb log

Benchmark: jusText fork's general split (train 10k / dev 1k), ROUGE-L F1 + Levenshtein sim
vs LLM-distilled gold. Test split untouched.

| iteration | change | train F1/lev | dev F1/lev | ms/doc |
|---|---|---|---|---|
| 0 | inner_text baseline | 0.7030 / 0.5575 | 0.7065 / 0.5607 | 0.4 |
| 1 | paragraph segmentation + fastText th=0.5 | 0.8314 / 0.7478 | — | 1.9 |
| 2 | + policy (th=0.40, linkgate, navgate, nbr, heading, fallback) | 0.8496 / 0.7715 | — | 0.8 |
| 3 | + noscript un-skip (OCR/noscript-wrapped content) | 0.8506 / 0.7724 | — | 0.9 |
| 4 | + stacked RF (30×d12, struct+ft+neighbors, overlap labels) th=0.55, + Rust dedup | 0.8812 / 0.8169 | **0.8835 / 0.8195** | 2.6 |
| 5 | + meta-description fallback for JS-SPA pages (floor 150 chars) | 0.8813 / 0.8170 | — | 2.6 |
| 6 | + RF2: 36 feats (punct/digit/upper ratio, bullets, doc-level ft stats), 60×d14, th=0.60 | 0.8845 / 0.8221 | — | 2.3 |
| 7 | dedup via MinHash-LSH banding (identical output, bounded worst case) | 0.8845 / 0.8221 | — | 3.0 |
| 8 | RF2 dev checkpoint (hourly #2) | 0.8845 / 0.8221 | **0.8851 / 0.8214** | 3.3 |
| 9 | + data-table rewrite (pipe tables / Label: Value blocks, classifier-gated) | 0.8843 / 0.8219 | — | 2.8 |
| 10 | + meta/JSON-LD fallbacks; dev checkpoint (hourly #3) | — | 0.8845 / 0.8208 | 2.6 |
| 11 | RF4: trained on big_train 100k only — dev checkpoint #4 | 0.8773 / 0.8126 | 0.8808 / 0.8158 | 2.1 |
| 12 | RF5: big_train + train-10k at 5x weight — dev checkpoint #5 | — | 0.8829 / 0.8190 | 1.8 |
| 13 | RF2 remains champion: in-distribution 10k beats 100k disjoint (data match > volume) | 0.8845 / 0.8221 | 0.8851 / 0.8214 | 2-3 |
| 14 | RF6: RF2 recipe retrained on new segmentation (tables) | 0.8846 / 0.8226 | 0.8852 / 0.8216 (@0.55) | 1.9 |
| 15 | textarea = classified (not force-kept, not dropped): wiki-edit leak fixed, OCR pages kept | 0.8851 / 0.8233 | — | 1.9 |
| 16 | FINAL config (RF6 @ th=0.60, all of the above) — dev checkpoint #6 | **0.8851 / 0.8233** | **0.8838 / 0.8206** | 1.9 |
| 17 | FINAL config on RAW HTML dev (no body_strip preprocessing) | — | **0.8889 / 0.8252** | 2.5 |

Byline rescue and forum-chrome gates tested and rejected (neutral/negative — gold is
inconsistent on both, matching the fork's own findings). Model quality plateaued at
~0.885 ± 0.001; remaining tail is gold-side noise (teacher truncation on long pages,
subset selection, unwinnable inputs).

## External check: WebMainBench (opendatalab, human-annotated, en subset n=6647)

Zero-shot with RF2 config: ROUGE-L F1 0.783 (simple 0.862 / mid 0.797 / hard 0.691), 3.2 ms/doc.
Their exact ROUGE-5 (jieba + rouge_score): 0.678 overall (simple 0.767 / mid 0.695 / hard 0.573)
— markdown chars count as tokens in their metric, so convention mismatch caps this number.
(Pulpie's encoder models report 0.862-0.873 ROUGE-5 on this set at 13.7 pages/s on an L4 GPU.)
Their gold keeps ALL main-area content incl. link-heavy listings/calendars as pipe tables —
diverges from Marin gold which drops those; zero-F1 docs are mostly that divergence, not bugs.
Genuine failures to mine: pages where we keep chrome junk instead of any main content
(moodys/openrice pattern) — under-extraction on "hard" tier (R=0.72 < P=0.78).

## Infra notes
- Recursive DOM walk segfaulted on deep pages (2MB rayon stacks) — walker now iterative.
- 100k-doc dump OOM'd twice (buffered rows, then concurrent heavy evals) — now streamed
  input+output, heavy jobs serialized.
- UPSTREAM BUG FOUND+FIXED: `HTMLTree::body()`/`head()` wrapped lexbor's inner pointer
  without null-checking it. A `<frameset>` document (no body element per spec — doc
  72,500 of big_train, a 270-byte AAMR.ORG frame page) yielded an ElementNode holding
  NULL; the DOM trait methods mask it via check_node!, but raw-pointer consumers
  segfault. Fixed in tree.rs (return None) + defensive guard in extract_paragraphs +
  regression test. This crashed ANY frameset page — a real production crasher for the
  WARC pipelines this library targets.
- SIP strips DYLD_* env vars across /usr/bin/time — rp-bench now embeds an rpath instead.

Reference (fork's committed runs, dev): stock justext 0.7619/0.6823; ftstack 0.8703/0.8007;
ftstack-dedup 0.8762/0.8080; 0019-nbr (best) 0.8798/0.8144. v4.2.0 test = 0.8927/0.8297.

Dev evals: session start (inner_text 0.7065), iter-4 (0.8835). Next dev eval due ≥1h after iter-4's.

Key data findings:
- inner_text over-extracts: 80% of pred lines are boilerplate (recall 0.976, precision 0.583).
- `<noscript>` holds real content (LoC OCR pages, JS-era sites serving whole articles in
  noscript fallback) — do not skip it; classifier kills "enable JS" junk.
- ~21/3000 train docs are unwinnable (gold content absent from the input html, e.g. 591-char
  ad-tracker fragments); gold for a few docs is degenerate (literal </div> spam).
- Gold formatting: "- " bullets + "**bold**", almost never "#"; multi-space runs inside table
  rows are preserved in gold but collapsed by our normalizer (small Lev leak, ambiguous fix).

## FINAL: held-out test set (single sanctioned run, 1000 docs)

| version | test F1 / lev | mean ms/doc (1 thread) | p50 / p95 / p99 ms |
|---|---|---|---|
| inner_text | 0.7081 / 0.5610 | 0.17 | 0.15 / 0.34 / 0.47 |
| fastText + policy (no RF) | 0.8636 / 0.7890 | 0.69 | 0.46 / 1.53 / 5.88 |
| + stacked RF2 | 0.8866 / 0.8232 | 1.23 | 0.94 / 2.55 / 7.69 |
| + stacked RF6 (shipped) | 0.8865 / 0.8234 | 1.23 | 0.94 / 2.48 / 7.62 |
| shipped, RAW html input | **0.8909 / 0.8271** | 1.25 | 0.96 / 2.48 / 7.93 |
| stock jusText v3.0.2 (their run) | 0.7729 / 0.6903 | 5.7 (8 workers) | — |
| jusText fork v4.2.0 (their run) | 0.8927 / 0.8297 | 353 (8 workers) | — |

Parallel throughput (all cores, M-series): ~5,000 docs/s.
Note: the "fastText + policy" tier at test time includes all segmentation fixes,
dedup, and fallbacks — it is today's build minus the stacked RF, not the historical
iteration-1 configuration.

## Post-report fix: structured-page rescue (the cross-benchmark finding)

Rule (in `KeepPolicy`): if kept text < 400 chars, greedily add highest-probability
remaining paragraphs (prob >= 0.05, link-density <= 0.9) until the floor is met.
Gate is "kept almost nothing", NOT "no confident paragraph" — on profile pages the
title scores well while the body doesn't, so a max-prob gate never fires.
Designed and tuned on train only (floor swept 150-1500; 400 = best aggregate).

| metric | before | after |
|---|---|---|
| train (10k) | 0.8851 / 0.8233 | **0.8859 / 0.8240** |
| structured-page family in train (n=114) | 0.371 F1 | **0.449 F1** |
| dev (verification) | 0.8838 / 0.8206 | **0.8853 / 0.8223** |
| dev idx 530 (pace.edu faculty profile) | 0.06 | **0.88** |
| WebMainBench ROUGE-L | 0.7849 | **0.8049** |
| WebMainBench ROUGE-5 (their metric) | 0.680 | **0.692** (hard: 0.576 -> 0.591) |

Zero dev docs regress >5pts. Residual: structured pages where we keep the *wrong*
subset (>400 chars of it) are untouched by design — that's the harder variant.
Test set NOT re-run after this change (would be a second touch — owner's call).

## Structured-page family hill-climb (post-fix continuation)

Family = 114 train docs whose gold is mostly short lines and extraction was recall-limited.
24/114 turned out UNWINNABLE (broken inputs: 1,377 U+FFFD chars and no Cyrillic in a Russian
page's html field, 122-byte html, gold text absent) — target is the 90 winnable.

| step | full train | family (114) | winnable family (90) |
|---|---|---|---|
| before rescue | 0.8851 | 0.371 | 0.438 |
| + content-floor rescue (400 chars) | 0.8859 | 0.449 | 0.530 |
| + RF7: 45 feats (label-line/short-line/doc-structure) @ th=0.60 | **0.8861** | **0.458** | **0.542** |

Dev verification (single run): 0.8846/0.8208 (within noise of 0.8853 best); structured
exemplars: faculty profile 0.88→0.92, blogspot registry (the "kept wrong subset" variant
the floor can't touch) 0.14→0.88, libguides 0.33→0.38.

Explored and rejected: adaptive floors scaled to page size (family +4 to +9 pts but
aggregate −0.3 to −1.7 — bad trade), kept-mass+label double gates (barely fire),
contiguity-biased rescue ordering (flat to negative). The floor=400 / prob-order /
RF7-features combination is the Pareto point. Residual winnable-family losses are
directory pages where >400 chars of adjacent-but-wrong content is kept.

## Mojibake repair (reversible class)

`extract::repair_mojibake`: reverses UTF-8-read-as-CP1252 (incl. double-encoded),
signature-gated, segment-wise, strict-decode with per-byte CP1252 fallback (lossless
by construction), accepted only when the signature count drops. Train: mojibake
subset (n=42) 0.8226 -> 0.8302, full train 0.8861 -> 0.8862, ZERO regressions
outside the subset, dev unchanged (0.8846). U+FFFD corruption remains unrepairable
(bytes destroyed upstream) — the fix for those 196 docs is re-extracting the
dataset's html from WARCs with proper encoding detection.

## Class/id token features (RF8) + container-level keep/drop experiments

RF8 = RF7 + 4 ancestor class/id features (log neg-token hits: nav/sidebar/comment/
banner/hidden/...; log pos-token hits: content/article/post/...; comment flag; ad
flag), computed in the walker from class+id attributes with token-boundary matching.

| | full train | dev | train family / winnable |
|---|---|---|---|
| RF7 + mojibake (prev champion) | 0.8862 / 0.8239 | 0.8846 / 0.8208 | 0.458 / 0.542 |
| **RF8 (shipped)** | **0.8867 / 0.8248** | **0.8856 / 0.8221** | 0.459 / 0.544 |

Dev per-doc: 16 helped >5pts vs 5 hurt. New dev best.

Container-level keep/drop: REJECTED at the tested granularity. Paragraphs now carry
container1/container2 ids (top-level and second-level block regions); char-weighted
container-mean-probability rules were all negative — container-rescue -1.5pts sample
(drags whole mixed regions in), container-suppress -0.05. Top-level containers mix
content and chrome on real pages, so region-mean doesn't discriminate. Finer-grained
container identity (innermost block, or DOM-subtree segmentation) is the remaining
open direction; the id plumbing is in place for it.

## FINAL test run #2 (RF8, closes the session)

| config | test F1 / lev | ms/doc (1 thread) |
|---|---|---|
| RF8 @ 0.60, preprocessed input | 0.8883 / 0.8256 | 2.8 |
| **RF8 @ 0.60, RAW html input** | **0.8924 / 0.8294** | 2.9 |
| fork v4.2.0 (their run, preprocessed) | 0.8927 / 0.8297 | 353 (8 workers) |

Raw-HTML parity with the fork's full system: deltas 0.0003/0.0003 — a dead heat,
with no preprocessing stage and ~125x less compute. Dev->test held both touches
(0.884->0.887 at RF6; 0.8856->0.8883 at RF8): no overfitting across the session.
Session bookends: test 0.7081/0.5610 -> 0.8924/0.8294.

## Side-by-side, held-out test (final)

| system | F1 / lev | ms/doc | input |
|---|---|---|---|
| resiliparse-rs inner_text (start) | 0.7081 / 0.5610 | 0.17 | preproc |
| stock jusText v3.0.2 | 0.7729 / 0.6903 | 5.7 (8 wk) | preproc |
| stock resiliparse main_content | 0.8166 / 0.7253 | 0.48 | preproc |
| ours: fastText + policy | 0.8636 / 0.7890 | 0.69 | preproc |
| ours: + stacked RF (RF2) | 0.8866 / 0.8232 | 1.23 | preproc |
| ours: FINAL (RF8) | 0.8883 / 0.8256 | 2.77 | preproc |
| ours: FINAL (RF8) | 0.8924 / 0.8294 | 2.86 | RAW |
| jusText fork v4.2.0 | 0.8927 / 0.8297 | 353 (8 wk) | preproc |

Per-doc vs v4.2.0 on test: 245 wins / 214 losses / 541 ties (raw: 245/210/545) —
we win more docs; their mean edge lives in a few large-margin forum-formatting docs.
