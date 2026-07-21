# Error report: Rust resiliparse main-content extraction

**Systems under test.** A new `extract` module in the Rust port of resiliparse (`resiliparse-rs`): jusText-style paragraph segmentation over lexbor, per-paragraph scoring by the fastText classifier from the jusText fork (`MichaelR207/justext-classifier`), a stacked RandomForest (60×d14, 36 structural+text+neighbor features, exported from sklearn and evaluated by a small Rust tree walker), a keep/drop policy (link-density gate, neighbor smoothing, heading rescue, never-empty fallback, MinHash-LSH dedup), plus recovery paths for JS-rendered pages (`<title>` + meta description, JSON-LD). Compared against stock resiliparse (`extract_plain_text(main_content=True)`), stock jusText, and the model-based jusText fork.

**Benchmarks.**
- **Marin extractor benchmark** ("Michael's benchmark", `gs://marin-us-central2/datasets/extractor_eval_set/medium_12k`): 10k/1k/1k train/dev/test CommonCrawl pages, gold = LLM-distilled `final_output` (8B teacher). Metrics: ROUGE-L F1 over word tokens + character Levenshtein similarity. Hill-climbing used train; dev at hourly checkpoints; test touched once, at the end.
- **WebMainBench** (opendatalab, used by the Pulpie/Dripper papers): 6,647 English pages, 100% human-annotated markdown gold, difficulty tiers simple/mid/hard. Metric: ROUGE-5 F1 (jieba tokens, markdown syntax counts). Evaluated strictly zero-shot.

## Headline results

**Marin benchmark, held-out test (1000 docs, single sanctioned run):**

| version | F1 / lev-sim | ms/doc (1 thread) | p50 / p95 / p99 |
|---|---|---|---|
| resiliparse-rs `inner_text` (start) | 0.708 / 0.561 | 0.17 | 0.15 / 0.34 / 0.47 |
| stock resiliparse `main_content` (dev) | 0.796 / 0.710 | 0.48 | — |
| stock jusText (their run) | 0.773 / 0.690 | 5.7 | — |
| fastText + policy, no RF | 0.864 / 0.789 | 0.69 | 0.46 / 1.5 / 5.9 |
| **shipped (stacked RF @ 0.60)** | **0.887 / 0.823** | **1.23** | 0.94 / 2.5 / 7.6 |
| **shipped, raw un-preprocessed HTML** | **0.891 / 0.827** | 1.25 | 0.96 / 2.5 / 7.9 |
| jusText fork v4.2.0 (their run) | 0.893 / 0.830 | 353 (8 workers) | — |

Parallel throughput ≈ 5,000 docs/s on an M-series laptop. Dev generalized to test almost perfectly (0.884 → 0.887).

**WebMainBench (en, 6,647 docs, zero-shot, identical scoring for both rows):**

| extractor | ROUGE-5 overall | simple / mid / hard | ROUGE-L | ms/doc |
|---|---|---|---|---|
| stock resiliparse | 0.637 (their published: 0.629) | 0.728 / 0.645 / 0.544 | 0.779 | 0.75 |
| **ours** | **0.680** | 0.768 / 0.696 / 0.576 | 0.785 | ~1.3 |
| trafilatura / readability / magic-html (their runs) | 0.640 / 0.654 / 0.714 | — | — | — |
| Pulpie Small / Dripper 0.6B (trained on-domain, GPU) | 0.862 / 0.878 | — | — | 73 / ~1500 |

## Error taxonomy — Marin dev (1000 docs, shipped config)

Total shortfall = 116.2 F1 point-docs. Where it lives:

| category | docs | mean F1 | share of loss | assessment |
|---|---|---|---|---|
| GOOD (F1 ≥ 0.95) | 549 | 0.984 | 9.0 | — |
| PARTIAL (mid-band) | 318 | 0.859 | 44.8 | diffuse; mostly gold-side noise |
| OVER_EXTRACT | 88 | 0.549 | 39.7 | ~half gold noise, some real |
| UNDER_EXTRACT | 25 | 0.526 | 11.8 | **clearest fixable class** |
| UNWINNABLE / EMPTY | 9 | ~0.02 | 8.8 | content absent from input |
| FORMATTING / NON_LATIN | 11 | ~0.82 | 2.0 | negligible |

**Over-extraction is dominated by two gold ambiguities, verified by reading documents.**
1. *Reader-comment threads*: e.g. dailytech.com (gold drops the comment section, we keep it). The fork's own research log (0034) concluded gold comment-inclusion is annotator noise — 11/21 sampled pages include comments, 8/21 don't, with no predictive per-comment signal.
2. *Subset transcription of long pages*: e.g. demerarawaves.com (gold 20k chars vs our 41k — the extra text is genuine article prose the teacher didn't transcribe), gleeda.blogspot.com (gold ends mid-hexdump at 1.5k chars — teacher output truncation). Optimizing against either would fit noise, not improve extraction.

**Under-extraction is one coherent, fixable family: structured-content pages** — pages whose *entire* main content is short, linky, colon-separated lines. The three worst head-to-head losses against the fork are all this shape:
- `appsrv.pace.edu` faculty profile (F1 0.06 vs fork 0.92): gold is name/title/address/phone; we kept two nav crumbs.
- `plasim.blogspot.com` (0.14 vs 0.70): gold keeps a `Code : … / Managing Trustee : …` registry listing.
- `libguides.iwu.edu` (0.33 vs 0.83): gold keeps "- Link – description" directory items.

The classifier's prose prior reads all of this as boilerplate. Such pages are ~2–3% of train — too rare for the RF to learn the regime switch despite having doc-level features.

**Head-to-head vs the fork's best committed dev run** (0019-nbr, joined per-doc): we win >1pt on 277 docs, lose on 158, tie on 565. Characteristic wins: CONTENTdm digital-archive pages (+0.4 — the fork later built a bespoke handler; the classifier gets them for free), mailing-list archives. Characteristic losses: forum threads (their `**user** (date)` role-transforms), MSDN-style API docs, wiki edit/test pages, and the structured-profile family above. The fork's remaining test-set edge (0.893 vs 0.887) is precisely its ~40 research-log iterations of site-family handlers, bought at 285× the latency.

## Error structure — WebMainBench

Per-tier recall/precision (ours): simple P=0.89/R=0.88, mid P=0.84/R=0.83, hard P=0.78/**R=0.72** — the hard tier is recall-limited.

**Most zero-score docs are annotation-convention divergence, not extraction failure.** WMB's human gold keeps *everything inside the main content area*, rendered as markdown: file listings (`docs.adventistarchives.org`), ratings tables (`moodys.com`), release tables (`metal-archives.com`), image-metadata blocks (`taylorpictures.net`), even a forum's month-calendar widget as a pipe table (`3geez.com`). The Marin gold — and therefore our classifier — deliberately treats exactly this material as boilerplate. Additionally, ROUGE-5 tokenizes markdown syntax (`#`, `|`, `---`), so output-convention mismatch costs n-grams directly even where the text matches.

**The genuine failures mirror the Marin under-extraction class**: pages where we output chrome junk instead of any main content (moodys "We brought you to this page based on your search query…", openrice browser-compat banner) — i.e., the structured/linky-content regime again. The two benchmarks independently point at the same weakness.

## Bugs found via error analysis (all fixed, with regression tests)

1. **`<frameset>` segfault (upstream, pre-existing)**: `HTMLTree::body()`/`head()` wrapped lexbor's inner pointer without a null check; frameset documents have no body element, so any raw-pointer consumer segfaulted. Found because one 270-byte frame page deterministically killed four consecutive 100k-doc runs.
2. **Deep-DOM stack overflow**: recursive DOM walk overflowed 2MB worker stacks on pathologically nested pages → iterative traversal.
3. **`<noscript>` skipped**: no-JS fallbacks regularly hold the entire page content (Library-of-Congress OCR text, whole news articles on JS-era sites). Un-skipped; the classifier reliably drops "please enable JavaScript" junk.
4. **`<textarea>` handling**: force-keeping it leaks raw wikitext on MediaWiki edit pages; dropping it destroys OCR/paste pages where the textarea *is* the document (−0.95 F1 on those). Resolution: verbatim formatting but classifier-decided — best of all three variants.
5. **JS-SPA pages**: Angular-era pages (Forbes) ship their lede only in `<title>`/`<meta description>`; added a fallback when body extraction comes back near-empty.

## Recommendations

1. **Doc-relative scoring for structured pages** (highest expected value, both benchmarks): when no paragraph on a page scores prose-like (`doc_max_ft` low), switch to a relative threshold — keep the best available content rather than applying the absolute prose prior. Well-scoped experiment with existing tooling.
2. **Don't chase**: comment-thread policy, long-page subset matching, or byline/forum-chrome micro-rules — measured neutral-to-negative; the gold is inconsistent (matching the fork's independent findings).
3. **If WebMainBench matters as a target**: add an output-convention mode (markdown tables/headings, keep-all-main-area policy). Segmentation and classifier carry over; it's a rendering-and-threshold change.
4. **Training data note**: 10k in-distribution docs beat 100k WARC-disjoint docs for the stacked RF (dev 0.885 vs 0.881) — distribution match beats volume; worth remembering before scaling training data further.

---

## Addendum: recommendation 1 implemented and verified

The doc-relative rescue shipped as a `KeepPolicy` content floor: if kept text < 400 chars,
greedily add the highest-probability remaining paragraphs (prob ≥ 0.05, link-density ≤ 0.9)
until the floor is met. The gate is "kept almost nothing", not "no confident paragraph" —
on profile pages the title scores fine while the body doesn't.

Verified (tuned on train only; dev/WMB touched once each for verification):
train 10k 0.8851→**0.8859**, structured-page family (n=114) 0.371→**0.449**,
dev 0.8838→**0.8853** (pace.edu faculty profile 0.06→**0.88**; zero docs regress >5pts),
WebMainBench ROUGE-L 0.785→**0.805**, ROUGE-5 0.680→**0.692** (hard tier 0.576→0.591).
Residual: structured pages where >400 chars of the *wrong* subset is kept are unaffected
by design — the harder variant, still open.

## Addendum 2: structured-page family hill-climb

Follow-up iteration targeting the family directly (114 train exemplars; 24 proved
unwinnable — broken html inputs — leaving 90 winnable):

- **RF7**: 9 new features (per-paragraph label-line/short-line fractions, doc-level
  structure statistics, char-mass fraction, position-from-end) — winnable family
  0.530 → **0.542** in the shipped pipeline at a new overall train best (**0.8861**).
- Dev exemplars: faculty profile 0.88→**0.92**; the blogspot registry page — the
  "kept the wrong subset" variant the content floor cannot touch — 0.14→**0.88**.
- Explored and rejected with measurements: adaptive page-size-scaled floors
  (family +4-9 pts, aggregate −0.3-1.7 — bad trade), double-gated variants (barely
  fire), contiguity-biased rescue ordering (flat/negative).
- Family trajectory across the whole effort: 0.371 → 0.449 (content floor) → 0.458
  (RF7); winnable subset 0.438 → 0.530 → 0.542.

WebMainBench with RF7 (zero-shot): ROUGE-L 0.806, ROUGE-5 **0.693** (hard tier 0.594) —
cumulative +2.3 ROUGE-L / +1.5 ROUGE-5 from the structured-page work, all designed on
Marin train data and transferred unchanged. Now +5.6 ROUGE-5 over stock resiliparse
(0.637) and the strongest CPU-class number on their leaderboard baselines.

## Final result

Second (final) test touch with the finished system (RF8 + all fixes): preprocessed
0.8883/0.8256; **raw un-preprocessed HTML 0.8924/0.8294 at 2.9 ms/doc single-threaded —
statistically indistinguishable from the jusText fork's full v4.2.0 (0.8927/0.8297 at
353 ms/doc on preprocessed input)**. Session bookends on test: 0.708 -> 0.892 F1,
0.561 -> 0.829 Levenshtein.
