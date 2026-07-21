"""Convert WebMainBench jsonl -> rp-bench input format (html + final_output),
English subset. Usage: python3 wmb_convert.py <in.jsonl> <out.jsonl.gz>
"""
import gzip
import json
import sys

n_in = n_out = 0
with open(sys.argv[1], encoding="utf-8") as f, gzip.open(sys.argv[2], "wt", encoding="utf-8") as out:
    for line in f:
        n_in += 1
        d = json.loads(line)
        meta = d.get("meta") or {}
        if isinstance(meta, str):
            try:
                import ast
                meta = ast.literal_eval(meta)
            except Exception:
                meta = {}
        lang = (meta.get("language") or d.get("language") or "").lower()
        html = d.get("html") or ""
        gold = d.get("groundtruth_content") or d.get("convert_main_content") or ""
        if not html or not gold:
            continue
        # English filter: explicit language field if present, else ASCII-ratio heuristic
        if lang:
            if not lang.startswith("en"):
                continue
        else:
            letters = [c for c in gold[:2000] if c.isalpha()]
            if letters and sum(1 for c in letters if ord(c) < 128) / len(letters) < 0.9:
                continue
        out.write(json.dumps({
            "html": html,
            "final_output": gold,
            "url": d.get("url"),
            "warc_record_id": d.get("track_id"),
            "snapshot": str(meta.get("level")),
        }, ensure_ascii=False) + "\n")
        n_out += 1
print(f"{n_in} in -> {n_out} english out")
