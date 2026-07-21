use fasttext::FastText;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use resiliparse::parse::html::dom::traits::Element;
use resiliparse::parse::html::extract::{
    assemble, extract_main_text, extract_paragraphs, meta_fallback, repair_mojibake,
    stack_features, DecisionForest, DecisionTree, KeepPolicy, Paragraph,
};
use resiliparse::parse::html::tree::HTMLTree;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

#[derive(Deserialize)]
struct Record {
    #[serde(default)]
    html: String,
    #[serde(default)]
    warc_record_id: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    snapshot: Option<String>,
}

struct Config {
    mode: String,
    threshold: f32,
    ft: Option<FastText>,
    forest: Option<DecisionForest>,
    meta_floor: usize,
}

fn load_forest(path: &str) -> DecisionForest {
    let raw: serde_json::Value =
        serde_json::from_reader(BufReader::new(File::open(path).expect("open forest")))
            .expect("parse forest json");
    let trees = raw["trees"]
        .as_array()
        .expect("trees array")
        .iter()
        .map(|t| DecisionTree {
            children_left: serde_json::from_value(t["children_left"].clone()).unwrap(),
            children_right: serde_json::from_value(t["children_right"].clone()).unwrap(),
            feature: serde_json::from_value(t["feature"].clone()).unwrap(),
            threshold: serde_json::from_value(t["threshold"].clone()).unwrap(),
            value: serde_json::from_value(t["value"].clone()).unwrap(),
        })
        .collect();
    DecisionForest { trees }
}

/// Mirror the jusText fork's fastText input preprocessing:
/// whitespace runs -> single space, strip, lowercase, first 1000 chars.
fn ft_input(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(1200));
    let mut in_ws = false;
    for c in text.chars() {
        if c.is_whitespace() {
            in_ws = true;
        } else {
            if in_ws && !out.is_empty() {
                out.push(' ');
            }
            in_ws = false;
            for lc in c.to_lowercase() {
                out.push(lc);
            }
        }
    }
    out.chars().take(1000).collect()
}

fn keep_prob(ft: &FastText, p: &Paragraph) -> f32 {
    let input = ft_input(&p.text);
    if input.is_empty() {
        return 0.0;
    }
    ft.predict(&input, 2, 0.0)
        .iter()
        .find(|pr| pr.label == "__label__1")
        .map(|pr| pr.prob)
        .unwrap_or(0.0)
}

/// Longest JSON-LD description/abstract string (>= 500 chars), if any — the
/// recovery path for article pages whose body the classifier dropped.
fn jsonld_description(html: &str) -> Option<String> {
    if !html.contains("application/ld+json") {
        return None;
    }
    let mut best = String::new();
    let mut rest = html;
    while let Some(start) = rest.find("application/ld+json") {
        let after = &rest[start..];
        let Some(open) = after.find('>') else { break };
        let body_start = open + 1;
        let Some(close) = after[body_start..].find("</script") else {
            rest = &after[body_start..];
            continue;
        };
        let body = &after[body_start..body_start + close];
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(body.trim()) {
            let mut stack = vec![&data];
            while let Some(node) = stack.pop() {
                match node {
                    serde_json::Value::Object(map) => {
                        for (k, v) in map {
                            if (k == "description" || k == "abstract")
                                && v.as_str().map(|s| s.len() > best.len()).unwrap_or(false)
                            {
                                best = v.as_str().unwrap().to_string();
                            }
                            stack.push(v);
                        }
                    }
                    serde_json::Value::Array(items) => stack.extend(items),
                    _ => {}
                }
            }
        }
        rest = &after[body_start + close..];
    }
    (best.chars().count() >= 500).then_some(best)
}

fn extract(html: &str, cfg: &Config) -> Result<String, String> {
    let repaired = repair_mojibake(html);
    let html: &str = repaired.as_deref().unwrap_or(html);
    let tree = HTMLTree::parse(html).map_err(|e| e.to_string())?;
    if cfg.mode == "inner" {
        return Ok(tree.body().map(|b| b.inner_text()).unwrap_or_default());
    }
    let th = cfg.threshold as f64;
    let out = match (&cfg.ft, &cfg.forest) {
        (Some(ft), Some(forest)) => {
            let policy = KeepPolicy {
                threshold: th,
                neighbour_threshold: th * 0.625,
                heading_threshold: th * 0.375,
                nav_threshold: 0.0, // forest sees nav/aside flags itself
                ..KeepPolicy::default()
            };
            extract_main_text(
                &tree,
                |paras: &[Paragraph]| {
                    let ft_probs: Vec<f64> =
                        paras.iter().map(|p| keep_prob(ft, p) as f64).collect();
                    stack_features(paras, &ft_probs)
                        .iter()
                        .map(|f| forest.predict(f))
                        .collect()
                },
                &policy,
            )
        }
        (Some(ft), None) => {
            let policy = KeepPolicy { threshold: th, ..KeepPolicy::default() };
            extract_main_text(
                &tree,
                |paras: &[Paragraph]| paras.iter().map(|p| keep_prob(ft, p) as f64).collect(),
                &policy,
            )
        }
        _ => {
            let paras = extract_paragraphs(&tree);
            let keep: Vec<bool> = paras.iter().map(|_| true).collect();
            assemble(&paras, &keep)
        }
    };
    if cfg.meta_floor > 0 && out.chars().count() < cfg.meta_floor {
        let mut best = out.clone();
        if let Some(fb) = meta_fallback(&tree) {
            if fb.chars().count() > best.chars().count() {
                best = fb;
            }
        }
        if let Some(ld) = jsonld_description(html) {
            if ld.chars().count() > best.chars().count() {
                best = ld;
            }
        }
        return Ok(best);
    }
    Ok(out)
}

/// Dump per-paragraph features + fastText prob for offline policy experiments.
/// Fully streaming (input and output) so 100k-doc datasets never sit in memory.
fn open_input(path: &str) -> Box<dyn BufRead> {
    let f = File::open(path).expect("open input");
    if path.ends_with(".gz") {
        Box::new(BufReader::new(flate2::read::MultiGzDecoder::new(f)))
    } else {
        Box::new(BufReader::new(f))
    }
}

fn dump_paragraphs(in_path: &str, ft: &FastText, out_path: &str) {
    let reader = open_input(in_path);
    let mut w = GzEncoder::new(
        BufWriter::new(File::create(out_path).expect("create dump")),
        Compression::fast(),
    );
    let mut n_done = 0usize;
    let mut chunk: Vec<Record> = Vec::with_capacity(1000);
    let mut lines = reader.lines();
    loop {
        chunk.clear();
        for line in lines.by_ref() {
            let line = line.expect("read line");
            if line.trim().is_empty() {
                continue;
            }
            chunk.push(serde_json::from_str(&line).expect("parse record"));
            if chunk.len() == 1000 {
                break;
            }
        }
        if chunk.is_empty() {
            break;
        }
        let offset = n_done;
        let rows: Vec<String> = chunk
            .par_iter()
            .enumerate()
            .map(|(j, r)| dump_one(offset + j, r, ft))
            .collect();
        for row in rows {
            writeln!(w, "{}", row).expect("write dump row");
        }
        n_done += chunk.len();
        if n_done % 10000 == 0 {
            eprintln!("dumped {} docs", n_done);
        }
    }
    w.finish().expect("finish gzip");
    eprintln!("dumped paragraphs for {} docs to {}", n_done, out_path);
}

fn dump_one(i: usize, r: &Record, ft: &FastText) -> String {
    let repaired = repair_mojibake(&r.html);
    let html: &str = repaired.as_deref().unwrap_or(&r.html);
    let paras = match HTMLTree::parse(html) {
        Ok(tree) => extract_paragraphs(&tree),
        Err(_) => Vec::new(),
    };
    let jp: Vec<serde_json::Value> = paras
        .iter()
        .map(|p| {
            serde_json::json!({
                "text": p.text,
                "prob": keep_prob(ft, p),
                "verbatim": p.verbatim,
                "heading": p.heading,
                "depth": p.depth,
                "link_chars": p.chars_in_links,
                "tags": p.tags_count,
                "nav": p.dom_nav, "aside": p.dom_aside,
                "header": p.dom_header, "footer": p.dom_footer,
                "form": p.dom_form, "list": p.dom_list,
                "table": p.dom_table, "main": p.dom_main,
                "bq": p.dom_blockquote,
            })
        })
        .collect();
    serde_json::json!({"idx": i, "paragraphs": jp}).to_string()
}

fn debug_html(path: &str) {
    use resiliparse::parse::html::dom::traits::{NodeInterface, ParentNode};
    let html = std::fs::read_to_string(path).expect("read html");
    let tree = HTMLTree::parse(&html).expect("parse");
    match tree.body() {
        Some(body) => {
            let kids = body.child_nodes();
            eprintln!("body: {} children, inner_text len {}", kids.len(), body.inner_text().len());
            for c in kids.iter() {
                let name = c.node_name().unwrap_or_default();
                let tlen = c.text_content().map(|t| t.len()).unwrap_or(0);
                let preview: String =
                    c.text_content().unwrap_or_default().chars().take(80).collect();
                eprintln!("  body child: {} (text {} chars) {:?}", name, tlen, preview);
            }
        }
        None => eprintln!("body: MISSING"),
    }
    fn dump(node: resiliparse::parse::html::dom::node::Node, depth: usize) {
        use resiliparse::parse::html::dom::traits::NodeInterface;
        if depth > 15 {
            return;
        }
        for c in node.child_nodes().iter() {
            let name = c.node_name().unwrap_or_default();
            let tlen = c.text_content().map(|t| t.len()).unwrap_or(0);
            if tlen > 20 {
                eprintln!("{}{} ({})", "  ".repeat(depth), name, tlen);
            }
            dump(c, depth + 1);
        }
    }
    if let Some(body) = tree.body() {
        dump(resiliparse::parse::html::dom::node::Node::Element(body), 0);
    }
}

fn leak_test(path: &str, n: usize) {
    let html = std::fs::read_to_string(path).expect("read html");
    for i in 0..n {
        if let Ok(tree) = HTMLTree::parse(&html) {
            let paras = extract_paragraphs(&tree);
            std::hint::black_box(paras.len());
        }
        if i % 10000 == 0 {
            let rss = unsafe {
                let mut info: libc_rusage = std::mem::zeroed();
                getrusage(0, &mut info as *mut _ as *mut _);
                info.ru_maxrss
            };
            eprintln!("iter {} peak-rss {:.1} MB", i, rss as f64 / 1048576.0);
        }
    }
}

#[repr(C)]
struct libc_rusage {
    ru_utime: [u64; 2],
    ru_stime: [u64; 2],
    ru_maxrss: i64,
    _rest: [i64; 13],
}
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut std::ffi::c_void) -> i32;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--debug-html" {
        debug_html(&args[2]);
        return;
    }
    if args.len() == 4 && args[1] == "--leak-test" {
        leak_test(&args[2], args[3].parse().expect("bad n"));
        return;
    }
    if args.len() == 3 && args[1] == "--extract-count" {
        let mut n = 0usize;
        let mut total_paras = 0usize;
        for line in open_input(&args[2]).lines() {
            let line = line.expect("read line");
            if line.trim().is_empty() {
                continue;
            }
            let r: Record = serde_json::from_str(&line).expect("parse record");
            if n >= 69000 {
                eprintln!("at doc {} (html {} bytes)", n, r.html.len());
            }
            if let Ok(tree) = HTMLTree::parse(&r.html) {
                total_paras += extract_paragraphs(&tree).len();
            }
            n += 1;
            if n % 10000 == 0 {
                let rss = unsafe {
                    let mut info: libc_rusage = std::mem::zeroed();
                    getrusage(0, &mut info as *mut _ as *mut _);
                    info.ru_maxrss
                };
                eprintln!("extracted {} docs, {} paras, peak-rss {:.0} MB", n, total_paras, rss as f64 / 1048576.0);
            }
        }
        eprintln!("done: {} docs, {} paras", n, total_paras);
        return;
    }
    if args.len() == 3 && args[1] == "--count" {
        let mut n = 0usize;
        let mut bytes = 0usize;
        for line in open_input(&args[2]).lines() {
            let line = line.expect("read line");
            bytes += line.len();
            if !line.trim().is_empty() {
                let r: Record = serde_json::from_str(&line).expect("parse record");
                n += 1;
                std::hint::black_box(r.html.len());
            }
            if n % 10000 == 0 {
                eprintln!("counted {} ({:.2} GiB)", n, bytes as f64 / 1073741824.0);
            }
        }
        eprintln!("total {} docs, {:.2} GiB", n, bytes as f64 / 1073741824.0);
        return;
    }
    if args.len() < 3 {
        eprintln!(
            "usage: rp-bench <input.jsonl.gz> <output.predictions.jsonl> \
             [--mode inner|para|dump] [--ft <model.bin>] [--threshold 0.5]"
        );
        std::process::exit(2);
    }
    let mut cfg = Config {
        mode: "inner".to_string(),
        threshold: 0.5,
        ft: None,
        forest: None,
        meta_floor: 150,
    };
    let mut i = 3;
    let mut ft_path: Option<String> = None;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                cfg.mode = args[i + 1].clone();
                i += 2;
            }
            "--ft" => {
                ft_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--stack" => {
                cfg.forest = Some(load_forest(&args[i + 1]));
                i += 2;
            }
            "--meta-floor" => {
                cfg.meta_floor = args[i + 1].parse().expect("bad meta-floor");
                i += 2;
            }
            "--threshold" => {
                cfg.threshold = args[i + 1].parse().expect("bad threshold");
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    if let Some(path) = ft_path {
        let t = Instant::now();
        let ft = FastText::load_model(&path).expect("load fastText model");
        eprintln!("loaded fastText model in {:.1}s", t.elapsed().as_secs_f64());
        cfg.ft = Some(ft);
    }

    if cfg.mode == "dump" {
        dump_paragraphs(&args[1], cfg.ft.as_ref().expect("--ft required for dump"), &args[2]);
        return;
    }

    let records: Vec<Record> = open_input(&args[1])
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).expect("parse record"))
        .collect();
    eprintln!("loaded {} records", records.len());

    let wall = Instant::now();
    let results: Vec<(usize, Result<String, String>, f64)> = records
        .par_iter()
        .enumerate()
        .map(|(i, r)| {
            let t = Instant::now();
            let pred = extract(&r.html, &cfg);
            (i, pred, t.elapsed().as_secs_f64() * 1000.0)
        })
        .collect();
    let wall_s = wall.elapsed().as_secs_f64();

    let mut out = BufWriter::new(File::create(&args[2]).expect("create output"));
    let mut errors = 0usize;
    for (i, pred, ms) in &results {
        let r = &records[*i];
        let (prediction, error) = match pred {
            Ok(p) => (p.as_str(), None),
            Err(e) => {
                errors += 1;
                ("", Some(e.as_str()))
            }
        };
        let row = serde_json::json!({
            "idx": i,
            "warc_record_id": r.warc_record_id,
            "url": r.url,
            "snapshot": r.snapshot,
            "prediction": prediction,
            "error": error,
            "elapsed_ms": ms,
        });
        writeln!(out, "{}", row).expect("write row");
    }
    let per_doc_ms: f64 =
        results.iter().map(|(_, _, ms)| ms).sum::<f64>() / results.len().max(1) as f64;
    eprintln!(
        "done: {} docs, {} errors, wall {:.1}s, mean {:.2} ms/doc (single-doc basis)",
        results.len(),
        errors,
        wall_s,
        per_doc_ms
    );
}
