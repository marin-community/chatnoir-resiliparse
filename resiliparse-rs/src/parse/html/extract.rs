// Copyright 2026 the Resiliparse contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Main-content extraction: segment an HTML document into block-level paragraphs
//! with structural features, so a caller-supplied classifier (or the built-in
//! heuristics) can separate content from boilerplate.
//!
//! Paragraph boundaries follow the jusText model: block-level container tags and
//! `<br><br>` start a new paragraph; cell/item-level tags (`td`, `li`, …) stay
//! inside their containing paragraph. `<pre>`/`<textarea>` content is verbatim.

use crate::parse::html::dom::traits::NodeInterfaceBaseImpl;
use crate::parse::html::lexbor::*;
use crate::parse::html::tree::HTMLTree;
use crate::third_party::lexbor::lxb_tag_id_enum_t::*;
use crate::third_party::lexbor::*;

/// One block-level paragraph with the structural features needed for
/// content/boilerplate classification.
#[derive(Debug, Clone, Default)]
pub struct Paragraph {
    /// Whitespace-normalized text (verbatim for `<pre>`/`<textarea>` blocks).
    pub text: String,
    /// Characters contributed by text inside `<a>` descendants.
    pub chars_in_links: usize,
    /// Number of (non-paragraph-break) tags encountered within the block.
    pub tags_count: usize,
    /// True if the block is preformatted (`<pre>`/`<textarea>`).
    pub verbatim: bool,
    /// True if the block's text came from a `<textarea>`: verbatim formatting,
    /// but form input rather than guaranteed content (wiki edit pages leak raw
    /// markup) — so it is classified, never force-kept.
    pub from_textarea: bool,
    /// True if the paragraph sits under an `h1`-`h6` element.
    pub heading: bool,
    /// DOM-path depth at paragraph start.
    pub depth: usize,
    /// True if any ancestor is `<nav>` (or `<menu>`).
    pub dom_nav: bool,
    /// True if any ancestor is `<aside>`.
    pub dom_aside: bool,
    /// True if any ancestor is `<header>`.
    pub dom_header: bool,
    /// True if any ancestor is `<footer>`.
    pub dom_footer: bool,
    /// True if any ancestor is a form-related element.
    pub dom_form: bool,
    /// True if any ancestor is a list element.
    pub dom_list: bool,
    /// True if any ancestor is a table element.
    pub dom_table: bool,
    /// True if any ancestor is `<article>`/`<section>`/`<main>`.
    pub dom_main: bool,
    /// True if any ancestor is `<blockquote>`.
    pub dom_blockquote: bool,
    /// Boilerplate-signal tokens (`nav`, `sidebar`, `comment`, …) found in
    /// ancestor `class`/`id` attributes.
    pub cls_neg: u32,
    /// Content-signal tokens (`content`, `article`, `post`, …) found in
    /// ancestor `class`/`id` attributes.
    pub cls_pos: u32,
    /// True if an ancestor class/id carries a `comment` token.
    pub cls_comment: bool,
    /// True if an ancestor class/id carries an ad token.
    pub cls_ad: bool,
    /// Identity of the top-level block container this paragraph belongs to
    /// (direct child-of-body region). 0 when outside any container.
    pub container1: u32,
    /// Identity of the second-level block container. 0 when absent.
    pub container2: u32,
}

impl Paragraph {
    /// Fraction of the paragraph's characters that sit inside links.
    pub fn link_density(&self) -> f64 {
        let n = self.text.chars().count();
        if n == 0 {
            0.0
        } else {
            self.chars_in_links as f64 / n as f64
        }
    }

    /// Number of whitespace-separated words.
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Number of features produced by [`stack_features`].
pub const N_STACK_FEATURES: usize = 49;

/// Scan a `class`/`id` attribute string for boilerplate/content signal tokens.
/// Returns (neg_hits, pos_hits, has_comment, has_ad). Tokens are split on
/// non-alphanumerics; short keywords match exactly, longer ones by substring.
fn class_token_hits(s: &str) -> (u32, u32, bool, bool) {
    const NEG_SUB: &[&str] = &[
        "menu",
        "sidebar",
        "footer",
        "header",
        "comment",
        "share",
        "social",
        "related",
        "widget",
        "breadcrumb",
        "banner",
        "promo",
        "cookie",
        "popup",
        "modal",
        "login",
        "signup",
        "search",
        "pager",
        "pagination",
        "toolbar",
        "advert",
        "sponsor",
        "navigation",
        "masthead",
        "subscribe",
        "copyright",
        "dropdown",
        "hidden",
    ];
    const POS_SUB: &[&str] = &["content", "article", "description", "abstract", "product"];
    let (mut neg, mut pos, mut comment, mut ad) = (0u32, 0u32, false, false);
    for tok in s.split(|c: char| !c.is_ascii_alphanumeric()) {
        if tok.is_empty() {
            continue;
        }
        match tok {
            "ad" | "ads" | "adv" => {
                neg += 1;
                ad = true;
            }
            "nav" => neg += 1,
            "main" | "body" | "post" | "text" | "story" | "entry" => pos += 1,
            _ => {}
        }
        if tok.len() >= 4 {
            if NEG_SUB.iter().any(|k| tok.contains(k)) {
                neg += 1;
                if tok.contains("comment") {
                    comment = true;
                }
            } else if POS_SUB.iter().any(|k| tok.contains(k)) {
                pos += 1;
            }
        }
    }
    (neg, pos, comment, ad)
}

/// Does the line open like a `Label: value` pair? (letter, then up to 25
/// label-ish characters, then a colon followed by whitespace.)
fn label_line(line: &str) -> bool {
    if let Some(pos) = line.find(':') {
        let head = &line[..pos];
        let mut cs = head.chars();
        let ok_head = matches!(cs.next(), Some(c) if c.is_ascii_alphabetic())
            && head.chars().count() <= 26
            && head
                .chars()
                .skip(1)
                .all(|c| c.is_alphanumeric() || matches!(c, '_' | ' ' | '/' | '(' | ')' | '&' | '.' | '-'));
        ok_head
            && line[pos + ':'.len_utf8()..]
                .chars()
                .next()
                .map(|c| c.is_whitespace())
                .unwrap_or(false)
    } else {
        false
    }
}

/// Per-paragraph feature vectors for a stacked paragraph classifier:
/// structural features plus a text-classifier probability (`text_probs`) and
/// neighbouring-paragraph context.
pub fn stack_features(paragraphs: &[Paragraph], text_probs: &[f64]) -> Vec<Vec<f64>> {
    let n = paragraphs.len();
    let base: Vec<(f64, f64)> = paragraphs
        .iter()
        .map(|p| ((1.0 + p.text.chars().count() as f64).ln(), p.link_density()))
        .collect();
    // per-paragraph line statistics + document-level structure statistics
    let line_stats: Vec<(f64, f64, f64)> = paragraphs
        .iter()
        .map(|p| {
            let lines: Vec<&str> = p.text.lines().filter(|l| !l.trim().is_empty()).collect();
            if lines.is_empty() {
                return (0.0, 0.0, 0.0);
            }
            let short = lines.iter().filter(|l| l.chars().count() < 60).count() as f64 / lines.len() as f64;
            let label = lines.iter().filter(|l| label_line(l.trim())).count() as f64 / lines.len() as f64;
            (lines.len() as f64, short, label)
        })
        .collect();
    let total_chars: usize = paragraphs.iter().map(|p| p.text.chars().count()).sum::<usize>().max(1);
    let (mut n_lines_all, mut n_short_all, mut n_label_all) = (0usize, 0usize, 0usize);
    for p in paragraphs {
        for l in p.text.lines().filter(|l| !l.trim().is_empty()) {
            n_lines_all += 1;
            if l.chars().count() < 60 {
                n_short_all += 1;
            }
            if label_line(l.trim()) {
                n_label_all += 1;
            }
        }
    }
    let doc_short = n_short_all as f64 / n_lines_all.max(1) as f64;
    let doc_label = n_label_all as f64 / n_lines_all.max(1) as f64;
    let doc_hi_ft = text_probs.iter().filter(|&&x| x > 0.5).count() as f64 / n.max(1) as f64;
    paragraphs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let chars = p.text.chars().count();
            let words = p.word_count();
            let ends_sentence = matches!(p.text.chars().last(), Some('.' | '!' | '?'));
            let mut f = Vec::with_capacity(N_STACK_FEATURES);
            f.push(base[i].0);
            f.push((1.0 + words as f64).ln());
            f.push(base[i].1);
            f.push(p.heading as u8 as f64);
            f.push(i as f64 / (n - 1).max(1) as f64);
            f.push(ends_sentence as u8 as f64);
            f.push(chars as f64 / words.max(1) as f64);
            f.push((1.0 + p.depth as f64).ln());
            f.push((1.0 + p.tags_count as f64).ln());
            f.push(p.verbatim as u8 as f64);
            for flag in [
                p.dom_nav,
                p.dom_aside,
                p.dom_header,
                p.dom_footer,
                p.dom_form,
                p.dom_list,
                p.dom_table,
                p.dom_main,
                p.dom_blockquote,
            ] {
                f.push(flag as u8 as f64);
            }
            f.push(text_probs[i]);
            f.push(if i > 0 { text_probs[i - 1] } else { 0.0 });
            f.push(if i + 1 < n { text_probs[i + 1] } else { 0.0 });
            f.push(if i > 0 { base[i - 1].0 } else { 0.0 });
            f.push(if i + 1 < n { base[i + 1].0 } else { 0.0 });
            f.push(if i > 0 { base[i - 1].1 } else { 0.0 });
            f.push(if i + 1 < n { base[i + 1].1 } else { 0.0 });
            let punct = p
                .text
                .chars()
                .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
                .count() as f64
                / chars.max(1) as f64;
            let digit = p.text.chars().filter(|c| c.is_numeric()).count() as f64 / chars.max(1) as f64;
            let upper = p.text.chars().filter(|c| c.is_uppercase()).count() as f64 / chars.max(1) as f64;
            let doc_mean = text_probs.iter().sum::<f64>() / n.max(1) as f64;
            let doc_max = text_probs.iter().cloned().fold(0.0f64, f64::max);
            f.push(punct);
            f.push(digit);
            f.push(upper);
            f.push((p.text.starts_with("- ") || p.text.starts_with("1. ")) as u8 as f64);
            f.push(p.text.trim_end().ends_with(':') as u8 as f64);
            f.push((1.0 + n as f64).ln());
            f.push(text_probs[i] - doc_mean);
            f.push(doc_max);
            f.push(if i > 1 { text_probs[i - 2] } else { 0.0 });
            f.push(if i + 2 < n { text_probs[i + 2] } else { 0.0 });
            // structured-content features (labels, line shape, doc structure)
            f.push((1.0 + line_stats[i].0).ln());
            f.push(line_stats[i].1);
            f.push(line_stats[i].2);
            f.push(label_line(p.text.trim()) as u8 as f64);
            f.push(doc_short);
            f.push(doc_label);
            f.push(doc_hi_ft);
            f.push(chars as f64 / total_chars as f64);
            f.push((n - 1 - i) as f64 / (n - 1).max(1) as f64);
            // ancestor class/id signal tokens
            f.push((1.0 + p.cls_neg as f64).ln());
            f.push((1.0 + p.cls_pos as f64).ln());
            f.push(p.cls_comment as u8 as f64);
            f.push(p.cls_ad as u8 as f64);
            f
        })
        .collect()
}

/// One decision tree in sklearn's flat-array encoding. `value[leaf]` is the
/// positive-class probability at that leaf.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
}

impl DecisionTree {
    fn predict(&self, features: &[f64]) -> f64 {
        let mut node = 0usize;
        while self.children_left[node] >= 0 {
            let f = self.feature[node] as usize;
            node = if features[f] <= self.threshold[node] {
                self.children_left[node] as usize
            } else {
                self.children_right[node] as usize
            };
        }
        self.value[node]
    }
}

/// A random forest over [`stack_features`] vectors (mean of tree probabilities).
#[derive(Debug, Clone, Default)]
pub struct DecisionForest {
    pub trees: Vec<DecisionTree>,
}

impl DecisionForest {
    /// Positive-class probability for one feature vector.
    pub fn predict(&self, features: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        self.trees.iter().map(|t| t.predict(features)).sum::<f64>() / self.trees.len() as f64
    }
}

#[inline]
fn is_paragraph_tag(t: lxb_tag_id_enum_t::Type) -> bool {
    matches!(
        t,
        LXB_TAG_BODY
            | LXB_TAG_BLOCKQUOTE
            | LXB_TAG_CENTER
            | LXB_TAG_COL
            | LXB_TAG_COLGROUP
            | LXB_TAG_DIV
            | LXB_TAG_DL
            | LXB_TAG_FIELDSET
            | LXB_TAG_FORM
            | LXB_TAG_LEGEND
            | LXB_TAG_OPTGROUP
            | LXB_TAG_P
            | LXB_TAG_PRE
            | LXB_TAG_TABLE
            | LXB_TAG_TEXTAREA
            | LXB_TAG_TFOOT
            | LXB_TAG_THEAD
            | LXB_TAG_TR
            | LXB_TAG_UL
            | LXB_TAG_H1
            | LXB_TAG_H2
            | LXB_TAG_H3
            | LXB_TAG_H4
            | LXB_TAG_H5
            | LXB_TAG_H6
    )
}

#[inline]
fn is_separator_tag(t: lxb_tag_id_enum_t::Type) -> bool {
    matches!(
        t,
        LXB_TAG_IMG | LXB_TAG_TD | LXB_TAG_TH | LXB_TAG_LI | LXB_TAG_DD | LXB_TAG_DT | LXB_TAG_OPTION | LXB_TAG_CAPTION
    )
}

/// Subtrees whose text is never document content. `<noscript>` is deliberately
/// NOT here: no-JS fallbacks regularly hold the real page content (whole
/// articles, OCR text) and the paragraph classifier drops the "enable
/// JavaScript" junk on its own.
#[inline]
fn is_skip_tag(t: lxb_tag_id_enum_t::Type) -> bool {
    matches!(
        t,
        LXB_TAG_SCRIPT
            | LXB_TAG_STYLE
            | LXB_TAG_IFRAME
            | LXB_TAG_EMBED
            | LXB_TAG_OBJECT
            | LXB_TAG_APPLET
            | LXB_TAG_INPUT
            | LXB_TAG_SELECT
            | LXB_TAG_BUTTON
            | LXB_TAG_DATALIST
            | LXB_TAG_SVG
            | LXB_TAG_TEMPLATE
    )
}

/// List classes that mark non-content (structural) lists: no `- ` / `1. ` markers.
const STRUCTURAL_LIST: &[&str] = &[
    "nav",
    "menu",
    "tab",
    "crumb",
    "pag",
    "sidebar",
    "widget",
    "toolbar",
    "social",
    "share",
    "related",
    "posts",
    "comment",
    "footer",
    "header",
    "breadcrumb",
    "links",
];

/// Collapse whitespace runs to a single space, or a single `\n` if the run
/// contains a newline / carriage return.
fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run_has_newline = false;
    let mut in_run = false;
    for c in text.chars() {
        if c.is_whitespace() {
            in_run = true;
            if c == '\n' || c == '\r' {
                run_has_newline = true;
            }
        } else {
            if in_run {
                out.push(if run_has_newline { '\n' } else { ' ' });
                in_run = false;
                run_has_newline = false;
            }
            out.push(c);
        }
    }
    if in_run {
        out.push(if run_has_newline { '\n' } else { ' ' });
    }
    out
}

#[derive(Default)]
struct AncestorFlags {
    nav: usize,
    aside: usize,
    header: usize,
    footer: usize,
    form: usize,
    list: usize,
    table: usize,
    main: usize,
    blockquote: usize,
    heading: usize,
}

impl AncestorFlags {
    fn bump(&mut self, t: lxb_tag_id_enum_t::Type, delta: isize) {
        let inc = |v: &mut usize| {
            *v = v.wrapping_add_signed(delta);
        };
        match t {
            LXB_TAG_NAV | LXB_TAG_MENU => inc(&mut self.nav),
            LXB_TAG_ASIDE => inc(&mut self.aside),
            LXB_TAG_HEADER => inc(&mut self.header),
            LXB_TAG_FOOTER => inc(&mut self.footer),
            LXB_TAG_FORM | LXB_TAG_LABEL => inc(&mut self.form),
            LXB_TAG_LI | LXB_TAG_UL | LXB_TAG_OL | LXB_TAG_DL | LXB_TAG_DD | LXB_TAG_DT => inc(&mut self.list),
            LXB_TAG_TABLE | LXB_TAG_TD | LXB_TAG_TH | LXB_TAG_TR | LXB_TAG_TBODY | LXB_TAG_THEAD => {
                inc(&mut self.table)
            }
            LXB_TAG_ARTICLE | LXB_TAG_SECTION | LXB_TAG_MAIN => inc(&mut self.main),
            LXB_TAG_BLOCKQUOTE => inc(&mut self.blockquote),
            LXB_TAG_H1 | LXB_TAG_H2 | LXB_TAG_H3 | LXB_TAG_H4 | LXB_TAG_H5 | LXB_TAG_H6 => inc(&mut self.heading),
            _ => {}
        }
    }
}

/// One table cell: tag (th/td) and normalized text.
struct Cell {
    is_th: bool,
    text: String,
}

/// Collect a cell's visible text (mini-walk: skip script/style/nested tables,
/// space-separate nested separator tags). Depth-capped: pathological nesting
/// disqualifies the table instead of overflowing the stack.
unsafe fn cell_text(
    node: *mut lxb_dom_node_t,
    link_depth: usize,
    out: &mut String,
    links: &mut usize,
    depth: usize,
) -> bool {
    if depth > 64 {
        return false;
    }
    unsafe {
        let mut child = (*node).first_child;
        let mut ok = true;
        while !child.is_null() {
            match (*child).type_ {
                lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_TEXT => {
                    let t = str_from_dom_node(child).unwrap_or_default();
                    out.push(' ');
                    out.push_str(t);
                    if link_depth > 0 {
                        *links += t.chars().count();
                    }
                }
                lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_ELEMENT => {
                    let tag = (*child).local_name as lxb_tag_id_enum_t::Type;
                    if tag == LXB_TAG_TABLE {
                        ok = false; // nested table -> not a data table
                    } else if !is_skip_tag(tag) {
                        let ld = link_depth + (tag == LXB_TAG_A) as usize;
                        if !cell_text(child, ld, out, links, depth + 1) {
                            ok = false;
                        }
                    }
                }
                _ => {}
            }
            child = (*child).next;
        }
        ok
    }
}

/// Render an unambiguous data table the way reference corpora transcribe them,
/// or return None to let the normal paragraph path handle it.
unsafe fn try_rewrite_table(table: *mut lxb_dom_node_t) -> Option<String> {
    // collect rows: ./tr, ./tbody/tr, ./thead/tr
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    let mut total_chars = 0usize;
    let mut link_chars = 0usize;
    unsafe {
        let mut group = (*table).first_child;
        let mut tr_nodes: Vec<*mut lxb_dom_node_t> = Vec::new();
        while !group.is_null() {
            if (*group).type_ == lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_ELEMENT {
                let tag = (*group).local_name as lxb_tag_id_enum_t::Type;
                if tag == LXB_TAG_TR {
                    tr_nodes.push(group);
                } else if matches!(tag, LXB_TAG_TBODY | LXB_TAG_THEAD | LXB_TAG_TFOOT) {
                    let mut tr = (*group).first_child;
                    while !tr.is_null() {
                        if (*tr).type_ == lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_ELEMENT
                            && (*tr).local_name as lxb_tag_id_enum_t::Type == LXB_TAG_TR
                        {
                            tr_nodes.push(tr);
                        }
                        tr = (*tr).next;
                    }
                }
            }
            group = (*group).next;
        }
        for tr in tr_nodes {
            let mut cells: Vec<Cell> = Vec::new();
            let mut cell = (*tr).first_child;
            while !cell.is_null() {
                if (*cell).type_ == lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_ELEMENT {
                    let tag = (*cell).local_name as lxb_tag_id_enum_t::Type;
                    if matches!(tag, LXB_TAG_TD | LXB_TAG_TH) {
                        let mut raw = String::new();
                        let mut links = 0usize;
                        if !cell_text(cell, 0, &mut raw, &mut links, 0) {
                            return None; // nested table
                        }
                        let text = normalize_whitespace(raw.trim()).replace('\n', " ");
                        total_chars += text.chars().count();
                        link_chars += links;
                        cells.push(Cell {
                            is_th: tag == LXB_TAG_TH,
                            text,
                        });
                    }
                }
                cell = (*cell).next;
            }
            if !cells.is_empty() {
                rows.push(cells);
            }
        }
    }
    let data_rows: Vec<&Vec<Cell>> = rows.iter().filter(|r| r.len() >= 2).collect();
    if data_rows.len() < 3 {
        return None;
    }
    if link_chars * 2 > total_chars.max(1) {
        return None; // nav/link table
    }
    let mut cell_lengths: Vec<usize> = data_rows
        .iter()
        .flat_map(|r| r.iter().map(|c| c.text.chars().count()))
        .collect();
    cell_lengths.sort_unstable();
    if cell_lengths[cell_lengths.len() / 2] > 80 {
        return None; // prose rows
    }
    let n_cells = cell_lengths.len();
    let n_empty = data_rows
        .iter()
        .flat_map(|r| r.iter())
        .filter(|c| c.text.is_empty())
        .count();
    if n_empty * 5 > n_cells * 2 {
        return None; // mostly empty -> layout/form table
    }
    // month-calendar widget: mostly day-of-month integers at width 5-8
    let widths: Vec<usize> = data_rows.iter().map(|r| r.len()).collect();
    let modal_w = {
        let mut counts = std::collections::HashMap::new();
        for w in &widths {
            *counts.entry(*w).or_insert(0usize) += 1;
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(w, _)| w).unwrap_or(0)
    };
    let non_empty: Vec<&str> = data_rows
        .iter()
        .flat_map(|r| r.iter())
        .map(|c| c.text.as_str())
        .filter(|t| !t.is_empty())
        .collect();
    if (5..=8).contains(&modal_w) && !non_empty.is_empty() {
        let day_frac = non_empty
            .iter()
            .filter(|t| t.parse::<u32>().map(|v| (1..=31).contains(&v)).unwrap_or(false))
            .count() as f64
            / non_empty.len() as f64;
        if day_frac >= 0.7 {
            return None; // calendar
        }
    }

    let uniform = rows.iter().all(|r| r.len() == rows[0].len()) && rows[0].len() >= 2;
    let pipe_row = |r: &Vec<Cell>| -> String {
        r.iter()
            .map(|c| {
                if c.text.is_empty() {
                    "\u{a0}".to_string()
                } else {
                    c.text.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    if uniform && rows[0].iter().all(|c| c.is_th) {
        // data table with a header -> GFM pipe table
        let mut lines: Vec<String> = Vec::with_capacity(rows.len() + 1);
        lines.push(pipe_row(&rows[0]));
        lines.push(vec!["---"; rows[0].len()].join(" | "));
        for r in &rows[1..] {
            lines.push(pipe_row(r));
        }
        return Some(lines.join("\n"));
    }
    // row-label table (infobox style): >=70% of data rows lead with a <th> label
    let th_led = data_rows.iter().filter(|r| r[0].is_th && !r[0].text.is_empty()).count();
    if rows.iter().all(|r| r.len() <= 2) && th_led * 10 >= data_rows.len() * 7 {
        let mut lines: Vec<String> = Vec::new();
        for r in &rows {
            if r.len() == 1 {
                if !r[0].text.is_empty() {
                    lines.push(r[0].text.clone());
                }
            } else if !r[0].text.is_empty() || !r[1].text.is_empty() {
                let label = r[0].text.trim_end_matches(':').trim_end();
                lines.push(format!("{}: {}", label, r[1].text));
            }
        }
        return Some(lines.join("\n"));
    }
    // label-value table: every first-row cell ends with ':'
    if uniform
        && rows[0].iter().filter(|c| !c.text.is_empty()).count() > 0
        && rows[0]
            .iter()
            .all(|c| c.text.is_empty() || c.text.trim_end().ends_with(':'))
    {
        return Some(
            rows.iter()
                .map(|r| r.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    None
}

struct Walker {
    paragraphs: Vec<Paragraph>,
    raw: String,
    raw_has_text: bool,
    chars_in_links: usize,
    tags_count: usize,
    verbatim: bool,
    depth: usize,
    link: usize,
    pre: usize,
    textarea: usize,
    br: bool,
    flags: AncestorFlags,
    list_stack: Vec<(lxb_tag_id_enum_t::Type, usize, bool)>,
    // ancestor class/id signal counters + per-element deltas for unwinding
    cls: (u32, u32, u32, u32), // (neg, pos, comment, ad)
    attr_stack: Vec<(u32, u32, u32, u32)>,
    // block-container identity: ids of enclosing containers, outermost first
    containers: Vec<u32>,
    container_counter: u32,
}

impl Walker {
    fn new() -> Self {
        Walker {
            paragraphs: Vec::new(),
            raw: String::new(),
            raw_has_text: false,
            chars_in_links: 0,
            tags_count: 0,
            verbatim: false,
            depth: 0,
            link: 0,
            pre: 0,
            textarea: 0,
            br: false,
            flags: AncestorFlags::default(),
            list_stack: Vec::new(),
            cls: (0, 0, 0, 0),
            attr_stack: Vec::new(),
            containers: Vec::new(),
            container_counter: 0,
        }
    }

    fn flush_paragraph(&mut self) {
        if self.raw_has_text {
            let text = if self.verbatim {
                self.raw.trim_matches('\n').trim_end().to_string()
            } else {
                normalize_whitespace(self.raw.trim())
            };
            if !text.is_empty() {
                self.paragraphs.push(Paragraph {
                    text,
                    chars_in_links: self.chars_in_links,
                    tags_count: self.tags_count,
                    verbatim: self.verbatim,
                    from_textarea: self.textarea > 0,
                    heading: self.flags.heading > 0,
                    depth: self.depth,
                    dom_nav: self.flags.nav > 0,
                    dom_aside: self.flags.aside > 0,
                    dom_header: self.flags.header > 0,
                    dom_footer: self.flags.footer > 0,
                    dom_form: self.flags.form > 0,
                    dom_list: self.flags.list > 0,
                    dom_table: self.flags.table > 0,
                    dom_main: self.flags.main > 0,
                    dom_blockquote: self.flags.blockquote > 0,
                    cls_neg: self.cls.0,
                    cls_pos: self.cls.1,
                    cls_comment: self.cls.2 > 0,
                    cls_ad: self.cls.3 > 0,
                    container1: self.containers.first().copied().unwrap_or(0),
                    container2: self.containers.get(1).copied().unwrap_or(0),
                });
            }
        }
        self.raw.clear();
        self.raw_has_text = false;
        self.chars_in_links = 0;
        self.tags_count = 0;
        self.verbatim = self.pre > 0;
    }

    fn append_text(&mut self, text: &str, normalize: bool) {
        let appended_len = if normalize {
            let n = normalize_whitespace(text);
            let len = n.chars().count();
            self.raw.push_str(&n);
            len
        } else {
            self.raw.push_str(text);
            text.chars().count()
        };
        self.raw_has_text = true;
        if self.link > 0 {
            self.chars_in_links += appended_len;
        }
    }

    unsafe fn element_attr(&self, node: *mut lxb_dom_node_t, name: &[u8]) -> String {
        unsafe {
            let mut size = 0usize;
            let val = lxb_dom_element_get_attribute(node.cast(), name.as_ptr(), name.len(), &mut size as *mut usize);
            if val.is_null() {
                String::new()
            } else {
                str_from_lxb_char_t(val, size).unwrap_or_default().to_lowercase()
            }
        }
    }

    unsafe fn element_class(&self, node: *mut lxb_dom_node_t) -> String {
        unsafe { self.element_attr(node, b"class") }
    }

    /// Iterative pre/post-order traversal — deep DOMs must not grow the call
    /// stack (a page nesting thousands of elements overflows recursion,
    /// especially on 2 MB worker-thread stacks).
    unsafe fn walk(&mut self, root: *mut lxb_dom_node_t) {
        unsafe {
            let mut node = (*root).first_child;
            while !node.is_null() {
                let mut descend = false;
                match (*node).type_ {
                    lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_TEXT => {
                        self.on_text(node);
                        self.br = false;
                    }
                    lxb_dom_node_type_t::LXB_DOM_NODE_TYPE_ELEMENT => {
                        let tag = (*node).local_name as lxb_tag_id_enum_t::Type;
                        if tag == LXB_TAG_TABLE && self.pre == 0 {
                            if let Some(text) = try_rewrite_table(node) {
                                // emit the rewritten table as ONE structured paragraph;
                                // the classifier still decides keep/drop (forum
                                // signature spec-boxes look identical to real
                                // infoboxes structurally, only content tells).
                                self.flush_paragraph();
                                self.flags.bump(LXB_TAG_TABLE, 1);
                                self.raw.push_str(&text);
                                self.raw_has_text = true;
                                self.flush_paragraph();
                                self.flags.bump(LXB_TAG_TABLE, -1);
                            } else if !is_skip_tag(tag) {
                                self.on_element_begin(tag, node);
                                descend = !(*node).first_child.is_null();
                                if !descend {
                                    self.on_element_end(tag);
                                }
                            }
                        } else if !is_skip_tag(tag) {
                            self.on_element_begin(tag, node);
                            descend = !(*node).first_child.is_null();
                            if !descend {
                                self.on_element_end(tag);
                            }
                        }
                    }
                    _ => {}
                }
                if descend {
                    self.depth += 1;
                    node = (*node).first_child;
                    continue;
                }
                // climb to the next sibling, closing elements on the way up
                loop {
                    if !(*node).next.is_null() {
                        node = (*node).next;
                        break;
                    }
                    node = (*node).parent;
                    if node == root || node.is_null() {
                        return;
                    }
                    self.depth -= 1;
                    let tag = (*node).local_name as lxb_tag_id_enum_t::Type;
                    self.on_element_end(tag);
                }
            }
        }
    }

    unsafe fn on_text(&mut self, node: *mut lxb_dom_node_t) {
        let content = unsafe { str_from_dom_node(node).unwrap_or_default() };
        if self.pre > 0 {
            self.verbatim = true;
            self.append_text(content, false);
            return;
        }
        if content.chars().all(char::is_whitespace) {
            // Whitespace between inline elements: keep a single separating space.
            if self.raw_has_text {
                self.append_text(" ", false);
            }
            return;
        }
        // Newlines inside a source text node are HTML pretty-printing; collapse them.
        let cleaned: String = content
            .chars()
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
            .collect();
        self.append_text(&cleaned, true);
    }

    unsafe fn on_element_begin(&mut self, tag: lxb_tag_id_enum_t::Type, node: *mut lxb_dom_node_t) {
        if matches!(tag, LXB_TAG_OL | LXB_TAG_UL) {
            let class = unsafe { self.element_class(node) };
            let emit = !STRUCTURAL_LIST.iter().any(|k| class.contains(k));
            self.list_stack.push((tag, 0, emit));
        }

        if is_paragraph_tag(tag) || (tag == LXB_TAG_BR && self.br) {
            if tag == LXB_TAG_BR {
                self.tags_count = self.tags_count.saturating_sub(1);
                // <br><br>: drop the line break the first <br> appended, it is a
                // paragraph separator, not content.
                while self.raw.ends_with('\n') {
                    self.raw.pop();
                }
            }
            self.flags.bump(tag, 1);
            self.flush_paragraph();
            self.flags.bump(tag, -1);
            if matches!(tag, LXB_TAG_PRE | LXB_TAG_TEXTAREA) {
                self.pre += 1;
                self.verbatim = true;
                if tag == LXB_TAG_TEXTAREA {
                    self.textarea += 1;
                }
            }
            self.flags.bump(tag, 1);
            self.br = false;
            // class/id + container context enter AFTER the flush: the closed
            // paragraph belongs to the outer context, what follows to this one
            self.enter_element_context(tag, node);
            return;
        }

        if is_separator_tag(tag) {
            if tag == LXB_TAG_LI {
                if let Some((ltag, count, emit)) = self.list_stack.last_mut() {
                    if *emit {
                        *count += 1;
                        let marker = if *ltag == LXB_TAG_OL {
                            format!("\n{}. ", count)
                        } else {
                            "\n- ".to_string()
                        };
                        self.raw.push_str(&marker);
                        self.raw_has_text = true;
                    } else {
                        self.append_text(" ", false);
                    }
                } else {
                    self.append_text(" ", false);
                }
            } else {
                self.append_text(" ", false);
            }
        }

        self.br = tag == LXB_TAG_BR;
        if self.br || tag == LXB_TAG_ADDRESS {
            self.raw.push('\n');
        } else if tag == LXB_TAG_A {
            self.link += 1;
        }
        self.tags_count += 1;
        self.flags.bump(tag, 1);
        self.enter_element_context(tag, node);
    }

    /// Track ancestor class/id signal tokens and block-container identity.
    /// Paired with [`Self::exit_element_context`]; called once per element.
    unsafe fn enter_element_context(&mut self, tag: lxb_tag_id_enum_t::Type, node: *mut lxb_dom_node_t) {
        let class = unsafe { self.element_attr(node, b"class") };
        let id = unsafe { self.element_attr(node, b"id") };
        let (mut neg, mut pos, mut com, mut ad) = (0u32, 0u32, 0u32, 0u32);
        for s in [&class, &id] {
            if !s.is_empty() {
                let (n, p, c, a) = class_token_hits(s);
                neg += n;
                pos += p;
                com += c as u32;
                ad += a as u32;
            }
        }
        self.cls.0 += neg;
        self.cls.1 += pos;
        self.cls.2 += com;
        self.cls.3 += ad;
        self.attr_stack.push((neg, pos, com, ad));
        if is_container_tag(tag) {
            self.container_counter += 1;
            let id = self.container_counter;
            self.containers.push(id);
        }
    }

    fn exit_element_context(&mut self, tag: lxb_tag_id_enum_t::Type) {
        if let Some((neg, pos, com, ad)) = self.attr_stack.pop() {
            self.cls.0 -= neg;
            self.cls.1 -= pos;
            self.cls.2 -= com;
            self.cls.3 -= ad;
        }
        if is_container_tag(tag) {
            self.containers.pop();
        }
    }

    fn on_element_end(&mut self, tag: lxb_tag_id_enum_t::Type) {
        if matches!(tag, LXB_TAG_OL | LXB_TAG_UL) {
            self.list_stack.pop();
        }
        if is_paragraph_tag(tag) {
            // flush BEFORE dropping pre/textarea state so the closing block
            // still carries its verbatim/from_textarea provenance
            self.flush_paragraph();
            self.flags.bump(tag, -1);
            if matches!(tag, LXB_TAG_PRE | LXB_TAG_TEXTAREA) && self.pre > 0 {
                self.pre -= 1;
                if tag == LXB_TAG_TEXTAREA && self.textarea > 0 {
                    self.textarea -= 1;
                }
                self.verbatim = self.pre > 0;
            }
        } else {
            self.flags.bump(tag, -1);
            if is_separator_tag(tag) {
                self.append_text(" ", false);
                // separator spacers alone must not turn an empty block into a paragraph
                if self.raw.chars().all(char::is_whitespace) {
                    self.raw_has_text = false;
                }
            }
            if tag == LXB_TAG_A && self.link > 0 {
                self.link -= 1;
            }
        }
        self.exit_element_context(tag);
    }
}

/// Block-level containers that define region identity for container-aware
/// classification.
#[inline]
fn is_container_tag(t: lxb_tag_id_enum_t::Type) -> bool {
    matches!(
        t,
        LXB_TAG_DIV
            | LXB_TAG_SECTION
            | LXB_TAG_ARTICLE
            | LXB_TAG_MAIN
            | LXB_TAG_ASIDE
            | LXB_TAG_NAV
            | LXB_TAG_HEADER
            | LXB_TAG_FOOTER
            | LXB_TAG_TABLE
            | LXB_TAG_UL
            | LXB_TAG_OL
            | LXB_TAG_DL
            | LXB_TAG_BLOCKQUOTE
            | LXB_TAG_TD
            | LXB_TAG_LI
            | LXB_TAG_FORM
    )
}

/// Tunable knobs for [`classify_paragraphs`].
#[derive(Debug, Clone)]
pub struct KeepPolicy {
    /// Keep a paragraph when its content probability reaches this value.
    pub threshold: f64,
    /// Rescue threshold for paragraphs sandwiched between two kept neighbours.
    pub neighbour_threshold: f64,
    /// Rescue threshold for headings directly above a kept paragraph.
    pub heading_threshold: f64,
    /// Drop kept paragraphs under `<nav>`/`<aside>` below this probability.
    pub nav_threshold: f64,
    /// Drop kept paragraphs shorter than this many words whose characters are
    /// almost entirely link text.
    pub max_link_words: usize,
    /// Link-density bound used with `max_link_words`.
    pub max_link_density: f64,
    /// Drop later near-duplicate kept prose paragraphs (repeated teasers,
    /// forum quotes). Reference corpora never repeat a paragraph.
    pub dedup: bool,
    /// Structured-page rescue: when the kept text is shorter than this many
    /// characters, greedily add the highest-probability remaining paragraphs
    /// until the floor is reached. Pages whose entire content is short, linky,
    /// label:value lines (profiles, directories, spec sheets) otherwise lose
    /// everything to the prose prior. 0 disables.
    pub min_content_chars: usize,
    /// Absolute probability floor below which the structured-page rescue will
    /// not add a paragraph.
    pub rescue_min_prob: f64,
}

impl Default for KeepPolicy {
    fn default() -> Self {
        KeepPolicy {
            threshold: 0.40,
            neighbour_threshold: 0.25,
            heading_threshold: 0.15,
            nav_threshold: 0.60,
            max_link_words: 10,
            max_link_density: 0.8,
            dedup: true,
            min_content_chars: 400,
            rescue_min_prob: 0.05,
        }
    }
}

/// Normalize for duplicate comparison: collapse whitespace, lowercase, fold
/// curly quotes, drop U+FFFD (same text in different encodings still matches).
fn dedup_norm(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_ws = false;
    for c in text.chars() {
        let c = match c {
            '\u{2019}' | '\u{2018}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' => '"',
            '\u{FFFD}' => continue,
            c => c,
        };
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
    out
}

const MINHASH_K: usize = 16;
const LSH_BANDS: usize = 4; // 4 bands x 4 rows

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// MinHash signature over 4-char shingles (one base hash per shingle, k
/// seed-mixed minima). Cheap: a single pass over the text.
fn minhash(text: &str) -> [u64; MINHASH_K] {
    let chars: Vec<char> = text.chars().collect();
    let mut sig = [u64::MAX; MINHASH_K];
    if chars.len() < 4 {
        let base = splitmix64(chars.iter().map(|&c| c as u64).fold(0, |a, c| a * 31 + c));
        for (j, s) in sig.iter_mut().enumerate() {
            *s = splitmix64(base ^ (j as u64).wrapping_mul(0xA5A5A5A5A5A5A5A5));
        }
        return sig;
    }
    for w in chars.windows(4) {
        let base = splitmix64(w.iter().map(|&c| c as u64).fold(0u64, |a, c| a.wrapping_mul(31) + c));
        for (j, s) in sig.iter_mut().enumerate() {
            let h = splitmix64(base ^ (j as u64).wrapping_mul(0xA5A5A5A5A5A5A5A5));
            if h < *s {
                *s = h;
            }
        }
    }
    sig
}

/// Mark later near-duplicate kept prose paragraphs as dropped, in document
/// order. Verbatim / punctuation-heavy (code-like) paragraphs are exempt:
/// code legitimately repeats.
///
/// Near-duplicate candidates are found via MinHash LSH banding (only
/// paragraphs sharing a band bucket get the expensive similarity check), so
/// documents with hundreds of kept paragraphs stay cheap.
fn dedup_kept(paragraphs: &[Paragraph], keep: &mut [bool]) {
    use std::collections::HashMap;
    let rows = MINHASH_K / LSH_BANDS;
    // band-hash -> indices into `seen`
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    let mut seen: Vec<String> = Vec::new();
    for (i, p) in paragraphs.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let n_chars = p.text.chars().count().max(1);
        let punct = p
            .text
            .chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
            .count();
        if p.verbatim || punct as f64 / n_chars as f64 > 0.13 {
            continue;
        }
        let n = dedup_norm(&p.text);
        if n.chars().count() < 12 {
            continue;
        }
        let sig = minhash(&n);
        let band_keys: Vec<(usize, u64)> = (0..LSH_BANDS)
            .map(|b| {
                let mut h = 0u64;
                for r in 0..rows {
                    h = splitmix64(h ^ sig[b * rows + r]);
                }
                (b, h)
            })
            .collect();
        let mut dup = false;
        // near-equal duplicates: only LSH band collisions get the ratio check
        let mut candidates: Vec<usize> = band_keys
            .iter()
            .filter_map(|k| buckets.get(k))
            .flatten()
            .copied()
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        for &c in &candidates {
            let s = &seen[c];
            let (a, b) = (n.len(), s.len());
            if a.abs_diff(b) * 16 <= a.max(b) && (n == *s || rapidfuzz::fuzz::ratio(n.chars(), s.chars()) >= 0.97) {
                dup = true;
                break;
            }
        }
        // containment in an earlier, longer paragraph (repeated teaser). Only
        // longer seen paragraphs can contain n; substring search is fast.
        if !dup && n.len() >= 40 {
            for s in &seen {
                if s.len() > n.len() && s.contains(&n) {
                    dup = true;
                    break;
                }
            }
        }
        if dup {
            keep[i] = false;
        } else {
            let idx = seen.len();
            seen.push(n);
            for k in band_keys {
                buckets.entry(k).or_default().push(idx);
            }
        }
    }
}

/// Decide keep/drop per paragraph from per-paragraph content probabilities
/// (e.g. from a text classifier) plus structural heuristics.
///
/// `probs[i]` is the probability that `paragraphs[i]` is main content. Verbatim
/// (`<pre>`/`<textarea>`) paragraphs are always kept. If nothing qualifies, the
/// highest-probability paragraph is kept so a document never comes back empty.
pub fn classify_paragraphs(paragraphs: &[Paragraph], probs: &[f64], policy: &KeepPolicy) -> Vec<bool> {
    let n = paragraphs.len();
    let mut keep: Vec<bool> = (0..n)
        .map(|i| {
            // verbatim (<pre>) blocks are intentional content — force-keep;
            // <textarea> is verbatim-formatted but form input, so it must earn
            // its keep from the classifier like any other paragraph.
            (paragraphs[i].verbatim && !paragraphs[i].from_textarea) || probs[i] >= policy.threshold
        })
        .collect();

    // Short, almost-all-link paragraphs are menus/breadcrumbs even when the
    // text model likes their wording.
    for (i, p) in paragraphs.iter().enumerate() {
        if keep[i]
            && !p.verbatim
            && p.word_count() < policy.max_link_words
            && p.link_density() > policy.max_link_density
        {
            keep[i] = false;
        }
    }

    // <nav>/<aside> ancestry needs extra confidence.
    for (i, p) in paragraphs.iter().enumerate() {
        if keep[i] && !p.verbatim && (p.dom_nav || p.dom_aside) && probs[i] < policy.nav_threshold {
            keep[i] = false;
        }
    }

    // Neighbour smoothing: content is contiguous, so a borderline paragraph
    // between two kept ones is almost always content too.
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if !keep[i] && probs[i] >= policy.neighbour_threshold && i > 0 && keep[i - 1] && i + 1 < n && keep[i + 1] {
                keep[i] = true;
                changed = true;
            }
        }
    }

    // A heading directly above kept content introduces it.
    for i in 0..n {
        if !keep[i] && paragraphs[i].heading && probs[i] >= policy.heading_threshold && i + 1 < n && keep[i + 1] {
            keep[i] = true;
        }
    }

    // Structured-page rescue: profiles, directories, and spec sheets consist
    // entirely of short label:value / link lines the prose prior rejects. When
    // the absolute-threshold pass keeps almost nothing, top up with the best
    // remaining paragraphs until the content floor is met.
    if policy.min_content_chars > 0 && n > 0 {
        let mut kept_chars: usize = keep
            .iter()
            .zip(paragraphs)
            .filter(|&(&k, _)| k)
            .map(|(_, p)| p.text.chars().count())
            .sum();
        if kept_chars < policy.min_content_chars {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            for i in order {
                if kept_chars >= policy.min_content_chars {
                    break;
                }
                if keep[i] || probs[i] < policy.rescue_min_prob {
                    continue;
                }
                if paragraphs[i].link_density() > 0.9 {
                    continue;
                }
                keep[i] = true;
                kept_chars += paragraphs[i].text.chars().count();
            }
        }
    }

    if n > 0 && !keep.iter().any(|&k| k) {
        let best = (0..n).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap();
        keep[best] = true;
    }

    if policy.dedup {
        dedup_kept(paragraphs, &mut keep);
    }
    keep
}

/// Map a char back to the CP1252 byte that decodes to it, if any.
fn cp1252_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    match cp {
        0x00..=0x7F | 0xA0..=0xFF => Some(cp as u8),
        // C1 codepoints: "sloppy cp1252" — pass through like latin-1
        0x80..=0x9F => Some(cp as u8),
        _ => Some(match c {
            '\u{20AC}' => 0x80, // €
            '\u{201A}' => 0x82, // ‚
            '\u{0192}' => 0x83, // ƒ
            '\u{201E}' => 0x84, // „
            '\u{2026}' => 0x85, // …
            '\u{2020}' => 0x86, // †
            '\u{2021}' => 0x87, // ‡
            '\u{02C6}' => 0x88, // ˆ
            '\u{2030}' => 0x89, // ‰
            '\u{0160}' => 0x8A, // Š
            '\u{2039}' => 0x8B, // ‹
            '\u{0152}' => 0x8C, // Œ
            '\u{017D}' => 0x8E, // Ž
            '\u{2018}' => 0x91, // '
            '\u{2019}' => 0x92, // '
            '\u{201C}' => 0x93, // "
            '\u{201D}' => 0x94, // "
            '\u{2022}' => 0x95, // •
            '\u{2013}' => 0x96, // –
            '\u{2014}' => 0x97, // —
            '\u{02DC}' => 0x98, // ˜
            '\u{2122}' => 0x99, // ™
            '\u{0161}' => 0x9A, // š
            '\u{203A}' => 0x9B, // ›
            '\u{0153}' => 0x9C, // œ
            '\u{017E}' => 0x9E, // ž
            '\u{0178}' => 0x9F, // Ÿ
            _ => return None,
        }),
    }
}

/// The char a lone CP1252 byte decodes to (inverse of [`cp1252_byte`]).
fn cp1252_char(b: u8) -> char {
    match b {
        0x80 => '\u{20AC}',
        0x82 => '\u{201A}',
        0x83 => '\u{0192}',
        0x84 => '\u{201E}',
        0x85 => '\u{2026}',
        0x86 => '\u{2020}',
        0x87 => '\u{2021}',
        0x88 => '\u{02C6}',
        0x89 => '\u{2030}',
        0x8A => '\u{0160}',
        0x8B => '\u{2039}',
        0x8C => '\u{0152}',
        0x8E => '\u{017D}',
        0x91 => '\u{2018}',
        0x92 => '\u{2019}',
        0x93 => '\u{201C}',
        0x94 => '\u{201D}',
        0x95 => '\u{2022}',
        0x96 => '\u{2013}',
        0x97 => '\u{2014}',
        0x98 => '\u{02DC}',
        0x99 => '\u{2122}',
        0x9A => '\u{0161}',
        0x9B => '\u{203A}',
        0x9C => '\u{0153}',
        0x9E => '\u{017E}',
        0x9F => '\u{0178}',
        b => b as char, // ASCII, latin-1 range, and sloppy C1 pass-through
    }
}

/// Decode *bytes* as UTF-8 where valid; each byte that is not part of a valid
/// UTF-8 sequence reverts to its CP1252 char (the identity for bytes that came
/// from [`cp1252_byte`]), so this transformation never loses information.
fn decode_utf8_or_cp1252(bytes: &[u8], out: &mut String) {
    let mut i = 0;
    while i < bytes.len() {
        match std::str::from_utf8(&bytes[i..]) {
            Ok(valid) => {
                out.push_str(valid);
                break;
            }
            Err(e) => {
                let good = e.valid_up_to();
                out.push_str(std::str::from_utf8(&bytes[i..i + good]).unwrap());
                i += good;
                out.push(cp1252_char(bytes[i]));
                i += 1;
            }
        }
    }
}

/// Count occurrences of tell-tale UTF-8-read-as-CP1252 sequences.
fn mojibake_signatures(s: &str) -> usize {
    let mut n = 0;
    let chars: Vec<char> = s.chars().collect();
    for w in chars.windows(2) {
        match (w[0], w[1]) {
            ('â', '\u{20AC}') // â€
            | ('â', '\u{201A}') // â‚ (as in â‚¬ = €)
            | ('Ã', '\u{0192}')
            | ('Ã', '\u{201A}')
            | ('Ã', '©')
            | ('Ã', '¨')
            | ('Ã', '¶')
            | ('Ã', '°')
            | ('Ã', '¢')
            | ('Â', '«')
            | ('Â', '»')
            | ('Â', '\u{A0}') => n += 1,
            _ => {}
        }
    }
    n
}

/// Reverse UTF-8-bytes-read-as-CP1252 mojibake (``â€™`` -> ``’``), including
/// the double-encoded variant, on documents that carry a tell-tale signature.
/// Clean documents return `None` untouched; repairs are segment-wise (runs of
/// CP1252-representable chars), strictly validated, and only accepted when
/// they reduce the signature count — so a false positive cannot corrupt text.
pub fn repair_mojibake(text: &str) -> Option<String> {
    if mojibake_signatures(text) == 0 {
        return None;
    }
    let mut current = text.to_string();
    let mut improved = false;
    for _ in 0..3 {
        let before = mojibake_signatures(&current);
        if before == 0 {
            break;
        }
        let mut out = String::with_capacity(current.len());
        let mut run: Vec<u8> = Vec::new();
        let flush = |run: &mut Vec<u8>, out: &mut String| {
            decode_utf8_or_cp1252(run, out);
            run.clear();
        };
        for c in current.chars() {
            match cp1252_byte(c) {
                Some(b) => run.push(b),
                None => {
                    flush(&mut run, &mut out);
                    out.push(c);
                }
            }
        }
        flush(&mut run, &mut out);
        if mojibake_signatures(&out) < before {
            current = out;
            improved = true;
        } else {
            break;
        }
    }
    improved.then_some(current)
}

/// Recover page content from `<title>` + the longest `description`-family meta
/// tag. JS-rendered SPAs often ship their entire lede this way while the DOM
/// body holds only chrome; the reference gold follows the meta text there.
pub fn meta_fallback(tree: &HTMLTree) -> Option<String> {
    use crate::parse::html::dom::traits::Element;
    let head = tree.head()?;
    let mut best = String::new();
    for el in head.get_elements_by_tag_name("meta").iter() {
        let key = el
            .attribute("name")
            .or_else(|| el.attribute("property"))
            .unwrap_or_default()
            .to_lowercase();
        if matches!(key.as_str(), "description" | "og:description" | "twitter:description") {
            if let Some(content) = el.attribute("content") {
                let content = normalize_whitespace(content.trim());
                if content.chars().count() > best.chars().count() {
                    best = content;
                }
            }
        }
    }
    if best.is_empty() {
        return None;
    }
    let title = tree.title().map(|t| normalize_whitespace(t.trim())).unwrap_or_default();
    Some(if title.is_empty() {
        best
    } else {
        format!("{}\n\n{}", title, best)
    })
}

/// Segment *tree* into block-level paragraphs with structural features.
pub fn extract_paragraphs(tree: &HTMLTree) -> Vec<Paragraph> {
    let Some(body) = tree.body() else {
        return Vec::new();
    };
    let root = *body.node_ptr_();
    if root.is_null() {
        return Vec::new();
    }
    let mut walker = Walker::new();
    unsafe {
        walker.walk(root);
    }
    walker.flush_paragraph();
    walker.paragraphs
}

/// End-to-end main-content extraction: segment, score with *scorer* (a
/// content-probability model over the paragraphs, e.g. a text classifier or a
/// [`DecisionForest`] over [`stack_features`]), classify with *policy*, join
/// kept paragraphs, and fall back to `<title>` + meta description when the
/// body yields almost nothing (JS-rendered pages).
pub fn extract_main_text<S>(tree: &HTMLTree, scorer: S, policy: &KeepPolicy) -> String
where
    S: FnOnce(&[Paragraph]) -> Vec<f64>,
{
    let paragraphs = extract_paragraphs(tree);
    let probs = scorer(&paragraphs);
    debug_assert_eq!(probs.len(), paragraphs.len());
    let keep = classify_paragraphs(&paragraphs, &probs, policy);
    let out = assemble(&paragraphs, &keep);
    if out.chars().count() < 150 {
        if let Some(fb) = meta_fallback(tree) {
            if fb.chars().count() > out.chars().count() {
                return fb;
            }
        }
    }
    out
}

/// Join the paragraphs marked `true` in *keep* with blank lines.
pub fn assemble(paragraphs: &[Paragraph], keep: &[bool]) -> String {
    paragraphs
        .iter()
        .zip(keep)
        .filter(|&(_, &k)| k)
        .map(|(p, _)| p.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_segmentation() {
        let tree =
            HTMLTree::parse("<html><body><p>Hello <a href='#'>world</a></p><div>Second block</div></body></html>")
                .unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].text, "Hello world");
        assert_eq!(paras[0].chars_in_links, 5);
        assert_eq!(paras[1].text, "Second block");
    }

    #[test]
    fn test_list_markers() {
        let tree =
            HTMLTree::parse("<html><body><ul><li>one</li><li>two</li></ul><ol><li>first</li></ol></body></html>")
                .unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras[0].text, "- one\n- two");
        // <ol> is not a paragraph break tag; its items join the following block
        assert!(paras.iter().any(|p| p.text.contains("1. first")));
    }

    #[test]
    fn test_structural_list_no_markers() {
        let tree =
            HTMLTree::parse("<html><body><ul class='nav-menu'><li>Home</li><li>About</li></ul></body></html>").unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras[0].text, "Home About");
        assert!(paras[0].dom_list);
    }

    #[test]
    fn test_br_br_paragraph_break() {
        let tree = HTMLTree::parse("<html><body>first<br><br>second</body></html>").unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].text, "first");
        assert_eq!(paras[1].text, "second");
    }

    #[test]
    fn test_single_br_newline() {
        let tree = HTMLTree::parse("<html><body>line one<br>line two</body></html>").unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].text, "line one\nline two");
    }

    #[test]
    fn test_pre_verbatim() {
        let tree = HTMLTree::parse("<html><body><pre>  indented\n    code\n</pre></body></html>").unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 1);
        assert!(paras[0].verbatim);
        assert_eq!(paras[0].text, "  indented\n    code");
    }

    #[test]
    fn test_script_style_skipped() {
        let tree =
            HTMLTree::parse("<html><body><p>keep</p><script>var x = 1;</script><style>.a{}</style></body></html>")
                .unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 1);
        assert_eq!(paras[0].text, "keep");
    }

    #[test]
    fn test_table_row_single_paragraph() {
        let tree = HTMLTree::parse(
            "<html><body><table><tr><td>Bedrooms:</td><td>4</td></tr><tr><td>Baths</td></tr></table></body></html>",
        )
        .unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 2);
        assert_eq!(paras[0].text, "Bedrooms: 4");
        assert!(paras[0].dom_table);
    }

    #[test]
    fn test_frameset_document_no_crash() {
        // A frameset document has no body element; lexbor leaves the document's
        // body pointer NULL. Regression test for a segfault on such pages.
        let tree = HTMLTree::parse(
            "<html><head><title>T</title><meta name=\"keywords\" content=\"\"</head>\
             <frameset rows=\"100%\", *\" border=\"0\"><frame src=\"http://example.com/\" \
             name=\"F\"></frameset></html>",
        )
        .unwrap();
        assert!(tree.body().is_none());
        assert!(extract_paragraphs(&tree).is_empty());
    }

    #[test]
    fn test_classify_policy_pipeline() {
        let tree = HTMLTree::parse(
            "<html><body><h1>Title</h1><p>Real content paragraph.</p>\
             <pre>verbatim code</pre>\
             <textarea>form input text</textarea>\
             <div><a href='/'>Home</a> <a href='/x'>About</a></div></body></html>",
        )
        .unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 5);
        assert!(paras[2].verbatim && !paras[2].from_textarea);
        assert!(paras[3].verbatim && paras[3].from_textarea);
        // heading low-prob but rescued above kept content; pre force-kept even
        // at prob 0; textarea NOT force-kept at prob 0; linky div gated out.
        let probs = vec![0.3, 0.9, 0.0, 0.0, 0.9];
        let policy = KeepPolicy {
            min_content_chars: 0,
            ..KeepPolicy::default()
        };
        let keep = classify_paragraphs(&paras, &probs, &policy);
        assert_eq!(keep, vec![true, true, true, false, false]);
    }

    #[test]
    fn test_structured_page_rescue() {
        // A profile-style page: everything short and structured, nothing
        // passing the absolute threshold. The content floor tops up with the
        // best-scoring paragraphs instead of returning near-nothing.
        let tree = HTMLTree::parse(
            "<html><body><h1>Jane Doe</h1><p>Professor of Chemistry</p>\
             <p>Phone: 555-0100</p><p>Email: jane@example.edu</p></body></html>",
        )
        .unwrap();
        let paras = extract_paragraphs(&tree);
        assert_eq!(paras.len(), 4);
        let probs = vec![0.35, 0.30, 0.20, 0.02];
        let keep = classify_paragraphs(&paras, &probs, &KeepPolicy::default());
        // floor unmet -> rescue adds all above rescue_min_prob, skips 0.02
        assert_eq!(keep, vec![true, true, true, false]);
        let no_rescue = KeepPolicy {
            min_content_chars: 0,
            ..KeepPolicy::default()
        };
        let keep2 = classify_paragraphs(&paras, &probs, &no_rescue);
        // without the rescue the never-empty fallback keeps only the best one
        assert_eq!(keep2.iter().filter(|&&k| k).count(), 1);
    }

    #[test]
    fn test_dedup_drops_repeated_prose() {
        let text = "This repeated teaser sentence should only ever be kept a single time here.";
        let html = format!(
            "<html><body><p>{t}</p><p>Fresh middle paragraph with different words.</p><p>{t}</p></body></html>",
            t = text
        );
        let tree = HTMLTree::parse(&html).unwrap();
        let paras = extract_paragraphs(&tree);
        let keep = classify_paragraphs(&paras, &[0.9, 0.9, 0.9], &KeepPolicy::default());
        assert_eq!(keep, vec![true, true, false]);
    }

    #[test]
    fn test_repair_mojibake() {
        // single UTF-8-as-CP1252: â€™ is the CP1252 reading of ’ (E2 80 99)
        assert_eq!(repair_mojibake("donâ€™t stop"), Some("don\u{2019}t stop".to_string()));
        // Ã© is the CP1252 reading of é
        assert_eq!(repair_mojibake("cafÃ© au lait"), Some("café au lait".to_string()));
        // double-encoded: Ã¢â‚¬â„¢ -> â€™ -> ’
        assert_eq!(repair_mojibake("donÃ¢â‚¬â„¢t"), Some("don\u{2019}t".to_string()));
        // clean text (even with accents) is untouched
        assert_eq!(repair_mojibake("café déjà vu"), None);
        assert_eq!(repair_mojibake("plain ascii"), None);
        // non-CP1252 chars segment the repair without breaking it
        assert_eq!(repair_mojibake("日本語 and donâ€™t"), Some("日本語 and don\u{2019}t".to_string()));
    }

    #[test]
    fn test_meta_fallback() {
        let tree = HTMLTree::parse(
            "<html><head><title>Page Title</title>\
             <meta name=\"description\" content=\"A description of the page.\">\
             </head><body></body></html>",
        )
        .unwrap();
        assert_eq!(meta_fallback(&tree).unwrap(), "Page Title\n\nA description of the page.");
    }

    #[test]
    fn test_heading_flag() {
        let tree = HTMLTree::parse("<html><body><h1>Title</h1><p>Body</p></body></html>").unwrap();
        let paras = extract_paragraphs(&tree);
        assert!(paras[0].heading);
        assert!(!paras[1].heading);
    }
}
