#!/usr/bin/env python3
"""
parse_c3pa_sentences.py

Builds sentence-level candidate pairs from the C3PA dataset (Htmls and Annotations).
Per instructions, original C3PA labels are preserved verbatim without CPRA label mapping.
"""

import os
import sys
import glob
import csv
import re
import unicodedata
import argparse
import platform
import bs4
from bs4 import BeautifulSoup, NavigableString

# Try loading spaCy
try:
    import spacy
    try:
        NLP = spacy.blank("en")
        NLP.add_pipe("sentencizer")
    except Exception:
        NLP = None
except ImportError:
    NLP = None

SEGMENTER_NAME = "spacy-sentencizer" if NLP is not None else "regex-fallback"
if NLP is None:
    print(
        "Warning: spaCy is unavailable; using the regex sentence splitter. "
        "Output will differ from spaCy-based runs.",
        file=sys.stderr,
    )

# Non-content elements. These are removed before text extraction.
NON_CONTENT_TAGS = [
    "script", "style", "noscript", "head", "header", "footer", "nav", "svg",
    "iframe", "template", "object", "embed", "canvas",
]

# Elements that define readable text blocks. Leaf-level walking preserves
# document order and avoids emitting a parent and its children twice.
BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "td", "th",
    "article", "section", "blockquote", "dd", "dt", "figcaption", "pre",
    "main", "ul", "ol", "table", "tbody", "thead", "tfoot", "form",
    "fieldset", "details", "summary", "button", "select", "option", "label",
}

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "th"}
UI_TAGS = {"button", "select", "option", "summary", "label"}
NON_SENTENCE_KINDS = {"heading", "cell", "ui"}

MIN_SENTENCE_WORDS = 3
LONG_SENTENCE_WORDS = 5
CASE_B_MIN_WORDS = 5
MAX_SENTENCE_CHARS = 5000


def normalize_text(text: str) -> str:
    """Normalizes text for precision containment matching."""
    if not text:
        return ""
    # Normalize unicode to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Replace non-breaking spaces and tabs
    text = text.replace("\xa0", " ").replace("\t", " ")
    # Normalize quotes and dashes
    quotes_map = {
        "“": '"', "”": '"', "‘": "'", "’": "'", "`": "'", "–": "-", "—": "-"
    }
    for orig, repl in quotes_map.items():
        text = text.replace(orig, repl)
    # Lowercase and collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _block_kind(tag_name: str) -> str:
    """Classifies a block element so non-prose text can be filtered later."""
    if tag_name in HEADING_TAGS:
        return "heading"
    if tag_name == "td":
        return "cell"
    if tag_name == "li":
        return "list"
    if tag_name in UI_TAGS:
        return "ui"
    return "prose"


def _inline_text(node) -> str:
    """Collects text from an inline node without adding separators."""
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif getattr(child, "name", None) == "br":
            parts.append("\n")
        elif getattr(child, "name", None) in NON_CONTENT_TAGS:
            continue
        else:
            parts.append(_inline_text(child))
    return "".join(parts)


def _append_block(text: str, kind: str, blocks: list[dict]) -> None:
    for line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        # Strip residual inline markup from prose, but leave raw tag/script
        # blobs intact so the machine-text filter can reject them.
        if line and not line.startswith("<"):
            line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            blocks.append({"text": line, "kind": kind})


def _walk_blocks(node, kind: str, blocks: list[dict]) -> None:
    """Walks the DOM, emitting direct text and recursing into block children.

    Text directly inside a block keeps that block's kind; inline children are
    folded into the same buffer so anchor/span text stays in its sentence.
    """
    buffer = []

    def flush() -> None:
        if buffer:
            _append_block("".join(buffer), kind, blocks)
            buffer.clear()

    for child in node.children:
        name = getattr(child, "name", None)
        if isinstance(child, NavigableString):
            buffer.append(str(child))
        elif name in NON_CONTENT_TAGS:
            continue
        elif name == "br":
            buffer.append("\n")
        elif name in BLOCK_TAGS:
            flush()
            _walk_blocks(child, _block_kind(name), blocks)
            flush()
        else:
            buffer.append(_inline_text(child))
    flush()


def extract_readable_blocks(html_content: str) -> list[dict]:
    """Extracts readable text blocks with their source element kind."""
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()
    for element in soup.find_all(attrs={"hidden": True}):
        element.decompose()

    blocks = []
    root = soup.body or soup
    _walk_blocks(root, "prose", blocks)
    return blocks


def extract_readable_text_from_html(html_content: str) -> str:
    """Backward-compatible flat text extraction."""
    return "\n".join(block["text"] for block in extract_readable_blocks(html_content))


def looks_like_machine_text(text: str) -> bool:
    """Heuristic filter for URLs, encoded IDs, and embedded JSON blobs.

    A normal sentence that happens to contain a URL is not machine text; the
    token-density checks only apply to short, blob-like strings.
    """
    stripped = text.strip()
    if len(stripped) < 200:
        return False
    if re.match(r"^(https?://|www\.)\S+$", stripped, re.IGNORECASE):
        return True
    if stripped.startswith("{"):
        return True
    if stripped.startswith("(function") or stripped.lower().startswith("<script"):
        return True
    if stripped.startswith("<") and stripped.count("<") >= 2 and stripped.count(">") >= 2:
        return True

    tokens = stripped.split()
    if len(tokens) <= 8:
        percent_encoded = len(re.findall(r"%[0-9A-Fa-f]{2}", stripped))
        uuid_like = len(re.findall(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-", stripped))
        if percent_encoded >= 3 or uuid_like >= 2:
            return True
        alnum_space = sum(ch.isalnum() or ch.isspace() for ch in stripped)
        if alnum_space / len(stripped) < 0.55:
            return True

    return False


def classify_sentence(text: str, kind: str) -> tuple[bool, str]:
    """Decides whether an extracted block line is usable as a sentence.

    Headings, table cells, and UI controls are not sentences. Everything else
    must have enough words and either terminal punctuation or enough length to
    be a plausible full sentence.
    """
    stripped = text.strip()
    if not stripped:
        return False, "empty"
    if kind in NON_SENTENCE_KINDS:
        return False, "non_sentence_block:%s" % kind
    if not any(ch.isalpha() for ch in stripped):
        return False, "no_alphabetic_text"
    if len(stripped) > MAX_SENTENCE_CHARS:
        return False, "excessive_length"
    if looks_like_machine_text(stripped):
        return False, "machine_text"

    words = stripped.split()
    if len(words) < MIN_SENTENCE_WORDS:
        return False, "too_few_words"
    if stripped[-1] in ".!?":
        return True, ""
    if len(words) <= 10 and stripped.upper() == stripped and any(ch.isalpha() for ch in stripped):
        return False, "all_caps_heading"
    if len(words) <= 10 and stripped[-1] not in ".!?:":
        capitalized = sum(1 for word in words if word[:1].isupper())
        if capitalized / len(words) >= 0.6:
            return False, "title_like_heading"
    if len(words) >= LONG_SENTENCE_WORDS:
        return True, ""
    return False, "no_terminal_punctuation"


def segment_sentences(text: str) -> list[str]:
    """Segments readable text into full grammatical sentences."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = []

    if NLP:
        for line in lines:
            doc = NLP(line)
            for sent in doc.sents:
                s_text = sent.text.strip()
                if s_text and len(s_text) > 1:
                    sentences.append(s_text)
    else:
        # Fallback sentencizer if spaCy model unavailable
        sentence_end = re.compile(r'(?<=[.!?])\s+')
        for line in lines:
            for s in sentence_end.split(line):
                s_text = s.strip()
                if s_text and len(s_text) > 1:
                    sentences.append(s_text)

    return sentences


def load_annotations(csv_path: str):
    """Loads and normalizes C3PA annotations from CSV.

    Returns (annotations, source_rows, skipped_empty). Rows with no usable
    text or label are counted rather than silently disappearing.
    """
    annotations = []
    source_rows = 0
    skipped_empty = 0
    if not os.path.exists(csv_path):
        return annotations, source_rows, skipped_empty

    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            source_rows += 1
            ranumb = row.get("RANumb", "").strip()
            text_orig = row.get("Text", "")
            label_orig = row.get("Label", "").strip()

            if not text_orig or not label_orig:
                skipped_empty += 1
                continue

            text_norm = normalize_text(text_orig)
            if not text_norm:
                skipped_empty += 1
                continue

            annotations.append({
                "annotation_id": row_idx,
                "ranumb": ranumb,
                "annotation_original": text_orig,
                "annotation_normalized": text_norm,
                "c3pa_label": label_orig,
                "source_csv": csv_path
            })

    return annotations, source_rows, skipped_empty


def _source_path_label(path: str, dataset_dir: str) -> str:
    """Returns a dataset-relative source path that is stable across invocations."""
    dataset_abs = os.path.abspath(dataset_dir)
    path_abs = os.path.abspath(path)
    rel = os.path.relpath(path_abs, dataset_abs)
    if rel.startswith(".."):
        return path_abs.replace(os.sep, "/")
    root = os.path.basename(os.path.normpath(dataset_abs))
    return os.path.join(root, rel).replace(os.sep, "/")


def process_document(doc_id: str, html_path: str, csv_path: str, dataset_dir: str = "."):
    """Processes a single document (HTML + CSV annotations)."""
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    source_html = _source_path_label(html_path, dataset_dir)
    source_csv = _source_path_label(csv_path, dataset_dir)

    raw_sentences = []
    for block in extract_readable_blocks(html_content):
        for sentence_text in segment_sentences(block["text"]):
            raw_sentences.append((sentence_text, block["kind"]))

    sentences = []
    fragments = []
    idx = 0
    for s_orig, kind in raw_sentences:
        s_norm = normalize_text(s_orig)
        if not s_norm:
            continue
        idx += 1
        sentence = {
            "doc_id": doc_id,
            "sentence_id": f"{doc_id}_S{idx}",
            "sentence_original": s_orig,
            "sentence_normalized": s_norm,
            "sentence_kind": kind,
            "source_html": source_html
        }
        eligible, reason = classify_sentence(s_orig, kind)
        if eligible:
            sentences.append(sentence)
        else:
            fragments.append({
                "doc_id": doc_id,
                "sentence_id": sentence["sentence_id"],
                "sentence_text": s_orig,
                "sentence_kind": kind,
                "exclusion_reason": reason,
                "source_html": source_html,
                "source_annotation_file": source_csv
            })

    annotations, source_rows, skipped_empty = load_annotations(csv_path)

    aligned_matches = []
    unmatched = []
    ambiguous = []
    sentence_tokens = {
        sent["sentence_id"]: set(sent["sentence_normalized"].split())
        for sent in sentences
    }

    for ann in annotations:
        ann_norm = ann["annotation_normalized"]
        ann_tokens = set(ann_norm.split())

        # 1. Case A: Sub-sentence fragment contained in sentence(s)
        case_a_matches = [s for s in sentences if ann_norm in s["sentence_normalized"]]
        if len(case_a_matches) == 1:
            aligned_matches.append((ann, case_a_matches[0], "annotation_in_sentence"))
            continue
        elif len(case_a_matches) > 1:
            # Short fragment appearing in multiple distinct sentences is ambiguous
            ambiguous.append((ann, [s["sentence_id"] for s in case_a_matches]))
            continue

        # 2. Case B: Multi-sentence paragraph span containing full sentences
        case_b_matches = [
            s for s in sentences
            if s["sentence_normalized"] in ann_norm
            and len(s["sentence_normalized"].split()) >= CASE_B_MIN_WORDS
        ]
        if case_b_matches:
            for sent in case_b_matches:
                aligned_matches.append((ann, sent, "sentence_in_annotation_paragraph"))
            continue

        # 3. Case C: Partial overlap for cross-sentence boundary selections.
        # The score is containment of the annotation in the sentence, not
        # Jaccard. Ties are reported as ambiguous instead of picking the first.
        if len(ann_tokens) >= 3:
            best_count = 0
            best_sentences = []
            for sent in sentences:
                intersection = ann_tokens.intersection(
                    sentence_tokens[sent["sentence_id"]]
                )
                if not intersection:
                    continue
                overlap = len(intersection) / len(ann_tokens)
                if overlap < 0.55:
                    continue
                if len(intersection) > best_count:
                    best_count = len(intersection)
                    best_sentences = [sent]
                elif len(intersection) == best_count:
                    best_sentences.append(sent)
            if len(best_sentences) == 1:
                aligned_matches.append((ann, best_sentences[0], "partial_overlap"))
                continue
            if len(best_sentences) > 1:
                ambiguous.append((ann, [s["sentence_id"] for s in best_sentences]))
                continue

        # If no alignment could be established
        unmatched.append(ann)

    # Aggregate sentence evidence
    evidence = {
        sent["sentence_id"]: {
            "doc_id": doc_id,
            "sentence_id": sent["sentence_id"],
            "sentence_text": sent["sentence_original"],
            "sentence_kind": sent["sentence_kind"],
            "source_html": source_html,
            "source_annotation_file": source_csv,
            "verbatim_labels": set(),
            "annotators": set(),
            "source_annotations": [],
            "alignment_types": set(),
            "has_ambiguous": False
        }
        for sent in sentences
    }

    for ann, sent, align_type in aligned_matches:
        sid = sent["sentence_id"]
        evidence[sid]["verbatim_labels"].add(ann["c3pa_label"])
        evidence[sid]["annotators"].add(ann["ranumb"])
        evidence[sid]["source_annotations"].append(ann["annotation_original"])
        evidence[sid]["alignment_types"].add(align_type)

    return {
        "doc_id": doc_id,
        "sentences": sentences,
        "fragments": fragments,
        "annotations": annotations,
        "source_rows": source_rows,
        "skipped_empty": skipped_empty,
        "aligned_matches": aligned_matches,
        "aligned_annotation_ids": {
            ann["annotation_id"] for ann, _, _ in aligned_matches
        },
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "evidence": list(evidence.values())
    }


def run_pipeline(dataset_dir: str, output_dir: str, single_doc: str = None, export_master: bool = False):
    """Runs pipeline over the dataset documents and returns diagnostic counts."""
    if single_doc and output_dir == "output":
        safe_doc = re.sub(r"[^A-Za-z0-9_.-]+", "_", single_doc).strip("_")
        output_dir = os.path.join("output", "sanity", safe_doc)
        print(f"Sanity-check mode: writing to {output_dir} (main output/ is untouched)")

    os.makedirs(output_dir, exist_ok=True)

    doc_pairs = []

    for category in ["DB", "WS"]:
        csv_dir = os.path.join(dataset_dir, "Annotations", category)
        html_dir = os.path.join(dataset_dir, "Htmls", category)

        if not os.path.exists(csv_dir) or not os.path.exists(html_dir):
            continue

        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        for csv_path in sorted(csv_files):
            num = os.path.splitext(os.path.basename(csv_path))[0]
            html_path = os.path.join(html_dir, f"{num}.html")
            if os.path.exists(html_path):
                doc_id = f"{category}_{num}"
                if single_doc is None or single_doc == doc_id or single_doc == f"{category}/{num}":
                    doc_pairs.append((doc_id, html_path, csv_path))

    if not doc_pairs:
        print(f"No documents matched {single_doc!r} under {dataset_dir}", file=sys.stderr)
        return None

    print(f"Processing {len(doc_pairs)} documents...")

    total_docs = len(doc_pairs)
    total_sentences = 0
    total_fragments = 0
    total_source_rows = 0
    total_skipped_empty = 0
    total_annotations = 0
    total_aligned_annotations = 0
    total_aligned_links = 0
    total_unmatched = 0
    total_ambiguous = 0

    all_single_candidates = []
    high_conf_candidates = []
    multi_label_candidates = []
    unannotated_candidates = []
    all_sentences_master = []
    all_fragments = []

    all_unmatched = []
    all_ambiguous = []

    sentences_no_labels = 0

    for doc_id, html_path, csv_path in doc_pairs:
        res = process_document(doc_id, html_path, csv_path, dataset_dir=dataset_dir)

        total_sentences += len(res["sentences"]) + len(res["fragments"])
        total_fragments += len(res["fragments"])
        total_source_rows += res["source_rows"]
        total_skipped_empty += res["skipped_empty"]
        total_annotations += len(res["annotations"])
        total_aligned_annotations += len(res["aligned_annotation_ids"])
        total_aligned_links += len(res["aligned_matches"])
        total_unmatched += len(res["unmatched"])
        total_ambiguous += len(res["ambiguous"])

        all_unmatched.extend([{**ann, "doc_id": doc_id} for ann in res["unmatched"]])
        all_ambiguous.extend([{**ann, "doc_id": doc_id, "matched_sentences": ";".join(sids)} for ann, sids in res["ambiguous"]])

        for fragment in res["fragments"]:
            all_fragments.append({
                "doc_id": fragment["doc_id"],
                "sentence_id": fragment["sentence_id"],
                "sentence_text": fragment["sentence_text"],
                "sentence_category": "fragment",
                "sentence_kind": fragment["sentence_kind"],
                "verbatim_label": "",
                "verbatim_labels": "",
                "annotator_count": 0,
                "annotators": "",
                "source_annotation_count": 0,
                "alignment_type": "",
                "exclusion_reason": fragment["exclusion_reason"],
                "source_html": fragment["source_html"],
                "source_annotation_file": fragment["source_annotation_file"]
            })

        for ev in res["evidence"]:
            v_labels = sorted(list(ev["verbatim_labels"]))
            annotators = sorted(list(ev["annotators"]))
            align_types = sorted(list(ev["alignment_types"]))

            category = "unannotated" if len(v_labels) == 0 else "single_label" if len(v_labels) == 1 else "multi_label"

            row = {
                "doc_id": ev["doc_id"],
                "sentence_id": ev["sentence_id"],
                "sentence_text": ev["sentence_text"],
                "sentence_category": category,
                "sentence_kind": ev["sentence_kind"],
                "verbatim_label": v_labels[0] if len(v_labels) == 1 else "MULTI_LABEL" if len(v_labels) > 1 else "",
                "verbatim_labels": ";".join(v_labels),
                "annotator_count": len(annotators),
                "annotators": ";".join(annotators),
                "source_annotation_count": len(ev["source_annotations"]),
                "alignment_type": ";".join(align_types),
                "exclusion_reason": "",
                "source_html": ev["source_html"],
                "source_annotation_file": ev["source_annotation_file"]
            }

            if len(v_labels) == 0:
                sentences_no_labels += 1
                unannotated_candidates.append(row)
            elif len(v_labels) == 1:
                all_single_candidates.append(row)
                if len(annotators) >= 2:
                    high_conf_candidates.append(row)
            else:
                multi_label_candidates.append(row)

    # Master export includes candidate tiers and excluded fragments.
    # high_conf is a subset of all_single, so do not add it separately.
    all_sentences_master = (
        list(all_single_candidates) + list(multi_label_candidates)
        + list(unannotated_candidates) + list(all_fragments)
    )

    def sentence_sort_key(row):
        sid = row["sentence_id"]
        try:
            number = int(sid.rsplit("_S", 1)[1])
        except (IndexError, ValueError):
            number = 0
        return (row["doc_id"], number)

    all_sentences_master.sort(key=sentence_sort_key)

    # Write output CSVs
    fieldnames = [
        "doc_id", "sentence_id", "sentence_text", "sentence_category", "sentence_kind",
        "verbatim_label", "verbatim_labels", "annotator_count", "annotators",
        "source_annotation_count", "alignment_type", "exclusion_reason",
        "source_html", "source_annotation_file"
    ]

    def write_csv(filename, rows, fields):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    unmatched_fields = ["doc_id", "annotation_id", "ranumb", "annotation_original", "c3pa_label", "source_csv"]
    ambiguous_fields = ["doc_id", "annotation_id", "ranumb", "annotation_original", "c3pa_label", "matched_sentences", "source_csv"]

    write_csv("single_label_sentences_all.csv", all_single_candidates, fieldnames)
    write_csv("single_label_sentences_high_confidence.csv", high_conf_candidates, fieldnames)
    write_csv("multi_label_sentences.csv", multi_label_candidates, fieldnames)
    write_csv("unannotated_sentences.csv", unannotated_candidates, fieldnames)
    write_csv("fragments_excluded.csv", all_fragments, fieldnames)
    write_csv("annotations_unmatched.csv", all_unmatched, unmatched_fields)
    write_csv("annotations_ambiguous.csv", all_ambiguous, ambiguous_fields)

    if export_master:
        write_csv("all_parsed_sentences.csv", all_sentences_master, fieldnames)
    else:
        master_path = os.path.join(output_dir, "all_parsed_sentences.csv")
        if os.path.exists(master_path):
            try:
                os.remove(master_path)
            except OSError:
                pass

    # Clean up old file names if present
    for old_file in ["candidate_sentences_all.csv", "candidate_sentences_high_confidence.csv", "excluded_multilabel.csv", "unmatched_annotations.csv", "ambiguous_annotations.csv"]:
        old_path = os.path.join(output_dir, old_file)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    # Calculate verbatim class distribution
    label_counts = {}
    for r in all_single_candidates:
        lbl = r["verbatim_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    evaluated = total_sentences - total_fragments
    reconciled = total_aligned_annotations + total_unmatched + total_ambiguous

    print("\n" + "="*60)
    print(" C3PA SENTENCE PARSER DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Documents processed:                    {total_docs}")
    print(f"HTML sentences extracted:               {total_sentences:,}")
    print(f"Non-sentence fragments excluded:        {total_fragments:,}")
    print(f"Sentences evaluated for alignment:      {evaluated:,}")
    print(f"Source annotation rows:                 {total_source_rows:,}")
    print(f"Empty annotation rows skipped:          {total_skipped_empty:,}")
    print(f"Total C3PA annotations considered:      {total_annotations:,}")
    print(f"Annotations aligned (distinct):         {total_aligned_annotations:,}")
    print(f"Annotation-sentence links:              {total_aligned_links:,}")
    print(f"Annotations unmatched:                  {total_unmatched:,}")
    print(f"Annotations ambiguous:                  {total_ambiguous:,}")
    if reconciled != total_annotations:
        print(f"WARNING: annotation accounting mismatch: {reconciled:,} != {total_annotations:,}")
    print(f"Sentences with 0 C3PA labels:           {sentences_no_labels:,}")
    print(f"Sentences with exactly 1 verbatim label:{len(all_single_candidates):,}")
    print(f"Sentences with >1 verbatim labels:      {len(multi_label_candidates):,}")
    print(f"High-confidence single-label candidates:{len(high_conf_candidates):,}")
    print("-" * 60)
    print(f"Segmenter: {SEGMENTER_NAME}")
    print(
        "Runtime versions: Python %s | spaCy %s | BeautifulSoup %s"
        % (
            platform.python_version(),
            getattr(spacy, "__version__", "n/a") if NLP is not None else "n/a",
            bs4.__version__,
        )
    )
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    print("Verbatim C3PA Class Distribution (Single-Label Candidates):")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:<55} {cnt:>6,}")
    print("="*60 + "\n")

    return {
        "documents": total_docs,
        "sentences_extracted": total_sentences,
        "fragments_excluded": total_fragments,
        "sentences_evaluated": evaluated,
        "source_annotation_rows": total_source_rows,
        "empty_annotation_rows_skipped": total_skipped_empty,
        "annotations_considered": total_annotations,
        "annotations_aligned_distinct": total_aligned_annotations,
        "annotation_sentence_links": total_aligned_links,
        "annotations_unmatched": total_unmatched,
        "annotations_ambiguous": total_ambiguous,
        "sentences_unannotated": len(unannotated_candidates),
        "single_label_all": len(all_single_candidates),
        "single_label_high_confidence": len(high_conf_candidates),
        "multi_label": len(multi_label_candidates),
        "label_counts": label_counts,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse C3PA dataset into sentence candidates.")
    parser.add_argument("--dataset-dir", default="C3PA_Dataset", help="Path to C3PA dataset directory")
    parser.add_argument("--output-dir", default="output", help="Directory to output candidate CSVs")
    parser.add_argument("--sanity-check", help="Sanity check a single document, e.g. DB/2. Writes to output/sanity/DB_2 by default.")
    parser.add_argument("--export-master", action="store_true", help="Export all_parsed_sentences.csv (all candidate tiers plus excluded fragments)")
    args = parser.parse_args()

    run_pipeline(args.dataset_dir, args.output_dir, single_doc=args.sanity_check, export_master=args.export_master)
