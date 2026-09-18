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


def extract_readable_text_from_html(html_content: str) -> str:
    """Extracts readable text from HTML preserving element structural breaks."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "head", "header", "footer", "nav", "svg"]):
        tag.decompose()
        
    # Append space/newline to block tags to prevent sentence concatenation
    block_tags = [
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "td", "article", "section", "blockquote", "br"
    ]
    for tag_name in block_tags:
        for element in soup.find_all(tag_name):
            element.append(NavigableString("\n"))
            
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


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


def load_annotations(csv_path: str) -> list[dict]:
    """Loads and normalizes C3PA annotations from CSV."""
    annotations = []
    if not os.path.exists(csv_path):
        return annotations
        
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            ranumb = row.get("RANumb", "").strip()
            text_orig = row.get("Text", "")
            label_orig = row.get("Label", "").strip()
            
            if not text_orig or not label_orig:
                continue
                
            text_norm = normalize_text(text_orig)
            if not text_norm:
                continue
                
            annotations.append({
                "annotation_id": row_idx,
                "ranumb": ranumb,
                "annotation_original": text_orig,
                "annotation_normalized": text_norm,
                "c3pa_label": label_orig,
                "source_csv": csv_path
            })
            
    return annotations


def process_document(doc_id: str, html_path: str, csv_path: str):
    """Processes a single document (HTML + CSV annotations)."""
    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()
        
    readable_text = extract_readable_text_from_html(html_content)
    raw_sentences = segment_sentences(readable_text)
    
    sentences = []
    for idx, s_orig in enumerate(raw_sentences, start=1):
        s_norm = normalize_text(s_orig)
        if s_norm:
            sentences.append({
                "doc_id": doc_id,
                "sentence_id": f"{doc_id}_S{idx}",
                "sentence_original": s_orig,
                "sentence_normalized": s_norm,
                "source_html": html_path
            })
            
    annotations = load_annotations(csv_path)
    
    aligned_matches = []
    unmatched = []
    ambiguous = []
    
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
            
        # 2. Case B: Multi-sentence paragraph span containing 1 or more full sentences
        case_b_matches = [
            s for s in sentences
            if s["sentence_normalized"] in ann_norm and len(s["sentence_normalized"]) > 10
        ]
        if case_b_matches:
            for sent in case_b_matches:
                aligned_matches.append((ann, sent, "sentence_in_annotation_paragraph"))
            continue
            
        # 3. Case C: Partial overlap matching for cross-sentence boundary selections
        if len(ann_tokens) >= 3:
            best_sent = None
            best_overlap = 0.0
            for sent in sentences:
                sent_tokens = set(sent["sentence_normalized"].split())
                intersection = ann_tokens.intersection(sent_tokens)
                if intersection:
                    overlap = len(intersection) / len(ann_tokens)
                    if overlap > best_overlap and overlap >= 0.55:
                        best_overlap = overlap
                        best_sent = sent
            if best_sent:
                aligned_matches.append((ann, best_sent, "partial_overlap"))
                continue
                
        # If no alignment could be established
        unmatched.append(ann)
            
    # Aggregate sentence evidence
    evidence = {
        sent["sentence_id"]: {
            "doc_id": doc_id,
            "sentence_id": sent["sentence_id"],
            "sentence_text": sent["sentence_original"],
            "source_html": html_path,
            "source_annotation_file": csv_path,
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
        "annotations": annotations,
        "aligned_matches": aligned_matches,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "evidence": list(evidence.values())
    }


def run_pipeline(dataset_dir: str, output_dir: str, single_doc: str = None):
    """Runs pipeline over the dataset documents."""
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
                    
    print(f"Processing {len(doc_pairs)} documents...")
    
    total_docs = len(doc_pairs)
    total_sentences = 0
    total_annotations = 0
    total_aligned = 0
    total_unmatched = 0
    total_ambiguous = 0
    
    all_single_candidates = []
    high_conf_candidates = []
    excluded_multilabel = []
    all_unmatched = []
    all_ambiguous = []
    
    sentences_no_labels = 0
    
    for doc_id, html_path, csv_path in doc_pairs:
        res = process_document(doc_id, html_path, csv_path)
        
        total_sentences += len(res["sentences"])
        total_annotations += len(res["annotations"])
        total_aligned += len(res["aligned_matches"])
        total_unmatched += len(res["unmatched"])
        total_ambiguous += len(res["ambiguous"])
        
        all_unmatched.extend([{**ann, "doc_id": doc_id} for ann in res["unmatched"]])
        all_ambiguous.extend([{**ann, "doc_id": doc_id, "matched_sentences": ";".join(sids)} for ann, sids in res["ambiguous"]])
        
        for ev in res["evidence"]:
            v_labels = sorted(list(ev["verbatim_labels"]))
            annotators = sorted(list(ev["annotators"]))
            align_types = sorted(list(ev["alignment_types"]))
            
            row = {
                "doc_id": ev["doc_id"],
                "sentence_id": ev["sentence_id"],
                "sentence_text": ev["sentence_text"],
                "verbatim_label": v_labels[0] if len(v_labels) == 1 else "MULTI_LABEL" if len(v_labels) > 1 else "",
                "verbatim_labels": ";".join(v_labels),
                "annotator_count": len(annotators),
                "annotators": ";".join(annotators),
                "source_annotation_count": len(ev["source_annotations"]),
                "alignment_type": ";".join(align_types),
                "source_html": ev["source_html"],
                "source_annotation_file": ev["source_annotation_file"]
            }
            
            if len(v_labels) == 0:
                sentences_no_labels += 1
            elif len(v_labels) == 1:
                all_single_candidates.append(row)
                if len(annotators) >= 2:
                    high_conf_candidates.append(row)
            else:
                excluded_multilabel.append(row)
                
    # Write output CSVs
    fieldnames = [
        "doc_id", "sentence_id", "sentence_text", "verbatim_label",
        "verbatim_labels", "annotator_count", "annotators",
        "source_annotation_count", "alignment_type", "source_html", "source_annotation_file"
    ]
    
    def write_csv(filename, rows, fields):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    write_csv("candidate_sentences_all.csv", all_single_candidates, fieldnames)
    write_csv("candidate_sentences_high_confidence.csv", high_conf_candidates, fieldnames)
    write_csv("excluded_multilabel.csv", excluded_multilabel, fieldnames)

    unmatched_fields = ["doc_id", "annotation_id", "ranumb", "annotation_original", "c3pa_label", "source_csv"]
    write_csv("unmatched_annotations.csv", all_unmatched, unmatched_fields)
    
    ambiguous_fields = ["doc_id", "annotation_id", "ranumb", "annotation_original", "c3pa_label", "matched_sentences", "source_csv"]
    write_csv("ambiguous_annotations.csv", all_ambiguous, ambiguous_fields)
    
    # Calculate verbatim class distribution
    label_counts = {}
    for r in all_single_candidates:
        lbl = r["verbatim_label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
    print("\n" + "="*60)
    print(" C3PA SENTENCE PARSER DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"Documents processed:                    {total_docs}")
    print(f"HTML sentences extracted:               {total_sentences:,}")
    print(f"Total C3PA annotations:                 {total_annotations:,}")
    print(f"Annotations aligned successfully:       {total_aligned:,}")
    print(f"Annotations unmatched:                  {total_unmatched:,}")
    print(f"Annotations ambiguous:                  {total_ambiguous:,}")
    print(f"Sentences with 0 C3PA labels:          {sentences_no_labels:,}")
    print(f"Sentences with exactly 1 verbatim label:{len(all_single_candidates):,}")
    print(f"Sentences with >1 verbatim labels:     {len(excluded_multilabel):,}")
    print(f"High-confidence single-label candidates:{len(high_conf_candidates):,}")
    print("-" * 60)
    print("Verbatim C3PA Class Distribution (Single-Label Candidates):")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:<55} {cnt:>6,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse C3PA dataset into sentence candidates.")
    parser.add_argument("--dataset-dir", default="C3PA_Dataset", help="Path to C3PA dataset directory")
    parser.add_argument("--output-dir", default="output", help="Directory to output candidate CSVs")
    parser.add_argument("--sanity-check", help="Sanity check a single document, e.g. DB/2")
    args = parser.parse_args()
    
    run_pipeline(args.dataset_dir, args.output_dir, single_doc=args.sanity_check)
