# C3PA Sentence-Level Candidate Parser (`c3pa-sentence-label-parser`)

A Python parser for processing the [C3PA Dataset](https://github.com/MaazBinMusa/C3PA_Dataset/) (California Consumer Privacy Act annotations on privacy policies) into structured **sentence-level dataset candidates**.

This parser is built for ML/AI training and evaluation scenarios that specifically require **full grammatical sentences** as candidate inputs (such as sentence-level classification, policy sentence auditing, or sentence-based retrieval). It extracts sentence boundaries from the original policy HTML documents and aligns C3PA annotation evidence back to those full sentences, preserving all original C3PA labels **verbatim**.

---

## Table of Contents

- [Background & Dataset Structure](#background--dataset-structure)
- [The Core Parsing Challenge](#the-core-parsing-challenge)
- [Parsing & Alignment Methodology](#parsing--alignment-methodology)
- [Supported Downstream Use Cases](#supported-downstream-use-cases)
- [Diagnostic Statistics Across Full Corpus](#diagnostic-statistics-across-full-corpus)
- [Quick Start & Usage](#quick-start--usage)
- [Output File Reference](#output-file-reference)

---

## Background & Dataset Structure

### The C3PA Dataset

The C3PA dataset contains human-annotated privacy policies designed to benchmark privacy disclosures under California privacy laws (CCPA / CPRA).

- **Corpus Composition:** 400 total privacy policy documents split into two subsets:
  - `DB` (230 documents): `Htmls/DB/*.html` and `Annotations/DB/*.csv`
  - `WS` (170 documents): `Htmls/WS/*.html` and `Annotations/WS/*.csv`
- **Annotations:** 45,121 total annotation entries created by 6 independent human annotators (`ra1` through `ra6`).
- **Annotation Fields:**
  - `RANumb`: Annotator identifier (e.g., `ra1`, `ra2`, `ra3`)
  - `Text`: The raw text span selected by the annotator
  - `Label`: The raw C3PA label assigned to the text selection

---

## The Core Parsing Challenge

In the raw C3PA dataset, the `Text` field represents **free-form text selections made by annotators**. These selections vary significantly in structure:

1. **Sub-Sentence Fragments:** Selections shorter than a sentence (e.g., `"the right to request deletion"` or `"or correction of your data"`).
2. **Multi-Sentence Paragraph Spans:** Selections spanning an entire multi-sentence paragraph (e.g., a 5-sentence disclosure section).
3. **Partial Cross-Sentence Selections:** Selections starting mid-sentence in line 1 and ending mid-sentence in line 2.
4. **Annotator Overlaps:** Multiple annotators selecting overlapping or differently bounded text spans for the same underlying policy statements.

If raw annotation text spans are used directly in sentence-oriented ML tasks, sub-sentence fragments produce syntactically incomplete training inputs, while multi-sentence spans produce oversized multi-concept blocks. 

To address this, sentence boundaries must be derived from the **original HTML documents**, and annotation evidence must be aligned back to those reconstructed sentences.

---

## Parsing & Alignment Methodology

The parser (`parse_c3pa_sentences.py`) processes each document pair independently:

```text
Original Source HTML (Htmls/DB/, Htmls/WS/)
        ↓
1. Extract readable policy text (preserving block structural breaks)
        ↓
2. Segment text into full grammatical sentences (spaCy)
        ↓
3. Align C3PA annotation spans to reconstructed sentences (3-Tier Match)
        ↓
4. Aggregate annotation evidence per sentence across all annotators
        ↓
5. Categorize sentences by label evidence (Single-Label, Multi-Label, Unannotated)
```

### Three-Tier Alignment Algorithm

For each annotation entry and each sentence in the same document, alignment is established using three complementary strategies:

1. **Case A — Sub-Sentence Fragment (`annotation_in_sentence`):**
   - The annotation span is fully contained inside a single grammatical sentence (`annotation_normalized in sentence_normalized`).
   - Resolves short text fragments back to their complete containing source sentence.
   - *Ambiguity Check:* If a short text span appears verbatim in multiple distant sections of a document, it is logged to `ambiguous_annotations.csv` and excluded from sentence evidence to maintain precision.

2. **Case B — Multi-Sentence Paragraph Span (`sentence_in_annotation_paragraph`):**
   - The annotation span covers a multi-sentence paragraph (`sentence_normalized in annotation_normalized`).
   - Every full sentence contained within the paragraph span inherits the annotation label.

3. **Case C — Partial Overlap (`partial_overlap`):**
   - Cross-sentence boundary selections where human annotators started or ended mid-sentence.
   - Matched via token Jaccard overlap ($\ge 55\%$) between annotation tokens and sentence tokens.

---

## Supported Downstream Use Cases

The parser categorizes extracted sentences into distinct CSV deliverables tailored for different ML/AI research tasks:

```
                                 Extracted HTML Sentences
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
   Single-Label Candidates      Multi-Label Sentences        Unannotated Sentences
  (len(verbatim_labels) == 1)   (len(verbatim_labels) > 1)   (len(verbatim_labels) == 0)
               │                            │                            │
      ┌────────┴────────┐                   │                            │
      ▼                 ▼                   ▼                            ▼
 All Candidates   High-Confidence   Multi-Label Dataset         Negative Sampling /
   (3,395 rows)     (908 rows)         (983 rows)           Unlabeled Background Text
```

### 1. Single-Label Sentence Classification
- **`candidate_sentences_all.csv` (44,987 rows):** Sentences receiving evidence for exactly 1 verbatim C3PA label.
- **`candidate_sentences_high_confidence.csv` (24,621 rows):** Single-label candidate sentences supported by $\ge 2$ distinct human annotators with 0 label conflicts.

### 2. Multi-Label Sentence Classification & Policy Nuance Analysis
- **`excluded_multilabel.csv` (41,556 rows):** Sentences containing evidence for $>1$ distinct verbatim C3PA label (e.g., multi-rights disclosure sentences covering deletion, correction, and opt-out rights).
- *Data Integrity Guarantee:* Genuinely multi-label sentences are routed here rather than being duplicated into contradictory single-label rows.

### 3. Negative Example Harvesting & Background Text Detection
- **Unannotated Policy Text (84,405 sentences):** Extracted HTML sentences receiving 0 annotations, suitable for negative sampling or background policy text detection.

---

## Diagnostic Statistics Across Full Corpus

Summary of running `parse_c3pa_sentences.py` on the complete 400-document dataset:

```text
============================================================
 C3PA SENTENCE PARSER DIAGNOSTIC SUMMARY
============================================================
Documents processed:                    400
HTML sentences extracted:               170,948
Total C3PA annotations:                 45,121
Annotations aligned successfully:       454,468
Annotations unmatched:                  626
Annotations ambiguous:                  2,214
Sentences with 0 C3PA labels:          84,405
Sentences with exactly 1 verbatim label:44,987
Sentences with >1 verbatim labels:     41,556
High-confidence single-label candidates:24,621
------------------------------------------------------------
Verbatim C3PA Class Distribution (Single-Label Candidates):
  Categories of Personal Information Collected            21,016
  Categories of Personal Information Shared / Disclosed    6,804
  Methods to exercise rights                               4,569
  Others                                                   4,138
  Updated Privacy Policy                                   2,485
  Description of Right to Delete                           1,415
  Description of Right to Opt-out of sale of PI            1,329
  Description of Right to Non-discrimination on exercising rights  1,049
  Description of Right to Know PI Collected                  592
  Categories of Personal Information Sold                    561
  Description of Right to Correct Information                479
  Description of Right to Limit use of PI                    327
  Description of Right to Know PI sold / shared              223
============================================================
```

---

## Quick Start & Usage

### 1. Automated Installation & Setup

Run `install.sh` to set up the Python virtual environment (`.venv`), install required dependencies (`requirements.txt`), and automatically clone the [C3PA Dataset](https://github.com/MaazBinMusa/C3PA_Dataset.git) repository into `C3PA_Dataset/`:

```bash
./install.sh
```

### 2. Execution

Run the complete pipeline over all 400 documents using `parse.sh`:

```bash
./parse.sh
```

### 3. Single-Document Sanity Check

Run a pre-flight check on a specific document (e.g., `DB/2`):

```bash
./parse.sh --sanity-check DB/2
```

---

## Output File Reference

The parser writes generated CSV deliverables to the `output/` directory (all files are under 15 MB for lightweight git tracking):

| Output File | Rows | File Size | Description |
| :--- | :--- | :--- | :--- |
| **`single_label_sentences_all.csv`** | 44,987 | ~14 MB | All sentences receiving **exactly 1** verbatim C3PA label. |
| **`single_label_sentences_high_confidence.csv`** | 24,621 | ~8.1 MB | Single-label sentences supported by **$\ge 2$ distinct annotators** with 0 conflicts. |
| **`multi_label_sentences.csv`** | 41,556 | ~15 MB | Sentences receiving **$>1$ verbatim C3PA labels** across annotators. |
| **`unannotated_sentences.csv`** | 84,405 | ~15 MB | HTML policy sentences receiving **0 annotations** (ideal for negative sampling). |
| **`annotations_unmatched.csv`** | 626 | ~308 KB | Annotation spans that could not be aligned to HTML text. |
| **`annotations_ambiguous.csv`** | 2,214 | ~694 KB | Short sub-sentence text spans matching $>1$ sentence in a document. |

> **Optional Master Export:** To generate `all_parsed_sentences.csv` (~45 MB master file containing all 170,948 sentences), pass `--export-master` to `parse_c3pa_sentences.py`.

### Sentence CSV Schema

All sentence CSV deliverables (`single_label_sentences_all.csv`, `single_label_sentences_high_confidence.csv`, `multi_label_sentences.csv`, `unannotated_sentences.csv`) share a consistent schema:

- `doc_id`: Document identifier (e.g., `DB_2`, `WS_17`)
- `sentence_id`: Reconstructed sentence identifier (e.g., `DB_2_S10`)
- `sentence_text`: Exact grammatical sentence string extracted from HTML
- `sentence_category`: Category flag (`single_label`, `multi_label`, or `unannotated`)
- `verbatim_label`: Primary verbatim C3PA label (`MULTI_LABEL` if multiple labels present, empty if unannotated)
- `verbatim_labels`: Semicolon-separated list of all aligned verbatim C3PA labels
- `annotator_count`: Number of distinct human annotators supporting this sentence
- `annotators`: Semicolon-separated list of supporting annotator IDs (`ra1`, `ra2`, etc.)
- `source_annotation_count`: Total count of aligned annotation entries
- `alignment_type`: Semicolon-separated alignment strategies used (`annotation_in_sentence`, `sentence_in_annotation_paragraph`, `partial_overlap`)
- `source_html`: Relative path to original HTML document
- `source_annotation_file`: Relative path to original CSV annotation file

---

## Credits & Tooling

The sentence parser relies on several open-source libraries and datasets:

- **[C3PA Dataset](https://github.com/MaazBinMusa/C3PA_Dataset)** — Expert-annotated privacy policy corpus created by Maaz Bin Musa et al. (EMNLP 2024).
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** — HTML parsing, DOM cleanup, and structural block text extraction.
- **[spaCy](https://spacy.io/)** — Fast, robust sentence boundary tokenization (`spacy.blank("en")` sentencizer).
- **Python Standard Library** — `unicodedata` (Unicode NFKC character normalization), `re` (regular expressions), `csv`, `argparse`.

---

## Citation Requirements

If you use the C3PA dataset or data derived from this parser in your research, please cite the original C3PA EMNLP 2024 paper:

```bibtex
@inproceedings{c3pa,
  title={C3PA: An Open Dataset of Expert-Annotated and Regulation-Aware Privacy Policies to Enable Scalable Regulatory Compliance Audits},
  author={Musa, Maaz Bin and Winston, Steven M. and Allen, Garrison and Schiller, Jacob and Moore, Kevin and Quick, Sean and Melvin, Johnathan and Srinivasan, Padmini and Diamantis, Mihailis E. and Nithyanand, Rishab},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}
```

---

## License & Terms of Use

This repository (`c3pa-sentence-label-parser`) is open-source software.

- **Reuse Policy:** Free to use, modify, adapt, and distribute for academic, personal, or commercial research.
- **Attribution:** If you use or adapt this parser code, please include an attribution link back to this repository and cite the original C3PA dataset paper above.
- **Data Rights:** Original C3PA annotations and raw policy HTML files remain under the terms and copyrights specified by their respective authors and the [C3PA Dataset](https://github.com/MaazBinMusa/C3PA_Dataset) project.
