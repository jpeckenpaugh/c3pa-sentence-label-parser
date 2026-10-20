# C3PA Sentence-Level Candidate Parser (`c3pa-sentence-label-parser`)

A Python parser for processing the [C3PA Dataset](https://github.com/MaazBinMusa/C3PA_Dataset/) (California Consumer Privacy Act annotations on privacy policies) into structured **sentence-level dataset candidates**.

This parser is built for ML/AI training and evaluation scenarios that specifically require **full sentences** (or near-complete sentence-like text) as candidate inputs (such as sentence-level classification, policy sentence auditing, or sentence-based retrieval). It extracts sentence boundaries from the original policy HTML documents, filters out headings, table cells, UI controls, and other non-sentence blocks, and aligns C3PA annotation evidence back to the remaining full sentences. Original C3PA labels are preserved **verbatim**.

---

## Table of Contents

- [Background & Dataset Structure](#background--dataset-structure)
- [The Core Parsing Challenge](#the-core-parsing-challenge)
- [Parsing & Alignment Methodology](#parsing--alignment-methodology)
- [Supported Downstream Use Cases](#supported-downstream-use-cases)
- [Diagnostic Statistics Across Full Corpus](#diagnostic-statistics-across-full-corpus)
- [Quick Start & Usage](#quick-start--usage)
- [Output File Reference](#output-file-reference)
- [Sentence Quality Filtering](#sentence-quality-filtering)
- [Label Scope](#label-scope)
- [Known Limitations](#known-limitations)
- [Reproducibility](#reproducibility)
- [Testing](#testing)

---

## Background & Dataset Structure

### The C3PA Dataset

The C3PA dataset contains human-annotated privacy policies designed to benchmark privacy disclosures under California privacy laws (CCPA / CPRA).

- **Corpus Composition:** 400 total privacy policy documents split into two subsets:
  - `DB` (230 documents): `Htmls/DB/*.html` and `Annotations/DB/*.csv`
  - `WS` (170 documents): `Htmls/WS/*.html` and `Annotations/WS/*.csv`
- **Annotations:** 46,483 annotation rows created by 6 independent human annotators (`ra1` through `ra6`). 45,121 rows have usable text and a label; the remaining 1,362 have empty `Text` fields and are reported as skipped.
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
1. Extract readable text blocks (preserving block element kind)
        ↓
2. Remove non-sentence blocks: headings, table cells, UI controls
        ↓
3. Segment remaining text into full sentences (spaCy)
        ↓
4. Align C3PA annotation spans to reconstructed sentences (3-Tier Match)
        ↓
5. Aggregate annotation evidence per sentence across all annotators
        ↓
6. Categorize sentences by label evidence (Single-Label, Multi-Label, Unannotated)
```

### Three-Tier Alignment Algorithm

Only sentence-like blocks participate in alignment. Headings, table cells, UI controls, and over-short or machine-generated lines are written to `fragments_excluded.csv` and never receive labels.

For each annotation entry and each eligible sentence in the same document, alignment is established using three complementary strategies:

1. **Case A — Sub-Sentence Fragment (`annotation_in_sentence`):**
   - The annotation span is fully contained inside a single grammatical sentence (`annotation_normalized in sentence_normalized`).
   - Resolves short text fragments back to their complete containing source sentence.
   - *Ambiguity Check:* If a short text span appears verbatim in multiple sentences of a document, it is logged to `annotations_ambiguous.csv` and excluded from sentence evidence to maintain precision.

2. **Case B — Multi-Sentence Paragraph Span (`sentence_in_annotation_paragraph`):**
   - The annotation span covers a multi-sentence paragraph (`sentence_normalized in annotation_normalized`).
   - Every eligible full sentence (at least 5 words) contained within the paragraph span inherits the annotation label.

3. **Case C — Partial Overlap (`partial_overlap`):**
   - Cross-sentence boundary selections where human annotators started or ended mid-sentence.
   - Matched via token containment overlap ($\ge 55\%$ of annotation tokens in the sentence). This is containment, not Jaccard.
   - If two or more sentences tie for the best overlap, the annotation is logged as ambiguous rather than assigned arbitrarily.

---

## Supported Downstream Use Cases

The parser categorizes extracted sentences into distinct CSV deliverables tailored for different ML/AI research tasks:

```
                                 Eligible Sentence Blocks
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               ▼                            ▼                            ▼
   Single-Label Candidates      Multi-Label Sentences        Unannotated Sentences
  (len(verbatim_labels) == 1)   (len(verbatim_labels) > 1)   (len(verbatim_labels) == 0)
               │                            │                            │
      ┌────────┴────────┐                   │                            │
      ▼                 ▼                   ▼                            ▼
 All Candidates   High-Confidence   Multi-Label Dataset         Negative Sampling /
   (34,642 rows)     (18,870 rows)      (24,408 rows)           Unlabeled Background Text
                                                                  (49,752 rows)
```

### 1. Single-Label Sentence Classification
- **`single_label_sentences_all.csv` (34,642 rows):** Sentences receiving evidence for exactly 1 verbatim C3PA label.
- **`single_label_sentences_high_confidence.csv` (18,870 rows):** Single-label candidate sentences supported by $\ge 2$ distinct human annotators with 0 label conflicts.

### 2. Multi-Label Sentence Classification & Policy Nuance Analysis
- **`multi_label_sentences.csv` (24,408 rows):** Sentences containing evidence for $>1$ distinct verbatim C3PA label (e.g., multi-rights disclosure sentences covering deletion, correction, and opt-out rights).
- *Data Integrity Guarantee:* Genuinely multi-label sentences are routed here rather than being duplicated into contradictory single-label rows.

### 3. Negative Example Harvesting & Background Text Detection
- **Unannotated Policy Text (49,752 sentences):** Eligible HTML sentences receiving 0 annotations, suitable for negative sampling or background policy text detection. Non-sentence fragments and duplicate boilerplate should still be reviewed before use.

---

## Diagnostic Statistics Across Full Corpus

Summary of running `parse_c3pa_sentences.py` on the complete 400-document dataset:

```text
============================================================
 C3PA SENTENCE PARSER DIAGNOSTIC SUMMARY
============================================================
Documents processed:                    400
HTML sentences extracted:               177,502
Non-sentence fragments excluded:        68,700
Sentences evaluated for alignment:      108,802
Source annotation rows:                 46,483
Empty annotation rows skipped:          1,362
Total C3PA annotations considered:      45,121
Annotations aligned (distinct):         39,678
Annotation-sentence links:              215,463
Annotations unmatched:                  3,443
Annotations ambiguous:                  2,000
Sentences with 0 C3PA labels:           49,752
Sentences with exactly 1 verbatim label:34,642
Sentences with >1 verbatim labels:      24,408
High-confidence single-label candidates:18,870
------------------------------------------------------------
Segmenter: spacy-sentencizer
Runtime versions: Python 3.13.5 | spaCy 3.8.16 | BeautifulSoup 4.15.0
------------------------------------------------------------
Verbatim C3PA Class Distribution (Single-Label Candidates):
  Categories of Personal Information Collected            16,269
  Categories of Personal Information Shared / Disclosed    5,288
  Methods to exercise rights                               3,298
  Others                                                   3,262
  Updated Privacy Policy                                   1,679
  Description of Right to Delete                           1,196
  Description of Right to Opt-out of sale of PI            1,072
  Description of Right to Non-discrimination on exercising rights    816
  Description of Right to Know PI Collected                  475
  Categories of Personal Information Sold                    447
  Description of Right to Correct Information                362
  Description of Right to Limit use of PI                    279
  Description of Right to Know PI sold / shared              199
============================================================
```

`Annotations aligned (distinct)` counts annotations that produced at least one sentence link. `Annotation-sentence links` counts the links themselves, so a paragraph-level annotation that supports several sentences contributes multiple links. The distinct-aligned, unmatched, and ambiguous counts always sum to the 45,121 annotations considered.

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

Sanity-check output is written to `output/sanity/DB_2/` so the full-corpus CSVs in `output/` are never overwritten.

### 4. Tests

Run the unit and CLI tests with the standard library test runner:

```bash
python3 -m unittest discover -s tests -v
```

---

## Output File Reference

The parser writes generated CSV deliverables to the `output/` directory (all standard deliverables are under 15 MB for lightweight git tracking):

| Output File | Rows | File Size | Description |
| :--- | :--- | :--- | :--- |
| **`single_label_sentences_all.csv`** | 34,642 | ~12.8 MB | All sentences receiving **exactly 1** verbatim C3PA label. |
| **`single_label_sentences_high_confidence.csv`** | 18,870 | ~7.1 MB | Single-label sentences supported by **$\ge 2$ distinct annotators** with 0 conflicts. |
| **`multi_label_sentences.csv`** | 24,408 | ~10.3 MB | Sentences receiving **$>1$ verbatim C3PA labels** across annotators. |
| **`unannotated_sentences.csv`** | 49,752 | ~12.0 MB | Eligible sentences receiving **0 annotations** (candidate negative pool). |
| **`fragments_excluded.csv`** | 68,700 | ~10.6 MB | Headings, table cells, UI text, and other non-sentence blocks removed before alignment. |
| **`annotations_unmatched.csv`** | 3,443 | ~2.8 MB | Annotation spans that could not be aligned to an eligible sentence. |
| **`annotations_ambiguous.csv`** | 2,000 | ~0.7 MB | Annotation spans matching more than one sentence (or tied in partial-overlap matching). |

> **Optional Master Export:** To generate `all_parsed_sentences.csv` (all 177,502 extracted blocks, including excluded fragments), pass `--export-master` to `parse_c3pa_sentences.py`.

### Sentence CSV Schema

All sentence CSV deliverables (`single_label_sentences_all.csv`, `single_label_sentences_high_confidence.csv`, `multi_label_sentences.csv`, `unannotated_sentences.csv`, `fragments_excluded.csv`) share a consistent schema:

- `doc_id`: Document identifier (e.g., `DB_2`, `WS_17`)
- `sentence_id`: Extracted sentence identifier (e.g., `DB_2_S10`)
- `sentence_text`: Sentence-like text extracted from HTML
- `sentence_category`: Category flag (`single_label`, `multi_label`, `unannotated`, or `fragment`)
- `sentence_kind`: Source block kind (`prose`, `list`, `heading`, `cell`, or `ui`)
- `verbatim_label`: Primary verbatim C3PA label (`MULTI_LABEL` if multiple labels present, empty if unannotated or excluded)
- `verbatim_labels`: Semicolon-separated list of all aligned verbatim C3PA labels
- `annotator_count`: Number of distinct human annotators supporting this sentence
- `annotators`: Semicolon-separated list of supporting annotator IDs (`ra1`, `ra2`, etc.)
- `source_annotation_count`: Total count of aligned annotation entries
- `alignment_type`: Semicolon-separated alignment strategies used (`annotation_in_sentence`, `sentence_in_annotation_paragraph`, `partial_overlap`)
- `exclusion_reason`: Why a fragment was excluded (empty for eligible sentences)
- `source_html`: Dataset-relative path to the original HTML document
- `source_annotation_file`: Dataset-relative path to the original CSV annotation file

The `fragments_excluded.csv` file uses the same columns with empty label and annotation fields, so it can be concatenated with the candidate files for auditing. Sentence IDs are assigned to every extracted block; excluded fragments occupy the gaps in the candidate files, and the union of all files is complete.

---

## Sentence Quality Filtering

Sentence candidates must be sentence-like text, not every visible string in a policy. The parser walks the DOM and tags each text block with its source kind (`prose`, `list`, `heading`, `cell`, `ui`), then applies these rules before alignment:

1. **Non-sentence blocks are excluded.** Headings (`h1`–`h6`, `th`), table cells (`td`), and UI controls (`button`, `select`, `option`, `summary`, `label`) never receive candidate labels. Raw tag/script blocks, embedded JSON, encoded ID blobs, and text longer than 5,000 characters are also excluded.
2. **Text must be substantive.** A block needs at least 3 words and at least one alphabetic character.
3. **A sentence needs evidence.** Text ending in `.`, `!`, or `?` passes. Text without terminal punctuation must have at least 5 words and must not look like an all-caps or title-case heading.
4. **Paragraph propagation needs a stronger sentence.** Case B only assigns labels to sentences with at least 5 words, so a short generic phrase cannot inherit a whole paragraph's labels.

Excluded blocks are written to `fragments_excluded.csv` with an `exclusion_reason` such as `non_sentence_block:heading`, `too_few_words`, `no_terminal_punctuation`, `title_like_heading`, `all_caps_heading`, or `machine_text`. The thresholds are constants near the top of `parse_c3pa_sentences.py`.

This filtering materially changes dataset size relative to older releases: the previous pipeline emitted 44,987 single-label rows; the filtered pipeline emits 34,642. The reduction is concentrated in headings, table rows, UI labels, and short list fragments that were never full sentences.

## Label Scope

C3PA labels are preserved **verbatim** (13 labels). The original `concept.md` instruction document specified mapping those 13 labels to six CPRA labels (`Notice_Requirement`, `Right_to_Correct`, `Right_to_Delete`, `Right_to_Know`, `Right_to_Limit_Sensitive`, `Right_to_Opt_Out`). That mapping is **not applied** by this pipeline and is intentionally out of scope for the sentence-extraction step.

The distinction matters for downstream use. Under the retired mapping:

- 7,376 of the 34,642 single-label rows (21.3%) carry labels the six-label spec said to ignore (`Methods to exercise rights`, `Others`, `Non-discrimination`).
- 21,925 of the 24,408 multi-label rows (89.8%) would collapse to exactly one CPRA target because four C3PA labels all map to `Notice_Requirement`; only 2,210 would remain genuinely multi-label, and 273 would map to no target.

If the six-label task is still required, add the mapping after this parser rather than changing the extraction/alignment stage.

## Known Limitations

- **Duplicate rows remain.** Identical sentence text can appear multiple times within one document because the source HTML repeats content (for example, responsive/mobile copies and repeated sections). The parser does not deduplicate. In the current outputs there are 2,926 extra single-label rows, 3,640 extra multi-label rows, and 4,385 extra unannotated rows from within-document duplicates. Deduplicate or split by document before training.
- **Some list fragments survive.** A bullet with at least 5 words and no terminal punctuation can pass the sentence gate (for example, an imperative bullet such as `Provide our Services, including Services we provide to clients`). Treat `sentence_kind=list` as a review signal.
- **Precision-first alignment loses annotation recall.** 3,443 annotations are unmatched and 2,000 are ambiguous (about 12% of the 45,121 considered). Most new unmatched annotations previously matched headings, table cells, or short fragments that are now excluded. This is intentional, but it does reduce label coverage.
- **Sentences can be split across block elements.** When a policy puts part of a sentence in one block and the rest in the next, the extractor can emit a fragment that starts mid-sentence. These usually remain in `sentence_kind=prose` and may need review.
- **Residual UI/boilerplate can reach the unannotated pool.** Semantic `header`/`footer`/`nav` elements are removed, but div-based navigation and cookie controls sometimes remain.
- **The dataset and dependencies are unpinned.** `install.sh` clones the dataset's default branch, and `requirements.txt` uses lower bounds. Regenerating later can change sentence boundaries and counts.
- **Raw annotation evidence is not exported.** Each sentence reports `source_annotation_count` and `alignment_type`, but not the original annotation texts. Use the annotation CSVs plus the alignment rules for a detailed audit.
- **No standalone `LICENSE` file is included** even though the README describes intended reuse terms. Add a license file before redistribution.

## Reproducibility

The committed CSVs were generated from `MaazBinMusa/C3PA_Dataset@d482b94` with:

```text
Python 3.13.5
spaCy 3.8.16 (blank English pipeline + sentencizer)
BeautifulSoup 4.15.0
```

Regenerate with:

```bash
python3 parse_c3pa_sentences.py --dataset-dir /path/to/C3PA_Dataset --output-dir output
```

`source_html` and `source_annotation_file` are always written as dataset-relative paths (`C3PA_Dataset/Htmls/...`), so running with an absolute or relative `--dataset-dir` produces the same CSV contents. If spaCy is unavailable, the parser prints a warning and uses a regex sentence splitter; that output is not comparable to a spaCy run. The diagnostic summary prints the active segmenter and library versions.

## Testing

The repository includes nine tests covering DOM block extraction, sentence-quality rules, machine-text filtering, fragment-to-sentence alignment, multi-label exclusion, diagnostic reconciliation, dataset-relative paths, and the guarantee that `--sanity-check` does not overwrite the full-corpus `output/` directory:

```bash
python3 -m unittest discover -s tests -v
```

---

## Credits & Tooling

The sentence parser relies on several open-source libraries and datasets:

- **[C3PA Dataset](https://github.com/MaazBinMusa/C3PA_Dataset)** — Expert-annotated privacy policy corpus created by Maaz Bin Musa et al. (EMNLP 2024).
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** — HTML parsing, DOM cleanup, and structural block text extraction.
- **[spaCy](https://spacy.io/)** — Fast, robust sentence boundary tokenization (`spacy.blank("en")` sentencizer).
- **Python Standard Library** — `unicodedata` (Unicode NFKC character normalization), `re` (regular expressions), `csv`, `argparse`, and `unittest`.

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
