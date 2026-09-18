# Instruction Document: Build a Simple C3PA Sentence-Level Candidate Parser

## Objective

Build a simple Python parser that converts the C3PA dataset ( https://github.com/MaazBinMusa/C3PA_Dataset/ )  into **candidate sentence/label pairs** suitable for a single-label sentence-classification experiment.

The intended downstream task is:

> **Input:** one grammatical sentence
> **Output:** one CPRA label

Do **not** treat the C3PA annotation `Text` field as though it is already a sentence.

C3PA defines `Text` as the **text span selected by an annotator**. Those spans may be:

* shorter than a sentence,
* exactly one sentence,
* several sentences long,
* overlapping with spans selected by other annotators,
* differently bounded by different annotators.

For example, in C3PA `DB/2.csv`, annotators selected fragments such as:

* `the right to request deletion`
* `or correction of your data`
* `disclose, and sell`

while another annotator selected an entire multi-concept sentence covering the same source material.

The parser should therefore derive sentence boundaries from the **original HTML document**, not from the annotation spans.

---

# Source Data

Use the public C3PA repository:

`https://github.com/MaazBinMusa/C3PA_Dataset`

Relevant folders:

```text
Annotations/DB/
Annotations/WS/

Htmls/DB/
Htmls/WS/
```

Each annotation CSV has:

```text
RANumb
Text
Label
```

C3PA documents state that `Text` is the annotated text span, not necessarily a grammatical sentence.

Each HTML file corresponds to the same numbered annotation file.

Example:

```text
Annotations/DB/2.csv
Htmls/DB/2.html
```

refer to the same privacy policy.

---

# Core Design Principle

The pipeline should be:

```text
Original HTML
    ↓
Extract readable policy text
    ↓
Segment into grammatical sentences
    ↓
Align C3PA annotation spans to those sentences
    ↓
Collect labels associated with each sentence
    ↓
Map C3PA labels to Matthew's six CPRA labels
    ↓
Keep only sentences with exactly one mapped label
```

The resulting sentence—not the original annotation span—is the candidate ML example.

---

# Important Constraint

Do **not** manufacture single-label examples from genuinely multi-label sentences.

For example, if one original sentence contains evidence for:

```text
Right_to_Delete
Right_to_Correct
Right_to_Opt_Out
```

that sentence should be classified as:

```text
MULTI_LABEL
```

and excluded from the single-label candidate dataset.

Do not create three copies of the same sentence with three different labels.

---

# Target CPRA Label Mapping

Use the following mapping from C3PA labels to Matthew's six target labels.

## Notice_Requirement

Map these C3PA labels to:

```text
Notice_Requirement
```

Source labels:

```text
Updated Privacy Policy
Categories of Personal Information Collected
Categories of Personal Information Shared / Disclosed
Categories of Personal Information Sold
```

## Right_to_Correct

```text
Description of Right to Correct Information
```

→

```text
Right_to_Correct
```

## Right_to_Delete

```text
Description of Right to Delete
```

→

```text
Right_to_Delete
```

## Right_to_Know

Both:

```text
Description of Right to Know PI Collected
Description of Right to Know PI sold / shared
```

→

```text
Right_to_Know
```

## Right_to_Limit_Sensitive

```text
Description of Right to Limit use of PI
```

→

```text
Right_to_Limit_Sensitive
```

## Right_to_Opt_Out

```text
Description of Right to Opt-out of sale of PI
```

→

```text
Right_to_Opt_Out
```

Ignore C3PA labels that do not map to one of these six categories for purposes of this candidate dataset.

Do not convert ignored labels to negatives.

---

# Step 1 — Load One Document at a Time

Process each C3PA document independently.

Use identifiers such as:

```text
DB_2
WS_17
```

Do not mix text between documents.

For the first implementation, it is acceptable to test only:

```text
DB/2
```

before scaling to the entire corpus.

---

# Step 2 — Extract Readable Text from HTML

Use a straightforward HTML parser such as:

```python
BeautifulSoup
```

Remove obvious non-content elements:

```text
script
style
noscript
```

Preserve visible textual structure where practical:

```text
paragraphs
headings
list items
table cells
```

Normalize obvious HTML artifacts such as:

```text
&nbsp;
repeated whitespace
Unicode quotation variants where needed for matching
```

Do not aggressively rewrite the actual text.

Preserve the original sentence text separately from any normalized matching representation.

---

# Step 3 — Sentence Segmentation

Use a real sentence segmentation library.

Preferred simple implementation:

```python
spaCy
```

A lightweight English pipeline or `sentencizer` is sufficient for the first version.

Do not use the C3PA annotation span itself as the sentence boundary.

Do not split clauses merely because they contain different labels.

For example:

```text
The right to request deletion or correction of your data.
```

is one sentence.

If it maps to both Delete and Correct, it should later be excluded from the single-label candidate set.

---

# Step 4 — Maintain Two Versions of Each Sentence

For each extracted sentence keep:

```text
sentence_original
sentence_normalized
```

Example:

```python
sentence_original =
"The right to request deletion or correction of your data."

sentence_normalized =
"the right to request deletion or correction of your data."
```

Normalization may include:

* lowercase,
* collapse whitespace,
* replace non-breaking spaces,
* normalize Unicode quotation characters,
* trim leading/trailing whitespace.

Do not use normalized text as the final training text.

---

# Step 5 — Normalize Annotation Spans

For every C3PA annotation row, keep:

```text
annotator
annotation_original
annotation_normalized
c3pa_label
mapped_label
```

Example:

```text
annotator: ra1

annotation_original:
"or correction of your data"

mapped_label:
Right_to_Correct
```

---

# Step 6 — Align Annotation Spans to Full Sentences

The first version should favor **precision over recall**.

Use conservative matching.

For each annotation and each sentence in the same document, consider the annotation aligned when either:

### Case A — Annotation is contained in sentence

Example:

```text
Annotation:
or correction of your data

Sentence:
The additional privacy protections include the right to request deletion or correction of your data.
```

If:

```python
annotation_normalized in sentence_normalized
```

then the annotation supports that full sentence.

This solves the short-fragment problem.

---

### Case B — Sentence is contained in annotation

Example:

The C3PA annotator selected a paragraph containing three sentences.

If:

```python
sentence_normalized in annotation_normalized
```

then that annotation supports the sentence.

This solves the oversized-span problem.

---

# Step 7 — Do Not Use Aggressive Fuzzy Matching Initially

For version 1:

Do not automatically assign labels based on weak semantic or fuzzy similarity.

If an annotation cannot be aligned through high-confidence containment matching, record it as:

```text
UNMATCHED
```

If it appears to match more than one plausible location, record it as:

```text
AMBIGUOUS
```

Produce reports for these cases.

Do not silently force them into the dataset.

A later version can explore controlled fuzzy matching if necessary.

---

# Step 8 — Aggregate All Annotation Evidence Per Sentence

For each reconstructed full sentence, collect all mapped labels supported by annotations from all annotators.

Example:

```text
Sentence ID:
DB_2_S87

Sentence:
"The additional privacy protections ... deletion or correction ... opt out ..."

Annotation evidence:

ra1 → Right_to_Delete
ra1 → Right_to_Correct
ra1 → Right_to_Opt_Out
ra2 → Right_to_Delete
ra2 → Right_to_Correct
ra3 → Right_to_Delete
ra3 → Right_to_Correct
ra3 → Right_to_Opt_Out
ra3 → Right_to_Know
```

The resulting label set is:

```python
{
    "Right_to_Delete",
    "Right_to_Correct",
    "Right_to_Opt_Out",
    "Right_to_Know"
}
```

That sentence is therefore:

```text
MULTI_LABEL
```

and should not enter the final single-label candidate dataset.

---

# Step 9 — Select Candidate Single-Label Sentences

A sentence qualifies as a candidate when:

```python
len(mapped_label_set) == 1
```

Example:

```text
Sentence:
"You can ask us to delete your data."

Mapped labels:
{Right_to_Delete}
```

Candidate:

```text
YES
```

Example:

```text
Sentence:
"You may request deletion or correction of your personal information."

Mapped labels:
{
  Right_to_Delete,
  Right_to_Correct
}
```

Candidate:

```text
NO — MULTI_LABEL
```

---

# Step 10 — Preserve Annotator Support

Do not reduce the result immediately to only:

```text
sentence
label
```

Preserve annotation evidence.

For every candidate sentence record:

```text
doc_id
sentence_id
sentence_text
mapped_label
annotator_count
annotators
source_annotation_count
source_annotations
alignment_status
```

Example:

```text
doc_id:
DB_2

sentence_id:
DB_2_S105

sentence_text:
"You can ask us to delete your data."

mapped_label:
Right_to_Delete

annotator_count:
2

annotators:
["ra1", "ra3"]

source_annotation_count:
2

alignment_status:
exact_containment
```

---

# Step 11 — Produce Two Candidate Tiers

Do not make a methodological decision yet about required annotator agreement.

Instead produce:

## All single-label candidates

Every reconstructed sentence with exactly one mapped CPRA label.

## High-confidence candidates

Suggested initial definition:

```text
exactly one mapped label
AND
support from at least 2 distinct annotators
AND
no conflicting mapped labels
AND
no ambiguous alignment
```

These should be separate outputs.

This lets us determine how much data would be lost by requiring stronger agreement.

---

# Required Output Files

Produce at least:

```text
candidate_sentences_all.csv
candidate_sentences_high_confidence.csv
excluded_multilabel.csv
unmatched_annotations.csv
ambiguous_annotations.csv
```

---

# Suggested Candidate CSV Schema

```text
doc_id
sentence_id
sentence_text
mapped_label
annotator_count
annotators
source_annotation_count
alignment_type
source_html
source_annotation_file
```

---

# Required Diagnostic Summary

At the end of the run, print a concise report.

Example:

```text
Documents processed: 399

HTML sentences extracted: 42,xxx

Target C3PA annotations: 35,xxx

Annotations aligned successfully: 33,xxx
Annotations unmatched: xxx
Annotations ambiguous: xxx

Sentences with no target labels: xx,xxx

Sentences with exactly one mapped label: x,xxx
Sentences with multiple mapped labels: x,xxx

High-confidence single-label candidates: x,xxx
```

Also print the final class distribution:

```text
Notice_Requirement        XXXX
Right_to_Correct           XXX
Right_to_Delete            XXX
Right_to_Know              XXX
Right_to_Limit_Sensitive   XXX
Right_to_Opt_Out           XXX
```

Do not rebalance, oversample, undersample, or augment anything.

We first want to observe the natural result.

---

# Required DB/2 Sanity Check

Before running the full corpus, verify the parser manually against:

```text
Annotations/DB/2.csv
Htmls/DB/2.html
```

Specifically verify that fragments such as:

```text
the right to request deletion
```

and:

```text
or correction of your data
```

are mapped back to their **full containing sentence**.

Also verify that the full source sentence receives multiple labels and is therefore excluded from the single-label candidate dataset.

The parser should **not** output:

```text
"or correction of your data"
```

as an ML sentence.

That would indicate failure.

---

# Important Non-Goals

Do not:

* train any ML model;
* rebalance classes;
* create synthetic examples;
* optimize train/validation/test splits;
* modify Matthew's current experiment;
* overwrite existing dataset files;
* infer labels from unlabeled text;
* treat unlabeled sentences as negatives;
* convert multi-label sentences into several contradictory single-label rows;
* use LLM classification to fill missing annotations.

This task is only:

> **Convert the C3PA annotation structure into an inspectable set of full grammatical sentence candidates with exactly one mapped CPRA label.**

---

# Implementation Preference

Keep the implementation understandable.

A reasonable structure would be:

```text
parse_c3pa_sentences.py
```

with a few small functions such as:

```python
extract_document_text()
segment_sentences()
normalize_for_matching()
load_annotations()
map_label()
align_annotation_to_sentences()
aggregate_sentence_labels()
export_candidates()
print_summary()
```

Avoid unnecessary abstractions, databases, APIs, classes, or frameworks.

This should be something Matthew can read and explain.

---

# Acceptance Criteria

The implementation is successful when all of the following are true:

1. Sentence boundaries come from the original HTML, not the annotation CSV.

2. Short C3PA fragments such as:

   ```text
   or correction of your data
   ```

   resolve to their complete source sentence.

3. Multi-sentence C3PA annotations can contribute labels to multiple full source sentences.

4. Multiple annotators' annotations are aggregated at the sentence level.

5. A sentence with more than one mapped CPRA label is excluded from the single-label candidate file.

6. No unlabeled sentence is automatically assigned a label.

7. Unmatched and ambiguous annotations are preserved for review.

8. The parser reports class counts and alignment statistics.

9. The original C3PA files remain untouched.

10. DB/2 can be manually inspected and the output clearly corresponds to the source HTML.

---

# Final Instruction to the Agent

Before attempting any full-dataset processing:

> Implement and demonstrate the pipeline using only `DB/2`. Show the reconstructed sentences, the C3PA annotations aligned to each sentence, the resulting mapped-label set, and whether each sentence would be retained or excluded. Do not proceed to the complete corpus until the DB/2 output has been reviewed.

