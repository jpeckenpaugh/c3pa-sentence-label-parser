import contextlib
import csv
import io
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import parse_c3pa_sentences as parser

SCRIPT = os.path.join(ROOT, "parse_c3pa_sentences.py")

HTML = """<!doctype html>
<html><body>
<h1>Information</h1>
<table>
  <tr><th>Information</th><th>Category</th></tr>
  <tr><td>Account Data</td><td>Email</td></tr>
</table>
<iframe>{"junk": "iframe content must not be extracted"}</iframe>
<div hidden>Hidden text must not be extracted.</div>
<p>We may share your personal information with service providers.</p>
<p>You can ask us to delete your data.</p>
<p>You can request correction of your data or opt out of the sale of your data.</p>
<p>Contact us at privacy@example.com for more information about this policy.</p>
<p>See <a href="#">our privacy policy</a> for details.</p>
<p>Skip to main content</p>
<a>Print</a>
</body></html>
"""

ANNOTATIONS = """RANumb,Text,Label
ra1,We may share your personal information with service providers.,Categories of Personal Information Shared / Disclosed
ra1,You can ask us to delete your data.,Description of Right to Delete
ra2,delete your data,Description of Right to Delete
ra1,"You can request correction of your data or opt out of the sale of your data.",Description of Right to Correct Information
ra2,opt out of the sale of your data,Description of Right to Opt-out of sale of PI
ra3,Information,Others
ra3,"Information We may share your personal information with service providers.",Categories of Personal Information Shared / Disclosed
ra4,Account Data,Others
ra4,Print,Others
"""


def build_dataset(root):
    dataset = os.path.join(root, "C3PA_Dataset")
    os.makedirs(os.path.join(dataset, "Htmls", "DB"))
    os.makedirs(os.path.join(dataset, "Annotations", "DB"))
    with open(os.path.join(dataset, "Htmls", "DB", "1.html"), "w", encoding="utf-8") as f:
        f.write(HTML)
    with open(os.path.join(dataset, "Annotations", "DB", "1.csv"), "w", encoding="utf-8") as f:
        f.write(ANNOTATIONS)
    return dataset


def run_pipeline_quiet(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return parser.run_pipeline(*args, **kwargs)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class ExtractionTests(unittest.TestCase):
    def test_blocks_have_kinds_and_skip_non_content(self):
        blocks = parser.extract_readable_blocks(HTML)
        texts = [block["text"] for block in blocks]
        kinds = {block["text"]: block["kind"] for block in blocks}

        self.assertNotIn('{"junk": "iframe content must not be extracted"}', texts)
        self.assertFalse(any("Hidden text" in text for text in texts))
        self.assertEqual(kinds["Information"], "heading")
        self.assertEqual(kinds["Account Data"], "cell")
        self.assertEqual(kinds["See our privacy policy for details."], "prose")

    def test_inline_anchor_text_stays_in_sentence(self):
        blocks = parser.extract_readable_blocks(HTML)
        texts = [block["text"] for block in blocks]
        self.assertIn("See our privacy policy for details.", texts)


class SentenceQualityTests(unittest.TestCase):
    def test_heading_and_cell_are_not_sentences(self):
        self.assertFalse(parser.classify_sentence("Information", "heading")[0])
        self.assertFalse(parser.classify_sentence("Account Data", "cell")[0])

    def test_short_and_punctuation_rules(self):
        self.assertTrue(parser.classify_sentence("We use cookies.", "prose")[0])
        self.assertFalse(parser.classify_sentence("Print", "prose")[0])
        self.assertFalse(parser.classify_sentence("Skip to main content", "prose")[0])
        self.assertFalse(parser.classify_sentence("COLLECTION OF PERSONAL DATA", "prose")[0])
        self.assertFalse(parser.classify_sentence("How We Use the Data We Collect", "list")[0])

    def test_machine_text_filter(self):
        self.assertTrue(parser.looks_like_machine_text("{" + '"k": "v", ' * 100 + "}"))
        self.assertTrue(parser.looks_like_machine_text("https://example.com/" + "a" * 300))
        self.assertFalse(
            parser.looks_like_machine_text(
                "For details, see https://example.com/privacy-policy for more information."
            )
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dataset = build_dataset(self.tmp.name)
        self.out = os.path.join(self.tmp.name, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fragment_alignment_and_tiers(self):
        summary = run_pipeline_quiet(
            self.dataset, self.out, single_doc="DB/1"
        )
        single = read_csv(os.path.join(self.out, "single_label_sentences_all.csv"))
        multi = read_csv(os.path.join(self.out, "multi_label_sentences.csv"))
        unannotated = read_csv(os.path.join(self.out, "unannotated_sentences.csv"))
        fragments = read_csv(os.path.join(self.out, "fragments_excluded.csv"))

        single_texts = {row["sentence_text"] for row in single}
        multi_texts = {row["sentence_text"] for row in multi}
        unannotated_texts = {row["sentence_text"] for row in unannotated}
        fragment_texts = {row["sentence_text"] for row in fragments}

        # A sub-sentence annotation resolves to the full source sentence.
        self.assertIn("You can ask us to delete your data.", single_texts)
        self.assertNotIn("delete your data", single_texts)

        # Two labels produce a multi-label row, not two single-label rows.
        self.assertIn(
            "You can request correction of your data or opt out of the sale of your data.",
            multi_texts,
        )
        self.assertNotIn(
            "You can request correction of your data or opt out of the sale of your data.",
            single_texts,
        )

        self.assertIn(
            "Contact us at privacy@example.com for more information about this policy.",
            unannotated_texts,
        )

        # Headings, table cells, and UI controls never enter candidate tiers.
        self.assertIn("Information", fragment_texts)
        self.assertIn("Account Data", fragment_texts)
        self.assertIn("Print", fragment_texts)
        for row in single + multi + unannotated:
            self.assertNotIn(row["sentence_kind"], {"heading", "cell", "ui"})
            self.assertNotEqual(row["sentence_text"], "Information")
            self.assertNotEqual(row["sentence_text"], "Account Data")
            self.assertNotEqual(row["sentence_text"], "Print")

    def test_diagnostics_reconcile(self):
        summary = run_pipeline_quiet(
            self.dataset, self.out, single_doc="DB/1"
        )
        tiers = (
            summary["single_label_all"]
            + summary["multi_label"]
            + summary["sentences_unannotated"]
        )
        self.assertEqual(tiers, summary["sentences_evaluated"])
        self.assertEqual(
            summary["sentences_evaluated"] + summary["fragments_excluded"],
            summary["sentences_extracted"],
        )
        self.assertEqual(
            summary["annotations_aligned_distinct"]
            + summary["annotations_unmatched"]
            + summary["annotations_ambiguous"],
            summary["annotations_considered"],
        )
        self.assertEqual(
            summary["annotations_considered"] + summary["empty_annotation_rows_skipped"],
            summary["source_annotation_rows"],
        )

    def test_source_paths_are_dataset_relative(self):
        run_pipeline_quiet(self.dataset, self.out, single_doc="DB/1")
        rows = read_csv(os.path.join(self.out, "single_label_sentences_all.csv"))
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["source_html"].startswith("C3PA_Dataset/"))
            self.assertTrue(row["source_annotation_file"].startswith("C3PA_Dataset/"))

    def test_overlap_threshold_is_configurable(self):
        html_path = os.path.join(self.dataset, "Htmls", "DB", "7.html")
        csv_path = os.path.join(self.dataset, "Annotations", "DB", "7.csv")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><p>Alpha beta gamma delta epsilon zeta.</p></body></html>")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("RANumb,Text,Label\nra1,Alpha beta gamma iota,Others\n")

        # The annotation shares 3 of its 4 tokens with the sentence: 0.75.
        permissive = parser.process_document(
            "DB_7",
            html_path,
            csv_path,
            dataset_dir=self.dataset,
            overlap_threshold=0.75,
        )
        strict = parser.process_document(
            "DB_7",
            html_path,
            csv_path,
            dataset_dir=self.dataset,
            overlap_threshold=0.76,
        )

        self.assertEqual(len(permissive["aligned_matches"]), 1)
        self.assertEqual(permissive["aligned_matches"][0][2], "partial_overlap")
        self.assertEqual(len(strict["aligned_matches"]), 0)
        self.assertEqual(len(strict["unmatched"]), 1)

    def test_overlap_threshold_cli_validation(self):
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--dataset-dir",
                self.dataset,
                "--output-dir",
                self.out,
                "--overlap-threshold",
                "1.5",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overlap threshold must be between 0 and 1", result.stderr)

    def test_sanity_check_does_not_overwrite_main_output(self):
        main_output = os.path.join(self.tmp.name, "output")
        os.makedirs(main_output)
        sentinel = os.path.join(main_output, "single_label_sentences_all.csv")
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write("SENTINEL\n")

        result = subprocess.run(
            [sys.executable, SCRIPT, "--sanity-check", "DB/1"],
            cwd=self.tmp.name,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(sentinel, encoding="utf-8") as f:
            self.assertEqual(f.read(), "SENTINEL\n")
        self.assertTrue(
            os.path.exists(
                os.path.join(main_output, "sanity", "DB_1", "single_label_sentences_all.csv")
            )
        )


if __name__ == "__main__":
    unittest.main()
