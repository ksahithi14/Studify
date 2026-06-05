import unittest

from app.services.pdf_utils import (
    chunk_sentences,
    clean_text,
    normalize_subject,
    parse_numbered_lines,
    split_sentences,
)


class TextUtilsTests(unittest.TestCase):
    def test_normalize_subject_handles_aliases(self) -> None:
        self.assertEqual(normalize_subject("rm1"), "RMI")
        self.assertEqual(normalize_subject("Research Methodology"), "RMI")
        self.assertEqual(normalize_subject("dbms"), "DBMS")

    def test_clean_and_split_text(self) -> None:
        text = "Hello@@ world!!\nThis is   a test."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Hello world!! This is a test.")
        self.assertEqual(
            split_sentences(cleaned),
            ["Hello world!!", "This is a test."],
        )

    def test_chunk_sentences_groups_content(self) -> None:
        sentences = ["A.", "B.", "C.", "D."]
        self.assertEqual(
            chunk_sentences(sentences, max_sentences=2),
            ["A. B.", "C. D."],
        )

    def test_parse_numbered_lines_handles_lists(self) -> None:
        raw = "1. What is AI?\n2) Define bias.\n- Explain RAG."
        self.assertEqual(
            parse_numbered_lines(raw),
            ["What is AI?", "Define bias.", "Explain RAG."],
        )


if __name__ == "__main__":
    unittest.main()
