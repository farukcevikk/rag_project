import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "app"))

from rag_pipeline import select_reranked_context  # noqa: E402


def document(section, content):
    return Document(
        page_content=content,
        metadata={
            "Source_File": "manual.md",
            "Ana_Baslik": "Main",
            "Alt_Baslik": section,
        },
    )


class RerankerSelectionTests(unittest.TestCase):
    def test_duplicate_top_sections_receive_diverse_supplement(self):
        first = document("Parts", "first")
        second = document("Parts", "second")
        third = document("Architecture", "third")

        selected = select_reranked_context([
            (3.0, first), (2.0, second), (1.0, third)
        ])

        self.assertEqual(selected, [(3.0, first), (2.0, second), (1.0, third)])

    def test_distinct_top_sections_keep_original_top_two(self):
        first = document("Parts", "first")
        second = document("Architecture", "second")
        third = document("FAQ", "third")

        selected = select_reranked_context([
            (3.0, first), (2.0, second), (1.0, third)
        ])

        self.assertEqual(selected, [(3.0, first), (2.0, second)])

    def test_best_ranked_distinct_section_is_selected(self):
        first = document("Parts", "first")
        second = document("Parts", "second")
        third = document("Parts", "third")
        fourth = document("GALC", "fourth")

        selected = select_reranked_context([
            (4.0, first), (3.0, second), (2.0, third), (1.0, fourth)
        ])

        self.assertEqual(selected[-1], (1.0, fourth))


if __name__ == "__main__":
    unittest.main()
