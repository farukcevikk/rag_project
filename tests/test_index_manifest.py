import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "app"))

from rag_pipeline import validate_index_manifest  # noqa: E402


class IndexManifestTests(unittest.TestCase):
    def create_fixture(self, directory, indexed=2, expected=2):
        source = directory / "manual.md"
        source.write_text("local test data", encoding="utf-8")
        manifest = directory / "index_manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "expected_parent_count": expected,
            "indexed_parent_count": indexed,
            "sources": [{
                "source_file": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "parent_count": expected,
            }],
        }), encoding="utf-8")
        store = SimpleNamespace(store={str(index): object() for index in range(indexed)})
        bm25 = SimpleNamespace(docs=[object() for _ in range(indexed)])
        return source, manifest, store, bm25

    def test_complete_matching_index_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, manifest, store, bm25 = self.create_fixture(directory)

            result = validate_index_manifest(
                store, bm25, str(manifest), str(directory)
            )

            self.assertEqual(result["status"], "verified")

    def test_partial_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, manifest, store, bm25 = self.create_fixture(
                directory, indexed=1, expected=2
            )

            with self.assertRaisesRegex(RuntimeError, "eksik parent"):
                validate_index_manifest(store, bm25, str(manifest), str(directory))

    def test_source_change_marks_index_stale(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, manifest, store, bm25 = self.create_fixture(directory)
            source.write_text("changed local test data", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "indeks güncellenmemiş"):
                validate_index_manifest(store, bm25, str(manifest), str(directory))


if __name__ == "__main__":
    unittest.main()
