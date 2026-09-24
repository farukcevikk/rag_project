"""GPU-free checks for the Jetson transfer boundary."""

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from deploy import copilot_bundle


class CopilotBundleTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="copilot-bundle-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.project = self.root / "project"
        source = b"An immutable, local test manual.\n"
        for name in (*copilot_bundle.CODE_REQUIRED, "deploy/copilot_bundle.py", "deploy/README.md", "docs/README.md"):
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# test code\n")
        data = self.project / "data"
        data.mkdir(parents=True)
        (data / "user_manuel_dev.md").write_bytes(source)
        (data / "bm25_index.pkl").write_bytes(b"test-bm25")
        (data / "parent_store.pkl").write_bytes(b"test-parents")
        manifest = {
            "schema_version": 2,
            "sources": [{
                "source_file": "user_manuel_dev.md",
                "sha256": hashlib.sha256(source).hexdigest(),
            }],
            "expected_parent_count": 1,
            "indexed_parent_count": 1,
            "indexed_child_count": 1,
            "embedding_model": "bge-m3:latest",
        }
        (data / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        chroma = self.project / "app/chroma_db"
        chroma.mkdir(parents=True)
        database = sqlite3.connect(chroma / "chroma.sqlite3")
        database.execute("CREATE TABLE embeddings (id TEXT)")
        database.execute("INSERT INTO embeddings VALUES ('child-1')")
        database.commit()
        database.close()
        vector = chroma / "test-collection"
        vector.mkdir()
        (vector / "data_level0.bin").write_bytes(b"test-vector")

    def test_build_verify_and_stage_never_change_source_or_existing_release(self):
        original = (self.project / "data/user_manuel_dev.md").read_bytes()
        bundle = copilot_bundle.build_bundle(self.project, self.root / "out", "release-test-1")
        self.assertEqual((bundle / "private-data.tar.gz").stat().st_mode & 0o777, 0o600)
        envelope, code, private = copilot_bundle.verify_bundle(bundle)
        self.assertEqual(envelope["release_id"], "release-test-1")
        self.assertIn("app/copilot_api.py", code)
        self.assertIn("deploy/v5_podman_copilot.git.patch", code)
        self.assertNotIn("data/user_manuel_dev.md", code)
        self.assertEqual(private["data/user_manuel_dev.md"], original)
        self.assertIn("app/chroma_db/chroma.sqlite3", private)
        staged = copilot_bundle.install_bundle(bundle, self.root / "target/install")
        self.assertEqual((staged / "data/user_manuel_dev.md").read_bytes(), original)
        self.assertEqual((staged / "data/user_manuel_dev.md").stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.project / "data/user_manuel_dev.md").read_bytes(), original)
        with self.assertRaises(FileExistsError):
            copilot_bundle.install_bundle(bundle, self.root / "target/install")

    def test_source_mismatch_stops_before_a_bundle_is_created(self):
        (self.project / "data/user_manuel_dev.md").write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "Manual/index hash mismatch"):
            copilot_bundle.build_bundle(self.project, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_chroma_child_count_mismatch_is_rejected(self):
        manifest_path = self.project / "data/index_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["indexed_child_count"] = 2
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Chroma child count differs"):
            copilot_bundle.build_bundle(self.project, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_archive_corruption_and_unsafe_member_are_rejected(self):
        bundle = copilot_bundle.build_bundle(self.project, self.root / "out", "release-test-2")
        with (bundle / "code.tar.gz").open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "Archive checksum mismatch"):
            copilot_bundle.verify_bundle(bundle)
        for name in ("../escape", "/absolute", "a/../b", "a//b", "."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                copilot_bundle._safe_name(name)

    def test_rendered_service_uses_target_paths_and_keeps_candidates_off(self):
        bundle = copilot_bundle.build_bundle(self.project, self.root / "out", "release-test-3")
        staged = copilot_bundle.install_bundle(bundle, self.root / "target/install")
        python = self.root / "target/jetson_env/bin/python"
        python.parent.mkdir(parents=True)
        system_python = self.root / "target/system-python"
        system_python.write_bytes(b"python placeholder")
        python.symlink_to(system_python)
        public_key = self.root / "target/v5/secrets/jwt_public.pem"
        public_key.parent.mkdir(parents=True)
        public_key.write_bytes(b"public test key")
        unit = copilot_bundle.render_service(staged, python, public_key, "toyota")
        content = unit.read_text(encoding="utf-8")
        self.assertIn(f"WorkingDirectory={staged}", content)
        self.assertIn(f"ExecStart={python} -m uvicorn", content)
        self.assertNotIn(f"ExecStart={system_python} -m uvicorn", content)
        self.assertIn(f"JWT_PUBLIC_KEY_PATH={public_key}", content)
        self.assertIn("COPILOT_PAGE_CONTEXT_ENABLED=false", content)
        self.assertIn("COPILOT_CLARIFICATION_ENABLED=false", content)
        self.assertIn(
            f"COPILOT_FEEDBACK_LOG_PATH={staged.parent.parent / 'log/user_feedback_logs.jsonl'}",
            content,
        )
        self.assertNotIn("public test key", content)
        with self.assertRaises(FileExistsError):
            copilot_bundle.render_service(staged, python, public_key, "toyota")


if __name__ == "__main__":
    unittest.main()
