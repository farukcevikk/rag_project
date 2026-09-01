"""Legacy-compatible index manifest validation."""

import hashlib
import json
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR.parent / "data"
INDEX_MANIFEST_PATH = DATA_DIR / "index_manifest.json"
INDEX_SCHEMA_VERSION = 2


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_index_manifest(
    store,
    bm25_retriever,
    manifest_path=INDEX_MANIFEST_PATH,
    data_dir=DATA_DIR,
):
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    if not manifest_path.exists():
        return {"status": "legacy_unverified", "schema_version": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RuntimeError("Desteklenmeyen indeks manifest sürümü. İndeksi yeniden oluşturun.")
    expected = manifest.get("expected_parent_count")
    indexed = manifest.get("indexed_parent_count")
    if not isinstance(expected, int) or expected <= 0 or indexed != expected:
        raise RuntimeError("İndeks manifesti eksik parent bildiriyor. İndeksi yeniden oluşturun.")
    store_values = getattr(store, "store", None)
    if not isinstance(store_values, dict) or len(store_values) != indexed:
        raise RuntimeError("Parent Store manifestle uyuşmuyor. İndeksi yeniden oluşturun.")
    bm25_documents = getattr(bm25_retriever, "docs", None)
    if bm25_documents is None or len(bm25_documents) != indexed:
        raise RuntimeError("BM25 indeksi manifestle uyuşmuyor. İndeksi yeniden oluşturun.")
    for source in manifest.get("sources", []):
        source_file = Path(str(source.get("source_file", ""))).name
        source_path = data_dir / source_file
        expected_hash = source.get("sha256")
        if not source_file or not expected_hash or not source_path.exists():
            raise RuntimeError("İndeks kaynak manifesti geçersiz. İndeksi yeniden oluşturun.")
        if file_sha256(source_path) != expected_hash:
            raise RuntimeError(
                f"Kılavuz değişmiş ancak indeks güncellenmemiş: {source_file}. "
                "İndeksi yeniden oluşturun."
            )
    return {"status": "verified", "schema_version": manifest["schema_version"]}
