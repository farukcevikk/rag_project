# ingest.py

import gc
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
try:
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain.retrievers import ParentDocumentRetriever
try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    from langchain.storage import InMemoryStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = APP_DIR / "chroma_db"
MANIFEST_PATH = DATA_DIR / "index_manifest.json"
PARENT_STORE_PATH = DATA_DIR / "parent_store.pkl"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
SOURCE_PATHS = (
    DATA_DIR / "user_manuel_dev.md",
    DATA_DIR / "ui-user-guide.md",
)
INDEX_SCHEMA_VERSION = 1
MAX_EMBEDDING_ATTEMPTS = 3


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_sha256(path):
    return sha256_bytes(path.read_bytes())


def split_source(path):
    """Create parent documents while keeping all paths independent of cwd."""
    markdown_text = path.read_text(encoding="utf-8")
    headers_to_split_on = [
        ("#", "Ana_Baslik"),
        ("##", "Alt_Baslik"),
        ("###", "Detay_Baslik"),
        ("####", "Mikro_Baslik"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    documents = parent_splitter.split_documents(
        markdown_splitter.split_text(markdown_text)
    )
    for document in documents:
        document.metadata["Source_File"] = path.name
    return documents


def add_headers(document):
    header_parts = [
        document.metadata[key]
        for key in ("Ana_Baslik", "Alt_Baslik", "Detay_Baslik", "Mikro_Baslik")
        if document.metadata.get(key)
    ]
    if header_parts:
        document.page_content = f"{' > '.join(header_parts)}\n\n{document.page_content}"


def deterministic_parent_id(document, ordinal):
    identity = (
        f"{document.metadata.get('Source_File', '')}:{ordinal}:"
        f"{sha256_bytes(document.page_content.encode('utf-8'))}"
    )
    return sha256_bytes(identity.encode("utf-8"))


def add_with_retry(vectorstore, documents, ids, source_file, ordinal):
    """Retry transient local embedding failures without logging manual text."""
    for attempt in range(1, MAX_EMBEDDING_ATTEMPTS + 1):
        try:
            vectorstore.add_documents(documents, ids=ids)
            return
        except Exception as exc:
            if attempt == MAX_EMBEDDING_ATTEMPTS:
                content_hash = sha256_bytes(
                    "\n".join(item.page_content for item in documents).encode("utf-8")
                )
                raise RuntimeError(
                    "Embedding başarısız; eksik indeks yayımlanmadı. "
                    f"source={source_file} parent={ordinal} "
                    f"content_sha256={content_hash} error={type(exc).__name__}"
                ) from exc
            time.sleep(0.5 * attempt)


def dump_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def publish_index(stage_dir, parent_tmp, bm25_tmp, manifest_tmp):
    """Publish all index artifacts together and keep the old version recoverable."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = DATA_DIR / "index_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    current_artifacts = (
        (CHROMA_DIR, backup_dir / "chroma_db"),
        (PARENT_STORE_PATH, backup_dir / PARENT_STORE_PATH.name),
        (BM25_PATH, backup_dir / BM25_PATH.name),
        (MANIFEST_PATH, backup_dir / MANIFEST_PATH.name),
    )
    moved_old = []
    try:
        for current, backup in current_artifacts:
            if current.exists():
                os.replace(current, backup)
                moved_old.append((current, backup))

        os.replace(stage_dir, CHROMA_DIR)
        os.replace(parent_tmp, PARENT_STORE_PATH)
        os.replace(bm25_tmp, BM25_PATH)
        os.replace(manifest_tmp, MANIFEST_PATH)
    except Exception:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        for current in (PARENT_STORE_PATH, BM25_PATH, MANIFEST_PATH):
            if current.exists():
                current.unlink()
        for current, backup in reversed(moved_old):
            if backup.exists():
                os.replace(backup, current)
        raise
    return backup_dir


def ingest_data():
    print(">>> 1. Kaynaklar doğrulanıyor...")
    missing = [str(path) for path in SOURCE_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Eksik kılavuz dosyaları: {missing}")

    all_parent_documents = []
    source_stats = []
    for path in SOURCE_PATHS:
        documents = split_source(path)
        all_parent_documents.extend(documents)
        source_stats.append({
            "source_file": path.name,
            "sha256": source_sha256(path),
            "parent_count": len(documents),
        })
        print(f"   -> {path.name}: {len(documents)} parent")

    if not all_parent_documents:
        raise RuntimeError("Hiç parent doküman üretilemedi.")

    stage_dir = Path(tempfile.mkdtemp(prefix=".chroma_stage_", dir=APP_DIR))
    token = sha256_bytes(str(time.time_ns()).encode("utf-8"))[:12]
    parent_tmp = DATA_DIR / f".parent_store.{token}.tmp"
    bm25_tmp = DATA_DIR / f".bm25_index.{token}.tmp"
    manifest_tmp = DATA_DIR / f".index_manifest.{token}.tmp"

    try:
        print(">>> 2. Yeni indeks staging alanında oluşturuluyor...")
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest",
            base_url="http://localhost:11434",
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
        )
        vectorstore = Chroma(
            persist_directory=str(stage_dir),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=None,
        )

        successful_parents = []
        child_count = 0
        for ordinal, document in enumerate(all_parent_documents, 1):
            add_headers(document)
            parent_id = deterministic_parent_id(document, ordinal)
            sub_documents = child_splitter.split_documents([document])
            valid_sub_documents = []
            for child in sub_documents:
                clean_text = child.page_content.replace("\x00", "").strip()
                if len(clean_text) <= 2:
                    continue
                child.page_content = clean_text
                child.metadata[retriever.id_key] = parent_id
                valid_sub_documents.append(child)
            if not valid_sub_documents:
                raise RuntimeError(
                    "Boş parent algılandı; eksik indeks yayımlanmadı. "
                    f"source={document.metadata.get('Source_File', '')} parent={ordinal}"
                )

            child_ids = [
                f"{parent_id}:{child_index}"
                for child_index in range(len(valid_sub_documents))
            ]
            add_with_retry(
                vectorstore,
                valid_sub_documents,
                child_ids,
                document.metadata.get("Source_File", ""),
                ordinal,
            )
            successful_parents.append((parent_id, document))
            child_count += len(valid_sub_documents)

        if len(successful_parents) != len(all_parent_documents):
            raise RuntimeError("Parent sayısı doğrulaması başarısız; indeks yayımlanmadı.")
        actual_child_count = vectorstore._collection.count()
        if actual_child_count != child_count:
            raise RuntimeError(
                "Child sayısı doğrulaması başarısız; indeks yayımlanmadı. "
                f"expected={child_count} actual={actual_child_count}"
            )

        store.mset(successful_parents)
        valid_parent_documents = [document for _, document in successful_parents]
        bm25_retriever = BM25Retriever.from_documents(valid_parent_documents)
        dump_pickle(parent_tmp, store)
        dump_pickle(bm25_tmp, bm25_retriever)

        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": source_stats,
            "expected_parent_count": len(all_parent_documents),
            "indexed_parent_count": len(successful_parents),
            "indexed_child_count": child_count,
            "parent_chunk_size": 1500,
            "parent_chunk_overlap": 100,
            "child_chunk_size": 400,
            "child_chunk_overlap": 80,
            "embedding_model": "bge-m3:latest",
        }
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        del retriever
        del vectorstore
        gc.collect()

        print(">>> 3. Tam indeks atomik olarak yayımlanıyor...")
        backup_dir = publish_index(stage_dir, parent_tmp, bm25_tmp, manifest_tmp)
        print(
            ">>> BAŞARILI: "
            f"{len(successful_parents)} parent / {child_count} child indekslendi."
        )
        print(f"   -> Önceki indeks yedeği: {backup_dir}")
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        for temporary in (parent_tmp, bm25_tmp, manifest_tmp):
            if temporary.exists():
                temporary.unlink()
        raise


if __name__ == "__main__":
    ingest_data()
