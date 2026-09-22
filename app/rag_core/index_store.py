import hashlib
import json
import pickle
import warnings
from pathlib import Path

import ollama
from langchain_community.vectorstores import Chroma
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = APP_DIR / "chroma_db"
MANIFEST_PATH = DATA_DIR / "index_manifest.json"
PARENT_STORE_PATH = DATA_DIR / "parent_store.pkl"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
INDEX_SCHEMA_VERSION = 2


class ResidentOllamaEmbeddings:
    """Index-compatible Ollama embeddings with explicit model residency.

    The prefixes match LangChain's legacy OllamaEmbeddings defaults used while
    building the existing index. Chroma uses cosine distance, and the direct
    Ollama vectors were verified cosine-equivalent to the legacy client.
    """

    def __init__(self, model, base_url, keep_alive="30m", client=None):
        self.model = model
        self.keep_alive = keep_alive
        self.client = client or ollama.Client(host=base_url)

    def embed_query(self, text):
        response = self.client.embed(
            model=self.model,
            input=f"query: {text}",
            keep_alive=self.keep_alive,
        )
        return response["embeddings"][0]

    def embed_documents(self, texts):
        if not texts:
            return []
        response = self.client.embed(
            model=self.model,
            input=[f"passage: {text}" for text in texts],
            keep_alive=self.keep_alive,
        )
        return response["embeddings"]

class AnchoredParentRetriever:
    """Search child chunks while returning their parent with the matched anchor.

    Parent documents give generation enough surrounding context, but reranking the
    whole parent can dilute the exact child passage that made the document relevant.
    This adapter preserves that passage as runtime-only metadata. Nothing is written
    back to the immutable store or source manuals.
    """

    def __init__(self, vectorstore, docstore, child_splitter, search_kwargs=None, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.child_splitter = child_splitter
        self.search_kwargs = dict(search_kwargs or {})
        self.id_key = id_key

    def invoke(self, query):
        children = self.vectorstore.similarity_search(query, **self.search_kwargs)
        parent_ids = []
        anchors_by_parent = {}
        for child in children:
            parent_id = child.metadata.get(self.id_key)
            if not parent_id:
                continue
            if parent_id not in anchors_by_parent:
                parent_ids.append(parent_id)
                anchors_by_parent[parent_id] = []
            anchor = child.page_content.strip()
            if anchor and anchor not in anchors_by_parent[parent_id]:
                anchors_by_parent[parent_id].append(anchor)

        parents = self.docstore.mget(parent_ids)
        results = []
        for parent_id, parent in zip(parent_ids, parents):
            if parent is None:
                continue
            metadata = dict(parent.metadata or {})
            metadata["_retrieval_anchors"] = anchors_by_parent[parent_id]
            results.append(Document(page_content=parent.page_content, metadata=metadata))
        return results


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(store, bm25_retriever):
    """Fail before serving when source, parent store, or BM25 index is stale."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RuntimeError("Desteklenmeyen indeks manifest sürümü")
    expected = manifest.get("expected_parent_count")
    indexed = manifest.get("indexed_parent_count")
    if not isinstance(expected, int) or expected <= 0 or indexed != expected:
        raise RuntimeError("İndeks manifest parent sayısı geçersiz")
    if len(getattr(store, "store", {})) != indexed:
        raise RuntimeError("Parent store manifestle uyuşmuyor")
    if len(getattr(bm25_retriever, "docs", [])) != indexed:
        raise RuntimeError("BM25 index manifestle uyuşmuyor")
    for source in manifest.get("sources", []):
        source_path = DATA_DIR / Path(str(source.get("source_file", ""))).name
        if not source_path.exists() or file_sha256(source_path) != source.get("sha256"):
            raise RuntimeError(f"Kaynak/index hash uyuşmazlığı: {source_path.name}")
    return manifest


def load_index(config):
    embeddings = ResidentOllamaEmbeddings(
        model=config.embedding_model,
        base_url=config.ollama_host,
        keep_alive=config.keep_alive,
    )
    # langchain-chroma is not part of the deployed Jetson environment yet.
    # Keep the index-compatible implementation and silence only its known
    # constructor deprecation notice; all other warnings remain visible.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The class `Chroma` was deprecated.*",
            category=LangChainDeprecationWarning,
        )
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    with PARENT_STORE_PATH.open("rb") as handle:
        store = pickle.load(handle)
    with BM25_PATH.open("rb") as handle:
        bm25 = pickle.load(handle)
    bm25.k = config.bm25_k
    vector = AnchoredParentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
        ),
        search_kwargs={"k": config.vector_k},
    )
    manifest = validate_manifest(store, bm25)
    return embeddings, vector, bm25, manifest
