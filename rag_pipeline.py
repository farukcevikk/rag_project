# rag_pipeline.py 

import time 
import pickle 
import numpy as np 
import json 
import os
import re
import hashlib
import ollama # Langchain yerine doğrudan Ollama'nın resmi kütüphanesi 

from langchain_community.embeddings import OllamaEmbeddings 
from langchain_community.vectorstores import Chroma 
try:
    # LangChain 1.x
    from langchain_classic.retrievers import ParentDocumentRetriever
except ImportError:
    # LangChain 0.2.x (Jetson ortamı)
    from langchain.retrievers import ParentDocumentRetriever
try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.retrievers import BM25Retriever 
from sentence_transformers import CrossEncoder 
import onnxruntime as ort

# ===================================================== 
# YARDIMCI FONKSİYONLAR 
# ===================================================== 

def cosine_similarity(vec1, vec2): 
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(np.dot(vec1, vec2) / denominator) if denominator else 0.0

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "data"))
INDEX_MANIFEST_PATH = os.path.join(DATA_DIR, "index_manifest.json")
INDEX_SCHEMA_VERSION = 2
EMPTY_ANSWER_FALLBACK = "Şu anda yanıt oluşturamadım. Lütfen tekrar deneyin."
OUT_OF_SCOPE_ANSWER = (
    "Bu asistan 1PAVI kullanım kılavuzu kapsamında yanıt verir. "
    "Bu soru için güvenilir bir kılavuz kanıtım yok."
)
GROUNDING_THRESHOLD = -2.5
EVIDENCE_RESCUE_MIN_SCORE = -4.0
EVIDENCE_RESCUE_MIN_MARGIN = 1.5
EVIDENCE_CONSENSUS_MAX_RANK = 3
DENSE_ANCHOR_MAX_SUPPORT_RANK = 2
DENSE_ANCHOR_MIN_CONTENT_SCORE = -6.0
DENSE_ANCHOR_MIN_TITLE_MARGIN = 2.0
DENSE_ANCHOR_RESCUE_MIN_SCORE = -5.0
CONTEXT_DOCUMENT_MIN_SCORE = -4.0
GENERIC_QUERY_STOPWORDS = {
    "acikla", "açıkla", "anlama", "anlamina", "anlamına", "bir", "bunu",
    "bu", "de", "da", "gelir", "hangi", "icin", "için", "ile", "kim",
    "mi", "mı", "mu", "mü", "nasil", "nasıl", "ne", "neden", "nedir",
    "nelerdir", "nerede", "olan", "olarak", "once", "önce", "sonra",
    "tarafindan", "tarafından", "ve", "veya",
}
TURKISH_LANGUAGE_TERMS = {
    "acaba", "bir", "bu", "hangi", "icin", "için", "ile", "mi", "mı",
    "mu", "mü", "nasıl", "nasil", "ne", "neden", "nedir", "nelerdir",
    "nerede", "ve", "veya",
}
ENGLISH_LANGUAGE_TERMS = {
    "a", "an", "and", "are", "can", "does", "for", "how", "is", "the",
    "to", "what", "when", "where", "which", "why", "with",
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_index_manifest(
    store,
    bm25_retriever,
    manifest_path=INDEX_MANIFEST_PATH,
    data_dir=DATA_DIR,
):
    """Reject stale or partially built indexes before serving questions."""
    if not os.path.exists(manifest_path):
        return {"status": "legacy_unverified", "schema_version": None}

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
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
        source_file = os.path.basename(str(source.get("source_file", "")))
        source_path = os.path.join(data_dir, source_file)
        expected_hash = source.get("sha256")
        if not source_file or not expected_hash or not os.path.exists(source_path):
            raise RuntimeError("İndeks kaynak manifesti geçersiz. İndeksi yeniden oluşturun.")
        if _file_sha256(source_path) != expected_hash:
            raise RuntimeError(
                f"Kılavuz değişmiş ancak indeks güncellenmemiş: {source_file}. "
                "İndeksi yeniden oluşturun."
            )
    return {"status": "verified", "schema_version": manifest["schema_version"]}


def _matching_tokens(value):
    normalized = str(value or "").casefold().replace("\u0307", "")
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def detect_text_language(value):
    """Detect Turkish/English for routing; return None when evidence is weak."""
    text = str(value or "")
    tokens = set(_matching_tokens(text))
    turkish_score = len(tokens & TURKISH_LANGUAGE_TERMS)
    english_score = len(tokens & ENGLISH_LANGUAGE_TERMS)
    if re.search(r"[çğıöşüÇĞİÖŞÜ]", text):
        turkish_score += 2
    if turkish_score > english_score:
        return "tr"
    if english_score > turkish_score:
        return "en"
    return None


def distinctive_query_terms(query):
    """Return content-bearing query terms using only generic language rules."""
    return {
        token for token in _matching_tokens(query)
        if len(token) >= 3 and token not in GENERIC_QUERY_STOPWORDS and not token.isdigit()
    }


def compress_matching_table_rows(query, content):
    """Keep only fully matched table rows to prevent cross-row generation errors."""
    terms = distinctive_query_terms(query)
    if not terms:
        return content, False
    lines = str(content or "").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        block = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index])
            index += 1
        if len(block) < 3:
            continue
        data_rows = block[2:]
        matching_rows = [
            line for line in data_rows
            if terms & set(_matching_tokens(line))
        ]
        matched_terms = (
            set().union(
                *(terms & set(_matching_tokens(line)) for line in matching_rows)
            )
            if matching_rows else set()
        )
        if matching_rows and matched_terms == terms:
            return "\n".join([block[0], block[1], *matching_rows]), True
    return content, False


def extract_single_table_answer(context):
    """Return a faithful answer when compressed evidence is exactly one data row."""
    table_rows = [
        line.strip() for line in str(context or "").splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    if len(table_rows) != 3:
        return None
    separator_cells = [
        cell.strip() for cell in table_rows[1].strip("|").split("|")
    ]
    if not separator_cells or not all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells
    ):
        return None
    cells = [
        re.sub(r"[*_`]", "", cell).strip()
        for cell in table_rows[2].strip("|").split("|")
    ]
    if len(cells) < 2 or not cells[0] or not any(cells[1:]):
        return None
    meaning = " | ".join(cell for cell in cells[1:] if cell)
    return f"{cells[0]}: {meaning}."


def assess_context_reliability(query, best_score, retrieved_documents):
    """Use calibrated reranker, lexical, and retrieval-consensus evidence."""
    documents = retrieved_documents or []
    scores = [
        float(document.get("reranker_score"))
        for document in documents
        if document.get("reranker_score") is not None
    ]
    score_margin = scores[0] - scores[1] if len(scores) >= 2 else None
    terms = distinctive_query_terms(query)
    evidence_terms = set()
    document_signals = []
    for rank, document in enumerate(documents, 1):
        document_terms = terms & set(_matching_tokens(document.get("content", "")))
        evidence_terms.update(document_terms)
        vector_rank = document.get("vector_rank")
        bm25_rank = document.get("bm25_rank")
        consensus = (
            isinstance(vector_rank, int)
            and isinstance(bm25_rank, int)
            and vector_rank <= EVIDENCE_CONSENSUS_MAX_RANK
            and bm25_rank <= EVIDENCE_CONSENSUS_MAX_RANK
        )
        document_signals.append({
            "rank": rank,
            "score": float(document.get("reranker_score", -999.0)),
            "term_hits": len(document_terms),
            "consensus": consensus,
        })
    top_term_hits = document_signals[0]["term_hits"] if document_signals else 0
    term_hits = len(evidence_terms)
    retrieval_consensus = any(item["consensus"] for item in document_signals)
    consensus_support = next(
        (
            item for item in document_signals
            if item["score"] > EVIDENCE_RESCUE_MIN_SCORE
            and item["consensus"]
            and item["term_hits"] >= 1
        ),
        None,
    )
    dense_anchor_support = next(
        (
            item for item, document in zip(document_signals, documents)
            if document.get("dense_anchor_promoted") is True
            and item["score"] > DENSE_ANCHOR_RESCUE_MIN_SCORE
        ),
        None,
    )

    strict_accept = float(best_score) > GROUNDING_THRESHOLD
    full_term_coverage_rescue = (
        not strict_accept
        and float(best_score) > EVIDENCE_RESCUE_MIN_SCORE
        and bool(terms)
        and top_term_hits == len(terms)
    )
    margin_rescue = (
        not strict_accept
        and not full_term_coverage_rescue
        and float(best_score) > EVIDENCE_RESCUE_MIN_SCORE
        and score_margin is not None
        and score_margin >= EVIDENCE_RESCUE_MIN_MARGIN
        and top_term_hits >= 1
    )
    consensus_rescue = (
        not strict_accept
        and not full_term_coverage_rescue
        and not margin_rescue
        and consensus_support is not None
    )
    dense_anchor_rescue = (
        not strict_accept
        and not full_term_coverage_rescue
        and not margin_rescue
        and not consensus_rescue
        and float(best_score) > EVIDENCE_RESCUE_MIN_SCORE
        and dense_anchor_support is not None
    )
    return {
        "is_reliable": (
            strict_accept or full_term_coverage_rescue or margin_rescue
            or consensus_rescue or dense_anchor_rescue
        ),
        "reason": (
            "strict_score" if strict_accept
            else "full_term_coverage_rescue" if full_term_coverage_rescue
            else "term_margin_rescue" if margin_rescue
            else "retrieval_consensus_rescue" if consensus_rescue
            else "dense_anchor_rescue" if dense_anchor_rescue
            else "rejected"
        ),
        "score_margin": round(score_margin, 4) if score_margin is not None else None,
        "distinctive_term_count": len(terms),
        "distinctive_term_hits": term_hits,
        "top_distinctive_term_hits": top_term_hits,
        "retrieval_consensus": retrieval_consensus,
        "consensus_support_rank": (
            consensus_support["rank"] if consensus_support is not None else None
        ),
    }


def finalize_grounded_answer(answer, sections, context_is_reliable):
    """İç protokolü ayıklar, boş çıktıyı güvenli hale getirir ve kaynağı ekler."""
    basis_match = re.search(r"\[\[BASIS:(MANUAL|CONVERSATION|NONE)\]\]", answer, re.IGNORECASE)
    basis = basis_match.group(1).upper() if basis_match else "NONE"
    clean = re.sub(r"\s*\[\[BASIS:(?:MANUAL|CONVERSATION|NONE)\]\]\s*", "", answer, flags=re.IGNORECASE).strip()
    clean = re.sub(
        r"(?im)^\s*\[Source\s+\d+\s*\|[^\]\n]*\]\s*",
        "",
        clean,
    ).strip()
    clean = re.sub(r"\n*Kaynak/Source\s*:.*$", "", clean, flags=re.IGNORECASE | re.DOTALL).strip()
    clean = re.sub(
        r"(?im)^\s*(?:MANUAL[_ ]CONTEXT[_ ]STATUS|INTERNAL EVIDENCE|İÇ KANIT)\s*(?::|—).*?(?:\n|$)",
        "",
        clean,
    ).strip()
    if not clean:
        clean = EMPTY_ANSWER_FALLBACK
        basis = "NONE"
    if basis == "MANUAL" and context_is_reliable and sections and clean:
        clean = f"{clean}\n\nKaynak/Source: {sections[0]}"
    return clean, basis, basis_match is not None

def custom_hybrid_search(vector_docs, bm25_docs, top_k=15, k=60): 
    fused_scores = {} 
    doc_map = {} 

    for rank, doc in enumerate(vector_docs): 
        doc_id = hash(doc.page_content) 
        if doc_id not in fused_scores: 
            fused_scores[doc_id] = 0 
            doc_map[doc_id] = doc 
        fused_scores[doc_id] += 0.5 * (1 / (rank + k)) 

    for rank, doc in enumerate(bm25_docs): 
        doc_id = hash(doc.page_content) 
        if doc_id not in fused_scores: 
            fused_scores[doc_id] = 0 
            doc_map[doc_id] = doc 
        fused_scores[doc_id] += 0.5 * (1 / (rank + k)) 

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True) 
    return [doc_map[doc_id] for doc_id, score in reranked[:top_k]] 


def _document_section_key(document):
    """Identify a manual section without relying on proprietary content."""
    metadata = getattr(document, "metadata", {}) or {}
    source = metadata.get("Source_File", "")
    main = metadata.get("Ana_Baslik", "")
    section = metadata.get("Alt_Baslik") or metadata.get("Detay_Baslik", "")
    if source or main or section:
        return source, main, section
    return "content", hash(getattr(document, "page_content", ""))


def _heading_number(value):
    """Extract a stable manual hierarchy number from localized headings."""
    match = re.search(r"(?<!\w)(\d+(?:\.\d+)*)", str(value or ""))
    return match.group(1) if match else None


def _document_structural_section_key(document, include_detail=False):
    """Match translated manual sections by hierarchy, not heading language."""
    metadata = getattr(document, "metadata", {}) or {}
    main_number = _heading_number(metadata.get("Ana_Baslik", ""))
    section_number = _heading_number(
        metadata.get("Alt_Baslik") or metadata.get("Detay_Baslik", "")
    )
    if not main_number or not section_number:
        return None
    key = (main_number, section_number)
    if include_detail:
        detail_number = _heading_number(metadata.get("Detay_Baslik", ""))
        return (*key, detail_number)
    return key


def build_section_document_index(documents):
    """Index parent documents by manual section for bounded sibling expansion."""
    section_index = {}
    seen_by_section = {}
    for document in documents or []:
        section_key = _document_section_key(document)
        document_id = hash(getattr(document, "page_content", ""))
        seen = seen_by_section.setdefault(section_key, set())
        if document_id in seen:
            continue
        seen.add(document_id)
        section_index.setdefault(section_key, []).append(document)
    return section_index


def build_structural_section_index(documents):
    """Index equivalent localized sections using their hierarchy numbers."""
    section_index = {}
    seen_by_section = {}
    for document in documents or []:
        section_key = _document_structural_section_key(document)
        if section_key is None:
            continue
        document_id = hash(getattr(document, "page_content", ""))
        seen = seen_by_section.setdefault(section_key, set())
        if document_id in seen:
            continue
        seen.add(document_id)
        section_index.setdefault(section_key, []).append(document)
    return section_index


def expand_anchor_section_candidates(
    ranked,
    dense_anchor,
    section_index,
    structural_section_index=None,
):
    """Return all known parent siblings for a promoted manual section."""
    if dense_anchor is None:
        return []
    anchor_key = _document_section_key(dense_anchor)
    candidates = []
    seen = set()
    for _, document in ranked:
        if _document_section_key(document) != anchor_key:
            continue
        document_id = hash(document.page_content)
        if document_id not in seen:
            candidates.append(document)
            seen.add(document_id)
    for document in section_index.get(anchor_key, []):
        document_id = hash(document.page_content)
        if document_id not in seen:
            candidates.append(document)
            seen.add(document_id)
    structural_key = _document_structural_section_key(dense_anchor)
    if structural_key is not None:
        for document in (structural_section_index or {}).get(structural_key, []):
            document_id = hash(document.page_content)
            if document_id not in seen:
                candidates.append(document)
                seen.add(document_id)
    return candidates


def select_expanded_section_context(ranked, max_documents=2):
    """Prefer an equivalent localized detail over a neighboring subsection."""
    if not ranked or max_documents <= 0:
        return []
    selected = [ranked[0]]
    top_detail_key = _document_structural_section_key(
        ranked[0][1],
        include_detail=True,
    )
    top_has_detail = top_detail_key is not None and top_detail_key[-1] is not None
    if top_has_detail:
        equivalent = next(
            (
                item for item in ranked[1:]
                if _document_structural_section_key(
                    item[1],
                    include_detail=True,
                ) == top_detail_key
            ),
            None,
        )
        if equivalent is not None:
            selected.append(equivalent)
        return selected[:max_documents]
    if len(selected) < max_documents:
        selected.extend(
            item for item in ranked[1:]
            if item not in selected
        )
    return selected[:max_documents]


def select_reranked_context(ranked, base_top_k=2):
    """Select relevant context adaptively instead of blindly filling top-k."""
    if not ranked:
        return []
    eligible = [ranked[0]]
    eligible.extend(
        item for item in ranked[1:]
        if float(item[0]) > CONTEXT_DOCUMENT_MIN_SCORE
    )
    selected = list(eligible[:base_top_k])
    if len(selected) < base_top_k:
        return selected
    top_keys = [_document_section_key(document) for _, document in selected]
    if len(set(top_keys)) != 1:
        return selected
    supplement = next(
        (
            item for item in eligible[base_top_k:]
            if _document_section_key(item[1]) != top_keys[0]
        ),
        None,
    )
    if supplement is not None:
        selected.append(supplement)
    return selected


def select_anchor_section_context(ranked, dense_anchor, max_documents=2):
    """Keep promoted evidence within the anchor's section to prevent topic mixing."""
    if dense_anchor is None:
        return []
    anchor_key = _document_section_key(dense_anchor)
    return [
        item for item in ranked
        if _document_section_key(item[1]) == anchor_key
    ][:max_documents]


def promote_dense_anchor(
    ranked,
    dense_anchor,
    retrieval_sources,
    top_title_score,
    anchor_title_score,
):
    """Resolve strong dense-vs-reranker disagreement without an extra LLM call."""
    if not ranked or dense_anchor is None:
        return ranked, False
    top_score, top_document = ranked[0]
    top_id = hash(top_document.page_content)
    anchor_id = hash(dense_anchor.page_content)
    if top_id == anchor_id:
        return ranked, False

    top_sources = retrieval_sources.get(top_id, {})
    top_has_retrieval_support = any(
        isinstance(top_sources.get(source), int)
        and top_sources[source] <= DENSE_ANCHOR_MAX_SUPPORT_RANK
        for source in ("vector", "bm25")
    )
    anchor_item = next(
        (
            item for item in ranked
            if hash(item[1].page_content) == anchor_id
        ),
        None,
    )
    if top_has_retrieval_support or anchor_item is None:
        return ranked, False
    if float(anchor_item[0]) <= DENSE_ANCHOR_MIN_CONTENT_SCORE:
        return ranked, False
    if float(anchor_title_score) - float(top_title_score) < DENSE_ANCHOR_MIN_TITLE_MARGIN:
        return ranked, False

    promoted = [anchor_item]
    promoted.extend(item for item in ranked if item is not anchor_item)
    return promoted, True


def _document_title(document):
    metadata = getattr(document, "metadata", {}) or {}
    return " | ".join(filter(None, [
        metadata.get("Ana_Baslik", ""),
        metadata.get("Alt_Baslik", ""),
        metadata.get("Detay_Baslik", ""),
        metadata.get("Mikro_Baslik", ""),
    ]))


def _document_language(document):
    metadata = getattr(document, "metadata", {}) or {}
    language = str(metadata.get("Language", "")).casefold()
    return language if language in {"tr", "en"} else None


def select_dense_anchor(documents, query_language=None):
    """Choose an answer-bearing numbered section, never cover/front matter."""
    structured = [
        document for document in documents or []
        if _document_structural_section_key(document) is not None
    ]
    if query_language:
        preferred = [
            document for document in structured
            if _document_language(document) == query_language
        ]
        if preferred:
            return preferred[0]
    return structured[0] if structured else None


def prefer_ranked_language(ranked, query_language, minimum_score=-4.0):
    """Prefer same-language evidence only when it is sufficiently relevant."""
    if not query_language:
        return ranked, False
    preferred = [
        item for item in ranked
        if _document_language(item[1]) == query_language
    ]
    if preferred and float(preferred[0][0]) > minimum_score:
        return preferred, True
    return ranked, False


# ===================================================== 
# PROMPTLAR 
# ===================================================== 

SYSTEM_PROMPT = """Sen 1PAVI kullanım kılavuzu asistanısın.

Kurallar:
- 1PAVI hakkındaki her iddia, sayı, alan, rota ve işlem adımı yalnız İÇ KANITTA açıkça bulunuyorsa söylenebilir. Tahmin etme.
- Kısaltmaları yalnız İÇ KANITTA açıkça verilen biçimde kullan; kaynakta bulunmayan açılım üretme.
- Kaynakta açıkça geçmeyen protokol, aracı bileşen, veri alanı, tablo adı veya iletişim yönü ekleme.
- İÇ KANIT yalnız mantıksal iş akışını anlatıyorsa fiziksel haberleşme protokolünü bildiğini varsayma.
- Zorunluluk sorularında zorunlu/required ve isteğe bağlı/optional alanları ayrı belirt; alternatifleri birlikte zorunluymuş gibi sunma.
- İÇ KANITTA açıkça bulunmayan garanti, başarı veya sonuç cümlesi ekleme.
- Konuşma geçmişi yalnız söylem bağlamıdır; 1PAVI gerçeği için kanıt değildir.
- İÇ KANIT ve aşağıdaki işaretler özel protokoldür; kullanıcıya gösterme veya açıklama.

Kullanıcının dilinde, doğrudan ve 140 kelimeden kısa yanıt ver. Kaynak yazma.
Yanıtın sonuna mutlaka tam olarak bir işaret ekle:
[[BASIS:MANUAL]], [[BASIS:CONVERSATION]] veya [[BASIS:NONE]]."""

SOCIAL_SYSTEM_PROMPT = """Kısa ve doğal bir sosyal sohbet asistanısın.
Yalnız kullanıcının selamlaşma, tanışma, hal-hatır, teşekkür veya veda mesajına karşılık ver.
Konuşma geçmişinde kullanıcı kendisi hakkında açıkça bilgi verdiyse ilgili soruda bu bilgiyi kullan; tahmin etme.
Kullanıcı önceki cevabı kısaltma, özetleme veya biçimlendirme isterse yalnız o cevabı dönüştür; yeni bilgi ekleme.
Ürün, kılavuz, kanıt veya iç sistemlerden söz etme. Bilgi sorusu yanıtlama.
Kullanıcının dilinde ve iki cümleden kısa yaz.
Yanıtın sonuna tam olarak [[BASIS:CONVERSATION]] ekle; bu işareti açıklama."""

FOLLOWUP_REWRITE_PROMPT = """GÜNCEL KULLANICI MESAJINDAKİ referansı doküman araması için çöz.

Kurallar:
- Güncel mesajın, sınırlı konuşma kesitine bağımlı olup olmadığına karar ver.
- Bağımsızsa eski kullanıcı turu seçme.
- Bağımlıysa yalnızca referans verilen konuyu içeren kullanıcı turunu/turlarını seç.
- Referans her zaman en son tura gitmez. "İlki", "iki mesaj önce", "önce konuştuğumuz"
  gibi sıra ve zaman ifadelerini dikkatle çöz.
- En fazla iki kullanıcı turu seç. Asistan turunu kanıt olarak seçme.
- Soruyu yanıtlama, yeniden yazma ve yeni bilgi ekleme.

Yalnızca JSON döndür:
{"depends_on_history": true, "referenced_user_turns": [2]}"""

# Bunlar ürün/soru ezberi değil, Türkçedeki genel söylem bağımlılığı işaretleridir.
# Amaç bağımsız teknik sorular için ek LLM çağrısını atlamaktır.
FOLLOWUP_REFERENCE_PATTERN = re.compile(
    r"\b(?:bu|bunu|buna|bunda|bundan|bunun|bunlar|bunları|bunların|"
    r"şu|şunu|şuna|şunda|şundan|şunun|şunlar|şunları|şunların|"
    r"o|onu|ona|onda|ondan|onun|onlar|onları|onların|"
    r"hangisi|hangileri|aynısı|önceki|yukarıdaki|bahsettiğin|söylediğin|"
    r"ilki|ilkinin|ilkine|ikincisi|ikincisinin)\b",
    re.IGNORECASE,
)
FOLLOWUP_CONTINUATION_PATTERN = re.compile(
    r"^\s*(?:devam(?:ında)?|daha(?:\s+da)?|biraz(?:\s+daha)?)\b",
    re.IGNORECASE,
)
FOLLOWUP_TRANSFORM_PATTERN = re.compile(
    r"\b(?:tablo(?:\s+halinde)?|özetle|özetler|karşılaştır|detaylandır|açıkla|"
    r"listele|maddele|göster|örnekle|devam et|kısalt|daha\s+kısa|uzat|yeniden\s+yaz)\b",
    re.IGNORECASE,
)
FOLLOWUP_FIRST_TURN_PATTERN = re.compile(
    r"\b(?:ilki|ilkinin|ilkine|ilk sorduğum|ilk sorum|ilk konu)\b", re.IGNORECASE
)
FOLLOWUP_SECOND_TURN_PATTERN = re.compile(
    r"\b(?:ikincisi|ikincisinin|ikinci sorum|ikinci konu)\b", re.IGNORECASE
)
FOLLOWUP_N_TURNS_BACK_PATTERN = re.compile(
    r"\b(iki|İki|2|üç|Üç|3)\s+(?:mesaj|soru|konu|tur)\s+önce(?:ki)?\b", re.IGNORECASE
)
ELLIPTICAL_GENERIC_PLURAL_PATTERN = re.compile(
    r"^\s*hangi\s+(?:türler(?:i)?|tipler(?:i)?|çeşitler(?:i)?|seçenekler(?:i)?)\b",
    re.IGNORECASE,
)
# Kısa adjacency-pair yanıtları ve öznesi konuşmadan düşürülmüş durum
# bildirimleri açık zamir taşımayabilir. Bunlar yalnız resolver adayıdır;
# geçmişe bağımlılık kararını bounded resolver verir.
FOLLOWUP_RESPONSE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:evet|hayır|tamam|yes|no|okay|ok)\b",
    re.IGNORECASE,
)
ELLIPTICAL_STATUS_PATTERN = re.compile(
    r"\b(?:hata\s+(?:alıyorum|aldım|veriyor)|çalışmıyor|olmadı|düzelmedi|"
    r"değişmedi|başarısız|hala\s+aynı|hâlâ\s+aynı|denedim|"
    r"not\s+working|didn['’]?t\s+work|still\s+fails?|still\s+the\s+same|"
    r"getting\s+an?\s+error|failed)\b",
    re.IGNORECASE,
)
SOCIAL_MESSAGE_PATTERN = re.compile(
    r"(?:^|\b)(?:merhaba|selam(?:lar)?|günaydın|iyi\s+(?:günler|akşamlar|geceler)|"
    r"nasılsın(?:ız)?|nasıl\s+gidiyor|ne\s+haber|naber|"
    r"teşekkür(?:ler|\s+ederim)?|sağ\s*ol(?:un)?|eyvallah|"
    r"görüşürüz|hoşça\s+kal(?:ın)?|kendine\s+iyi\s+bak|"
    r"(?:benim\s+)?adım\s+ne(?:ydi)?)(?:\b|$)",
    re.IGNORECASE,
)
CAPABILITY_QUERY_PATTERN = re.compile(
    r"(?:\b(?:bana\s+)?(?:ne|neler|hangi\s+konular)\s+hakkında\s+"
    r"yardımcı\s+olabilirsin\b|\bneler\s+yapabilirsin\b|"
    r"\bhow\s+can\s+you\s+help\b|\bwhat\s+can\s+you\s+help\s+with\b)",
    re.IGNORECASE,
)
NAME_RECALL_PATTERN = re.compile(r"\b(?:benim\s+)?adım\s+ne(?:ydi)?\b", re.IGNORECASE)
SOCIAL_GREETING_PATTERN = re.compile(
    r"\b(?:merhaba|selam(?:lar)?|günaydın|iyi\s+(?:günler|akşamlar|geceler))\b",
    re.IGNORECASE,
)
SOCIAL_WELLBEING_PATTERN = re.compile(
    r"\b(?:nasılsın(?:ız)?|nasıl\s+gidiyor|ne\s+haber|naber)\b", re.IGNORECASE
)
SOCIAL_THANKS_PATTERN = re.compile(
    r"\b(?:teşekkür(?:ler|\s+ederim)?|sağ\s*ol(?:un)?|eyvallah)\b", re.IGNORECASE
)
SOCIAL_FAREWELL_PATTERN = re.compile(
    r"\b(?:görüşürüz|hoşça\s+kal(?:ın)?|kendine\s+iyi\s+bak)\b", re.IGNORECASE
)
DECLARED_NAME_PATTERNS = (
    re.compile(
        r"\b(?:benim\s+adım|adım)\s+(?!ne(?:ydi)?\b)([^\W\d_]{2,30})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:(?:merhaba|selam)[,! ]+)?ben\s+([^\W\d_]{2,30})\s*[.!]?\s*$",
        re.IGNORECASE,
    ),
)


def is_social_message(message):
    """Yalnız kısa sosyal konuşma eylemlerini yönlendirir; cevap üretmez."""
    compact = " ".join(str(message or "").split())
    declared_name = declared_user_name([{"role": "user", "content": compact}])
    return len(compact.split()) <= 12 and bool(
        SOCIAL_MESSAGE_PATTERN.search(compact)
        or CAPABILITY_QUERY_PATTERN.search(compact)
        or declared_name
    )


def declared_user_name(history):
    """Kullanıcının açık tanışma ifadesinden oturum adını çıkarır; tahmin etmez."""
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        content = " ".join(str(message.get("content", "")).split())
        for pattern in DECLARED_NAME_PATTERNS:
            match = pattern.search(content)
            if match:
                return match.group(1).strip().title()
    return None


def fast_social_response(message, history):
    """Handle finite, non-informational speech acts without retrieval or an LLM."""
    compact = " ".join(str(message or "").split())
    if not is_social_message(compact):
        return None

    remembered_name = declared_user_name(history)
    current_name = declared_user_name([{"role": "user", "content": compact}])
    name = current_name or remembered_name

    if CAPABILITY_QUERY_PATTERN.search(compact):
        return (
            "1PAVI kullanım kılavuzu kapsamındaki ürün kullanımı, kurulum, "
            "yapılandırma, çalıştırma ve hata giderme konularında yardımcı olabilirim."
        )
    if NAME_RECALL_PATTERN.search(compact):
        return f"Adınız {name}." if name else "Adınızı henüz bilmiyorum."
    if SOCIAL_FAREWELL_PATTERN.search(compact):
        return "Görüşmek üzere."
    if SOCIAL_THANKS_PATTERN.search(compact):
        return "Rica ederim. Başka nasıl yardımcı olabilirim?"
    if SOCIAL_WELLBEING_PATTERN.search(compact):
        address = f" {name}" if name else ""
        return f"İyiyim, teşekkür ederim{address}. Size nasıl yardımcı olabilirim?"
    if SOCIAL_GREETING_PATTERN.search(compact) or current_name:
        address = f" {name}" if name else ""
        return f"Merhaba{address}, nasıl yardımcı olabilirim?"
    return None


def unsupported_technical_terms(answer, context, user_input=""):
    """Expose novel acronym/identifier tokens for faithfulness diagnostics."""
    candidate_pattern = re.compile(
        r"(?<!\w)(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9_]{1,}(?:-[A-Za-z0-9_]+)*|`[^`]+`)(?!\w)"
    )
    allowed_text = f"{context}\n{user_input}".casefold()
    ignored = {"manual", "conversation", "none", "basis", "source"}
    unsupported = set()
    for token in candidate_pattern.findall(str(answer or "")):
        normalized = token.strip("`").casefold()
        if normalized not in ignored and normalized not in allowed_text:
            unsupported.add(token.strip("`"))
    return sorted(unsupported, key=str.casefold)


def last_answer_basis(history):
    """Son asistan turunun mimari tarafından atanmış kanıt türünü okur."""
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        metadata = message.get("metadata", {})
        basis = str(metadata.get("answer_basis", "")).upper()
        if basis:
            return basis
        debug_intent = str(metadata.get("debug_info", {}).get("intent", "")).upper()
        if debug_intent == "SOHBET":
            return "CONVERSATION"
        return None
    return None

class TensorRTCrossEncoder:
    def __init__(self, model_name, onnx_path):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # SİHİRLİ SATIR: TensorrtExecutionProvider ile GPU'yu tetikliyoruz
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_fp16_enable": True,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
        
        print(">>> TensorRT Cross-Encoder Yükleniyor...")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
    def predict(self, pairs, show_progress_bar=False):
        # MİMARİ DÜZELTME: TensorRT çöpmesin diye padding="max_length" yapıyoruz
        inputs = self.tokenizer(pairs, padding="max_length", truncation=True, max_length=512, return_tensors="np")
        
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        logits = self.session.run(None, ort_inputs)[0]
        return logits.flatten()

# ===================================================== 
# PIPELINE 
# ===================================================== 

class RAGPipeline: 
    def __init__(
        self,
        model_name="llama3.1",
        generation_num_ctx=3072,
        max_answer_tokens=None,
        warmup_num_ctx=None,
    ): 
        print(f">>> Modeller yükleniyor (Seçilen Model: {model_name})") 
        self.model_name = model_name 
        self.rewrite_model = None 
        self.think_mode = "low" if model_name.casefold().startswith("gpt-oss") else False
        default_answer_tokens = 512 if self.think_mode == "low" else 256
        self.generation_num_ctx = int(generation_num_ctx)
        self.warmup_num_ctx = int(
            self.generation_num_ctx if warmup_num_ctx is None else warmup_num_ctx
        )
        self.max_answer_tokens = int(
            default_answer_tokens if max_answer_tokens is None else max_answer_tokens
        )
        if self.generation_num_ctx < 512 or self.warmup_num_ctx < 512:
            raise ValueError("num_ctx en az 512 olmalıdır")
        if self.max_answer_tokens < 1:
            raise ValueError("max_answer_tokens pozitif olmalıdır")
        
        # Langchain yerine saf Ollama client'ı başlatıyoruz 
        self.client = ollama.Client(host="http://localhost:11434") 
        self.embeddings = OllamaEmbeddings( 
            model="bge-m3:latest", 
            base_url="http://localhost:11434" 
        ) 

        self.vectorstore = Chroma( 
            persist_directory=os.path.join(APP_DIR, "chroma_db"), 
            embedding_function=self.embeddings, 
            collection_metadata={"hnsw:space": "cosine"} 
        ) 

        try: 
            with open(os.path.join(DATA_DIR, "parent_store.pkl"), "rb") as f: 
                store = pickle.load(f) 
        except FileNotFoundError: 
            store = InMemoryStore() 

        self.chroma_retriever = ParentDocumentRetriever( 
            vectorstore=self.vectorstore, 
            docstore=store, 
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50), 
            parent_splitter=None, 
            search_kwargs={"k": 10} 
        ) 

        try: 
            with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f: 
                self.bm25_retriever = pickle.load(f) 
            self.bm25_retriever.k = 10 
        except FileNotFoundError: 
            self.bm25_retriever = self.chroma_retriever 

        self.index_validation = validate_index_manifest(store, self.bm25_retriever)
        store_values = getattr(store, "store", {})
        self.section_documents = build_section_document_index(store_values.values())
        self.structural_section_documents = build_structural_section_index(
            store_values.values()
        )

        # SEÇENEK 1: Standart PyTorch/CPU versiyonu (Şu an kullandığın)
        self.cross_encoder = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            max_length=512,
            device="cuda",
            local_files_only=True,
        )
        
        # SEÇENEK 2: TensorRT/GPU versiyonu (ONNX dosyası gerektirir)
        #self.cross_encoder = TensorRTCrossEncoder(
        #    model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
        #    onnx_path="cross_encoder.onnx"
        #)
        
        # --------------------------------------------------------- 

        self.pipeline_config = { 
            "architecture": "Jetson RAG (Trusted Section State + Hierarchical Expansion)",
            "rewrite_model": f"{self.model_name} (conditional)",
            "history_user_turns": 3,
            "has_heuristic_bypass": True,
            "routing": "social speech acts + fail-closed evidence gate",
            "chunk_size": 400, 
            "chunk_overlap": 50, 
            "vector_search_k": 10,
            "rerank_top_k": 2,
            "diversity_supplement_max_k": 3,
            "section_sibling_expansion": "conditional_after_dense_anchor_promotion",
            "localized_section_alignment": "numeric_manual_hierarchy",
            "followup_state": "reliable_used_section_headings_only",
            "language_routing": "same_language_first_with_cross_language_fallback",
            "max_answer_tokens": self.max_answer_tokens,
            "generation_num_ctx": self.generation_num_ctx,
            "condense_num_ctx": self.generation_num_ctx,
            "warmup_num_ctx": self.warmup_num_ctx,
            "think_mode": self.think_mode,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "evidence_rescue_min_score": EVIDENCE_RESCUE_MIN_SCORE,
            "evidence_rescue_min_margin": EVIDENCE_RESCUE_MIN_MARGIN,
            "evidence_consensus_max_rank": EVIDENCE_CONSENSUS_MAX_RANK,
            "dense_anchor_max_support_rank": DENSE_ANCHOR_MAX_SUPPORT_RANK,
            "dense_anchor_min_content_score": DENSE_ANCHOR_MIN_CONTENT_SCORE,
            "dense_anchor_min_title_margin": DENSE_ANCHOR_MIN_TITLE_MARGIN,
            "dense_anchor_rescue_min_score": DENSE_ANCHOR_RESCUE_MIN_SCORE,
            "index_validation": self.index_validation,
        } 
    def warmup(self):
        print("\n[INFO] Sistem Isınıyor: Modeller GPU'ya yükleniyor (Cold Start)...")
        start_time = time.perf_counter()
        
        # 1. Embedding Modelini Uyandır
        self.embeddings.embed_query("hello")
        
        # 2. CrossEncoder'ı (Reranker) Uyandır
        self.cross_encoder.predict([["hello", "world"]], show_progress_bar=False)
        
        # Jetsonda ikinci LLMyi aynı anda VRAMde tutmak model değişimine yol açar.
        try:
            self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                think=getattr(self, "think_mode", False),
                options={"num_predict": 1, "num_ctx": self.warmup_num_ctx},
                keep_alive="30m",
            )
        except Exception as e:
            pass
            
        elapsed = round(time.perf_counter() - start_time, 2)
        print(f"[INFO] Warmup Tamamlandı! ({elapsed} sn) - Bütün modeller VRAM'de hazır.\n")

    @staticmethod
    def _needs_followup_resolution(question, history):
        """Genel dil işaretleriyle bağlama bağımlı olabilecek mesajları seçer."""
        if not history:
            return False
        compact = " ".join(str(question or "").split())
        if not compact:
            return False
        if FOLLOWUP_REFERENCE_PATTERN.search(compact):
            return True
        if FOLLOWUP_CONTINUATION_PATTERN.search(compact):
            return True
        if ELLIPTICAL_GENERIC_PLURAL_PATTERN.search(compact):
            return True
        if len(compact.split()) <= 12 and (
            FOLLOWUP_RESPONSE_PREFIX_PATTERN.search(compact)
            or ELLIPTICAL_STATUS_PATTERN.search(compact)
        ):
            return True
        # Kısa biçim-dönüşümü isteklerinde özne çoğunlukla önceki turdadır.
        return len(compact.split()) <= 10 and bool(FOLLOWUP_TRANSFORM_PATTERN.search(compact))

    @staticmethod
    def _bounded_dialogue(history, max_user_turns=3):
        """Son N kullanıcı turunu ve aradaki yanıtları ham içerikten döndürür."""
        valid = []
        for message in history:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            valid.append({
                "role": role,
                "content": " ".join(str(message.get("content", "")).split())[:500],
            })
        user_indexes = [index for index, item in enumerate(valid) if item["role"] == "user"]
        if not user_indexes:
            return []
        first_user_position = user_indexes[max(0, len(user_indexes) - max_user_turns)]
        selected = valid[first_user_position:]
        user_turn = 0
        for item in selected:
            if item["role"] == "user":
                user_turn += 1
                item["user_turn"] = user_turn
        return selected

    @staticmethod
    def _explicit_history_reference(question, dialogue):
        """Açık sıra referanslarını LLM'e bırakmadan çözer."""
        users = [item for item in dialogue if item["role"] == "user"]
        if not users:
            return []
        if FOLLOWUP_FIRST_TURN_PATTERN.search(question):
            return [users[0]["content"]]
        if FOLLOWUP_SECOND_TURN_PATTERN.search(question) and len(users) >= 2:
            return [users[1]["content"]]
        if ELLIPTICAL_GENERIC_PLURAL_PATTERN.search(question):
            return [users[-1]["content"]]
        turns_back = FOLLOWUP_N_TURNS_BACK_PATTERN.search(question)
        if turns_back:
            number_word = turns_back.group(1).casefold().replace("\u0307", "")
            distance = 2 if number_word in {"iki", "2"} else 3
            if len(users) >= distance:
                return [users[-distance]["content"]]
        return []

    @staticmethod
    def _latest_reliable_section_hint(history):
        """Read only trusted source-heading state; never reuse generated answers."""
        for message in reversed(history or []):
            if message.get("role") != "assistant":
                continue
            metadata = message.get("metadata", {}) or {}
            if str(metadata.get("answer_basis", "")).upper() != "MANUAL":
                continue
            debug_info = metadata.get("debug_info", {}) or {}
            if debug_info.get("context_is_reliable") is False:
                continue
            sections = metadata.get("used_sections") or debug_info.get(
                "retrieved_sections", []
            )
            if not isinstance(sections, list):
                continue
            for section in sections:
                compact = " ".join(str(section or "").split())[:160]
                if compact:
                    return compact
        return None

    def condense_question(self, question, history):
        """Gerekirse sınırlı ham geçmişten bağımsız bir retrieval sorgusu üretir."""
        rewrite_start = time.perf_counter()
        question = " ".join(str(question or "").split())
        if not self._needs_followup_resolution(question, history):
            return question, question, 0.0

        dialogue = self._bounded_dialogue(history, max_user_turns=3)
        if not dialogue:
            return question, question, 0.0

        explicit_reference = self._explicit_history_reference(question, dialogue)
        if explicit_reference:
            section_hint = None
            if ELLIPTICAL_GENERIC_PLURAL_PATTERN.search(question):
                section_hint = self._latest_reliable_section_hint(history)
            contextual = (
                f"{question} {section_hint}"
                if section_hint
                else " ".join([question, *explicit_reference])
            )
            return question, contextual, round(time.perf_counter() - rewrite_start, 3)

        excerpt_lines = []
        for item in dialogue:
            label = f"USER_TURN_{item['user_turn']}" if item["role"] == "user" else "ASSISTANT"
            excerpt_lines.append(f"{label}: {item['content']}")
        excerpt = "\n".join(excerpt_lines)
        resolver_input = (
            f"BOUNDED CONVERSATION:\n{excerpt}\n\n"
            f"CURRENT USER MESSAGE:\n{question}"
        )
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": FOLLOWUP_REWRITE_PROMPT},
                    {"role": "user", "content": resolver_input},
                ],
                format="json",
                think=getattr(self, "think_mode", False),
                options={
                    "temperature": 0.0,
                    # The same Ollama runner serves rewrite and generation.
                    # Changing num_ctx forces a costly runner reload on Jetson;
                    # the bounded resolver prompt does not need a smaller runner.
                    "num_ctx": self.generation_num_ctx,
                    "num_predict": 192 if getattr(self, "think_mode", False) == "low" else 96,
                },
                keep_alive="30m",
            )
            payload = json.loads(response["message"]["content"])
            depends_on_history = payload.get("depends_on_history") is True
            if not depends_on_history:
                return question, question, round(time.perf_counter() - rewrite_start, 3)
            raw_turns = payload.get("referenced_user_turns", [])
            if not isinstance(raw_turns, list):
                raise ValueError("Geçersiz follow-up turn listesi")
            selected_turns = []
            for value in raw_turns[:2]:
                match = re.search(r"\d+", str(value))
                if match:
                    selected_turns.append(int(match.group()))
            referenced = [
                item["content"] for item in dialogue
                if item["role"] == "user" and item.get("user_turn") in selected_turns
            ]
            if not referenced:
                raise ValueError("Resolver bir kullanıcı turu seçmedi")
            # Güncel niyet başta kalır; tokenizer kırpmasında eski bağlam öncelik kazanmaz.
            rewritten = " ".join([question, *referenced])
        except Exception:
            # Güvenli ve sabit boyutlu geri dönüş; metadata.query asla kullanılmaz.
            previous_users = [item["content"] for item in dialogue if item["role"] == "user"]
            rewritten = f"{question} {previous_users[-1]}".strip() if previous_users else question

        rewrite_time = round(time.perf_counter() - rewrite_start, 3)
        return question, rewritten, rewrite_time

    def retrieve_context(self, query, contextual_query=None):
        retrieval_start = time.perf_counter()
        queries = [query]
        if contextual_query and contextual_query != query:
            queries.append(contextual_query)
        query_language = detect_text_language(queries[-1])

        vector_rankings = [self.chroma_retriever.invoke(item) for item in queries]
        bm25_rankings = [self.bm25_retriever.invoke(item) for item in queries]
        retrieval_time = round(time.perf_counter() - retrieval_start, 2)

        candidate_map = {}
        fused_scores = {}
        retrieval_sources = {}
        named_rankings = [
            *(('vector', ranking) for ranking in vector_rankings),
            *(('bm25', ranking) for ranking in bm25_rankings),
        ]
        for source, ranking in named_rankings:
            for rank, doc in enumerate(ranking, 1):
                doc_id = hash(doc.page_content)
                candidate_map[doc_id] = doc
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rank - 1 + 60)
                source_ranks = retrieval_sources.setdefault(doc_id, {})
                previous_rank = source_ranks.get(source)
                source_ranks[source] = min(previous_rank, rank) if previous_rank else rank

        candidates = [
            candidate_map[doc_id]
            for doc_id, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
        dense_anchor = select_dense_anchor(
            vector_rankings[-1] if vector_rankings else [],
            query_language,
        )
        if dense_anchor is not None:
            anchor_key = _document_section_key(dense_anchor)
            anchor_section_documents = [
                document for document in vector_rankings[-1]
                if _document_section_key(document) == anchor_key
            ][:2]
            for anchor_document in anchor_section_documents:
                if all(
                    hash(document.page_content) != hash(anchor_document.page_content)
                    for document in candidates
                ):
                    candidates.append(anchor_document)
        if not candidates:
            return "", [], retrieval_time, 0.0, -999.0, []

        rerank_start = time.perf_counter()
        pairs = [[item, doc.page_content] for item in queries for doc in candidates]
        raw_scores = np.asarray(
            self.cross_encoder.predict(pairs, show_progress_bar=False), dtype=float
        ).reshape(len(queries), len(candidates))
        scores = raw_scores.max(axis=0)
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        best_ce_score = round(float(ranked[0][0]), 3)
        dense_anchor_promoted = False
        section_expansion_used = False
        language_preference_used = False
        if dense_anchor is not None and ranked:
            top_document = ranked[0][1]
            top_sources = retrieval_sources.get(hash(top_document.page_content), {})
            top_has_retrieval_support = any(
                isinstance(top_sources.get(source), int)
                and top_sources[source] <= DENSE_ANCHOR_MAX_SUPPORT_RANK
                for source in ("vector", "bm25")
            )
            anchor_item = next(
                (
                    item for item in ranked
                    if hash(item[1].page_content) == hash(dense_anchor.page_content)
                ),
                None,
            )
            if (
                not top_has_retrieval_support
                and anchor_item is not None
                and float(anchor_item[0]) > DENSE_ANCHOR_MIN_CONTENT_SCORE
                and hash(top_document.page_content) != hash(dense_anchor.page_content)
            ):
                title_scores = self.cross_encoder.predict(
                    [
                        [queries[-1], _document_title(top_document)],
                        [queries[-1], _document_title(dense_anchor)],
                    ],
                    show_progress_bar=False,
                )
                ranked, dense_anchor_promoted = promote_dense_anchor(
                    ranked,
                    dense_anchor,
                    retrieval_sources,
                    title_scores[0],
                    title_scores[1],
                )
        if dense_anchor_promoted:
            section_candidates = expand_anchor_section_candidates(
                ranked,
                dense_anchor,
                self.section_documents,
                self.structural_section_documents,
            )
            existing_section_count = len(
                select_anchor_section_context(ranked, dense_anchor, max_documents=99)
            )
            section_expansion_used = len(section_candidates) > existing_section_count
            section_pairs = [
                [item, document.page_content]
                for item in queries
                for document in section_candidates
            ]
            section_raw_scores = np.asarray(
                self.cross_encoder.predict(section_pairs, show_progress_bar=False),
                dtype=float,
            ).reshape(len(queries), len(section_candidates))
            section_scores = section_raw_scores.max(axis=0)
            expanded_ranked = sorted(
                zip(section_scores, section_candidates),
                key=lambda item: item[0],
                reverse=True,
            )
            expanded_ranked, language_preference_used = prefer_ranked_language(
                expanded_ranked,
                query_language,
            )
            final_ranked = select_expanded_section_context(
                expanded_ranked,
                max_documents=2,
            )
        else:
            language_ranked, language_preference_used = prefer_ranked_language(
                ranked,
                query_language,
            )
            final_ranked = select_reranked_context(language_ranked, base_top_k=2)
        final_docs = [doc for score, doc in final_ranked]
        rerank_time = round(time.perf_counter() - rerank_start, 2)

        sections = []
        context_parts = []
        for index, doc in enumerate(final_docs, 1):
            main = doc.metadata.get("Ana_Baslik", "")
            sub = doc.metadata.get("Alt_Baslik", "")
            detail = doc.metadata.get("Detay_Baslik", "")
            micro = doc.metadata.get("Mikro_Baslik", "")
            # Kaynak gösteriminde en özgül başlık önce gelir.
            for heading in (micro, detail, sub, main):
                if heading and heading not in sections:
                    sections.append(heading)
            header = " | ".join(filter(None, [main, sub, detail, micro]))
            evidence_content, compressed = compress_matching_table_rows(
                queries[-1],
                doc.page_content,
            )
            context_parts.append(f"[Source {index} | {header}]\n{evidence_content}")

        retrieved_documents = []
        for rank, (score, doc) in enumerate(final_ranked, 1):
            doc_id = hash(doc.page_content)
            source_ranks = retrieval_sources.get(doc_id, {})
            retrieved_documents.append({
                "rank": rank,
                "reranker_score": round(float(score), 4),
                "hybrid_score": round(float(fused_scores.get(doc_id, 0.0)), 6),
                "vector_rank": source_ranks.get("vector"),
                "bm25_rank": source_ranks.get("bm25"),
                "source_file": doc.metadata.get("Source_File", ""),
                "main_section": doc.metadata.get("Ana_Baslik", ""),
                "section": doc.metadata.get("Alt_Baslik", ""),
                "detail_section": doc.metadata.get("Detay_Baslik", ""),
                "language": _document_language(doc),
                "content": doc.page_content,
                "dense_anchor_promoted": dense_anchor_promoted and rank == 1,
                "section_expansion_used": section_expansion_used,
                "language_preference_used": language_preference_used,
                "evidence_compressed": compressed,
            })

        return "\n\n---\n\n".join(context_parts), sections, retrieval_time, rerank_time, best_ce_score, retrieved_documents

    def answer_question(self, user_input, history_messages=None, stream_callback=None):
        history_messages = history_messages or []
        total_start = time.perf_counter()

        social_answer = fast_social_response(user_input, history_messages)
        if social_answer is not None:
            if stream_callback:
                stream_callback(social_answer)
            standalone_query = " ".join(str(user_input or "").split())
            fast_path_reason = (
                "capability_fast_path"
                if CAPABILITY_QUERY_PATTERN.search(standalone_query)
                else "social_fast_path"
            )
            return {
                "answer": social_answer,
                "intent": "SOHBET",
                "answer_basis": "CONVERSATION",
                "evidence_marker_compliance": None,
                "empty_model_output": False,
                "search_query": standalone_query,
                "evidence_query": standalone_query,
                "standalone_question": standalone_query,
                "query_language": detect_text_language(standalone_query),
                "best_distance": None,
                "avg_distance": None,
                "threshold": GROUNDING_THRESHOLD,
                "context_is_reliable": False,
                "context_reliability_reason": fast_path_reason,
                "reranker_score_margin": None,
                "distinctive_term_count": 0,
                "distinctive_term_hits": 0,
                "top_distinctive_term_hits": 0,
                "retrieval_consensus": False,
                "consensus_support_rank": None,
                "context": "",
                "retrieved_sections": [],
                "used_sections": [],
                "candidate_sections": [],
                "retrieved_documents": [],
                "retrieved_document_count": 0,
                "diversity_supplement_used": False,
                "dense_anchor_promoted": False,
                "section_expansion_used": False,
                "language_preference_used": False,
                "evidence_compression_used": False,
                "best_chat_score": 0.0,
                "rewrite_time": 0.0,
                "retrieval_time": 0.0,
                "rerank_time": 0.0,
                "generation_time": 0.0,
                "time_to_first_token": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": 0.0,
                "ollama_load_time": 0.0,
                "total_time": round(time.perf_counter() - total_start, 3),
                "prompt": "",
            }

        conversation_followup = (
            self._needs_followup_resolution(user_input, history_messages)
            and last_answer_basis(history_messages) == "CONVERSATION"
        )
        if conversation_followup:
            standalone_query = " ".join(str(user_input or "").split())
            contextual_query = standalone_query
            rewrite_time = 0.0
        else:
            standalone_query, contextual_query, rewrite_time = self.condense_question(
                user_input, history_messages
            )
        context, sections, retrieval_time, rerank_time, best_ce_score, retrieved_documents = self.retrieve_context(
            standalone_query, contextual_query
        )

        grounding_threshold = GROUNDING_THRESHOLD
        evidence_query = contextual_query or standalone_query
        query_language = detect_text_language(evidence_query)
        reliability = assess_context_reliability(
            evidence_query, best_ce_score, retrieved_documents
        )
        context_is_reliable = reliability["is_reliable"]
        social_message = is_social_message(user_input) or conversation_followup

        def bypass_generation(answer, basis):
            if basis == "MANUAL" and sections and "Kaynak/Source:" not in answer:
                answer = f"{answer}\n\nKaynak/Source: {sections[0]}"
            if stream_callback:
                stream_callback(answer)
            intent = (
                "SOHBET" if basis == "CONVERSATION"
                else "TEKNİK" if basis == "MANUAL"
                else "KANIT_YETERSİZ"
            )
            return {
                "answer": answer,
                "intent": intent,
                "answer_basis": basis,
                "evidence_marker_compliance": None,
                "empty_model_output": False,
                "search_query": contextual_query,
                "evidence_query": evidence_query,
                "standalone_question": standalone_query,
                "query_language": query_language,
                "best_distance": best_ce_score,
                "avg_distance": best_ce_score,
                "threshold": grounding_threshold,
                "context_is_reliable": context_is_reliable,
                "context_reliability_reason": reliability["reason"],
                "reranker_score_margin": reliability["score_margin"],
                "distinctive_term_count": reliability["distinctive_term_count"],
                "distinctive_term_hits": reliability["distinctive_term_hits"],
                "top_distinctive_term_hits": reliability["top_distinctive_term_hits"],
                "retrieval_consensus": reliability["retrieval_consensus"],
                "consensus_support_rank": reliability["consensus_support_rank"],
                "context": context,
                "retrieved_sections": sections,
                "used_sections": sections if basis == "MANUAL" else [],
                "candidate_sections": sections,
                "retrieved_documents": retrieved_documents,
                "retrieved_document_count": len(retrieved_documents),
                "diversity_supplement_used": len(retrieved_documents) > 2,
                "dense_anchor_promoted": any(
                    document.get("dense_anchor_promoted", False)
                    for document in retrieved_documents
                ),
                "section_expansion_used": any(
                    document.get("section_expansion_used", False)
                    for document in retrieved_documents
                ),
                "language_preference_used": any(
                    document.get("language_preference_used", False)
                    for document in retrieved_documents
                ),
                "evidence_compression_used": any(
                    document.get("evidence_compressed", False)
                    for document in retrieved_documents
                ),
                "best_chat_score": 0.0,
                "rewrite_time": rewrite_time,
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": 0.0,
                "time_to_first_token": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": 0.0,
                "ollama_load_time": 0.0,
                "total_time": round(time.perf_counter() - total_start, 2),
                "prompt": "",
            }

        if NAME_RECALL_PATTERN.search(user_input):
            remembered_name = declared_user_name(history_messages)
            if remembered_name:
                return bypass_generation(f"Adınız {remembered_name}.", "CONVERSATION")

        # Güvenilir ürün kanıtı ve sosyal konuşma sinyali yoksa küçük modele
        # genel bilgi uydurtma. Bu yol aynı zamanda gereksiz generation'ı atlar.
        if not context_is_reliable and not social_message:
            return bypass_generation(OUT_OF_SCOPE_ANSWER, "NONE")

        if context_is_reliable and not social_message:
            extractive_answer = extract_single_table_answer(context)
            if extractive_answer:
                return bypass_generation(extractive_answer, "MANUAL")

        if social_message:
            messages = [{"role": "system", "content": SOCIAL_SYSTEM_PROMPT}]
            generation_history = history_messages[-2:]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.append({
                "role": "system",
                "content": f"İÇ KANIT — güvenilir 1PAVI kılavuz bölümleri:\n{context}",
            })
            # Teknik gerçekler yalnız güncel retrieval kanıtından üretilir.
            # Çözümlenmiş niyet aşağıda ayrıca verilir; eski cevap taşınmaz.
            generation_history = []
        for message in generation_history:
            role = message.get("role")
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": message.get("content", "")[:700]})

        if not social_message and contextual_query != standalone_query:
            messages.append({
                "role": "system",
                "content": f"Çözümlenmiş arama niyeti (kanıt değildir): {contextual_query}",
            })
        messages.append({"role": "user", "content": user_input})

        generation_start = time.perf_counter()
        first_token_time = None
        metric_response = None
        chat_options = {
            "temperature": 0.0,
            "num_ctx": self.generation_num_ctx,
            "num_predict": self.max_answer_tokens,
            "repeat_penalty": 1.1,
            "top_k": 10,
            "top_p": 0.8,
        }
        if stream_callback:
            chunks = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                think=self.think_mode,
                options=chat_options,
                keep_alive="30m",
            )
            generated_parts = []
            for chunk in chunks:
                metric_response = chunk
                token = chunk["message"]["content"]
                if token:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - generation_start
                    generated_parts.append(token)
                    partial = "".join(generated_parts)
                    visible = re.sub(
                        r"\s*\[\[BASIS:.*$", "", partial,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    stream_callback(visible)
            generated_text = "".join(generated_parts)
        else:
            metric_response = self.client.chat(
                model=self.model_name,
                messages=messages,
                think=self.think_mode,
                options=chat_options,
                keep_alive="30m",
                
            )
            generated_text = metric_response["message"]["content"]

        raw_response, answer_basis, marker_compliance = finalize_grounded_answer(
            generated_text.strip(), sections, context_is_reliable
        )
        technical_term_violations = unsupported_technical_terms(
            generated_text, context, user_input
        )
        if raw_response != EMPTY_ANSWER_FALLBACK:
            answer_basis = "CONVERSATION" if social_message else "MANUAL"
        if (
            answer_basis == "MANUAL" and sections
            and "Kaynak/Source:" not in raw_response
        ):
            raw_response = f"{raw_response}\n\nKaynak/Source: {sections[0]}"
        generation_time = round(time.perf_counter() - generation_start, 3)

        def response_metric(name, default=0):
            value = getattr(metric_response, name, None)
            if value is None and hasattr(metric_response, "get"):
                value = metric_response.get(name)
            return default if value is None else value

        output_tokens = int(response_metric("eval_count", 0))
        input_tokens = int(response_metric("prompt_eval_count", 0))
        eval_duration_ns = int(response_metric("eval_duration", 0))
        tokens_per_second = (
            round(output_tokens / (eval_duration_ns / 1e9), 2)
            if output_tokens and eval_duration_ns else 0.0
        )
        intent = "TEKNİK" if answer_basis == "MANUAL" else "SOHBET"

        return {
            "answer": raw_response,
            "intent": intent,
            "answer_basis": answer_basis,
            "evidence_marker_compliance": marker_compliance,
            "empty_model_output": not bool(generated_text.strip()),
            "search_query": contextual_query,
            "evidence_query": evidence_query,
            "standalone_question": standalone_query,
            "query_language": query_language,
            "best_distance": best_ce_score,
            "avg_distance": best_ce_score,
            "threshold": grounding_threshold,
            "context_is_reliable": context_is_reliable,
            "context_reliability_reason": reliability["reason"],
            "reranker_score_margin": reliability["score_margin"],
            "distinctive_term_count": reliability["distinctive_term_count"],
            "distinctive_term_hits": reliability["distinctive_term_hits"],
            "top_distinctive_term_hits": reliability["top_distinctive_term_hits"],
            "retrieval_consensus": reliability["retrieval_consensus"],
            "consensus_support_rank": reliability["consensus_support_rank"],
            "context": context,
            "retrieved_sections": sections,
            "used_sections": sections if answer_basis == "MANUAL" else [],
            "candidate_sections": sections,
            "retrieved_documents": retrieved_documents,
            "retrieved_document_count": len(retrieved_documents),
            "diversity_supplement_used": len(retrieved_documents) > 2,
            "dense_anchor_promoted": any(
                document.get("dense_anchor_promoted", False)
                for document in retrieved_documents
            ),
            "section_expansion_used": any(
                document.get("section_expansion_used", False)
                for document in retrieved_documents
            ),
            "language_preference_used": any(
                document.get("language_preference_used", False)
                for document in retrieved_documents
            ),
            "evidence_compression_used": any(
                document.get("evidence_compressed", False)
                for document in retrieved_documents
            ),
            "unsupported_technical_terms": technical_term_violations,
            "best_chat_score": 0.0,
            "rewrite_time": rewrite_time,
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "generation_time": generation_time,
            "time_to_first_token": round(first_token_time, 3) if first_token_time is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "ollama_load_time": round(int(response_metric("load_duration", 0)) / 1e9, 3),
            "total_time": round(time.perf_counter() - total_start, 2),
            "prompt": "",
        }
