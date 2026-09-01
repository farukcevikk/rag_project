"""Deterministic evidence acceptance and passive diagnostics."""

import re


GROUNDING_THRESHOLD = -2.5
EVIDENCE_CONSENSUS_MAX_RANK = 3

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


def matching_tokens(value):
    normalized = str(value or "").casefold().replace("\u0307", "")
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def detect_text_language(value):
    text = str(value or "")
    tokens = set(matching_tokens(text))
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
    return {
        token for token in matching_tokens(query)
        if len(token) >= 3
        and token not in GENERIC_QUERY_STOPWORDS
        and not token.isdigit()
    }


def assess_context_reliability(
    query,
    best_score,
    retrieved_documents,
    threshold=GROUNDING_THRESHOLD,
):
    """Use one fail-closed score gate; other signals remain diagnostics only."""
    documents = retrieved_documents or []
    scores = sorted(
        (
            float(document["reranker_score"])
            for document in documents
            if document.get("reranker_score") is not None
        ),
        reverse=True,
    )
    score_margin = scores[0] - scores[1] if len(scores) >= 2 else None
    terms = distinctive_query_terms(query)
    evidence_terms = set()
    top_term_hits = 0
    consensus_ranks = []
    for rank, document in enumerate(documents, 1):
        document_terms = terms & set(matching_tokens(document.get("content", "")))
        evidence_terms.update(document_terms)
        if rank == 1:
            top_term_hits = len(document_terms)
        vector_rank = document.get("vector_rank")
        bm25_rank = document.get("bm25_rank")
        if (
            isinstance(vector_rank, int)
            and isinstance(bm25_rank, int)
            and vector_rank <= EVIDENCE_CONSENSUS_MAX_RANK
            and bm25_rank <= EVIDENCE_CONSENSUS_MAX_RANK
        ):
            consensus_ranks.append(rank)
    accepted = float(best_score) > float(threshold)
    return {
        "is_reliable": accepted,
        "reason": "score_gate_pass" if accepted else "score_gate_rejected",
        "score_margin": round(score_margin, 4) if score_margin is not None else None,
        "distinctive_term_count": len(terms),
        "distinctive_term_hits": len(evidence_terms),
        "top_distinctive_term_hits": top_term_hits,
        "retrieval_consensus": bool(consensus_ranks),
        "consensus_support_rank": consensus_ranks[0] if consensus_ranks else None,
    }
