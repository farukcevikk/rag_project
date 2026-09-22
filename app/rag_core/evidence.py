"""Deterministic evidence acceptance and passive diagnostics."""

import math
import re


# Production BGE reranker emits sigmoid scores. This model-specific threshold
# must move together with the configured reranker; it is not a sufficiency score.
GROUNDING_THRESHOLD = 0.03
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


def structural_evidence_check(context, retrieved_documents, best_score):
    """Reject only mechanically invalid evidence inputs, never semantic content."""
    documents = list(retrieved_documents or [])
    if not str(context or "").strip():
        return False, "empty_context"
    if not documents:
        return False, "no_selected_documents"
    if best_score is None:
        return False, "reranker_not_run"
    try:
        if not math.isfinite(float(best_score)) or float(best_score) <= -999.0:
            return False, "invalid_reranker_output"
    except (TypeError, ValueError):
        return False, "invalid_reranker_output"
    for document in documents:
        if not str(document.get("content", "")).strip():
            return False, "empty_passage"
        if not str(document.get("source_file", "")).strip():
            return False, "invalid_source_metadata"
        if not any(
            str(document.get(key, "")).strip()
            for key in ("main_section", "section", "detail_section")
        ):
            return False, "invalid_source_metadata"
    return True, "structural_gate_pass"


def evaluate_evidence_gate(
    *,
    backend,
    query,
    context,
    retrieved_documents,
    best_score,
    threshold=GROUNDING_THRESHOLD,
    semantic_checker=None,
):
    """Evaluate evidence without treating reranker relevance as sufficiency."""
    if backend == "legacy_ce":
        result = assess_context_reliability(
            query,
            best_score,
            retrieved_documents,
            threshold=threshold,
        )
        result.update({
            "semantic_decision": None,
            "requested_aspects": [],
            "supported_aspects": [],
            "missing_aspects": [],
            "evidence_gate_time": 0.0,
        })
        return result
    if backend not in {"conservative_hybrid", "single_call_structured"}:
        raise ValueError(f"Bilinmeyen evidence gate backend: {backend}")

    structurally_valid, structural_reason = structural_evidence_check(
        context, retrieved_documents, best_score
    )
    base = assess_context_reliability(
        query,
        best_score,
        retrieved_documents,
        threshold=float("inf"),
    )
    if not structurally_valid:
        base.update({
            "is_reliable": False,
            "reason": structural_reason,
            "semantic_decision": "UNSUPPORTED",
            "requested_aspects": [],
            "supported_aspects": [],
            "missing_aspects": [],
            "evidence_sufficiency": "structural_reject",
            "evidence_gate_time": 0.0,
        })
        return base
    if backend == "single_call_structured":
        base.update({
            "is_reliable": True,
            "reason": "structural_gate_pass_pending_single_call",
            "semantic_decision": None,
            "requested_aspects": [],
            "supported_aspects": [],
            "missing_aspects": [],
            "evidence_sufficiency": "decided_by_single_generation_call",
            "evidence_gate_time": 0.0,
        })
        return base
    if semantic_checker is None:
        base.update({
            "is_reliable": False,
            "reason": "semantic_checker_unavailable",
            "semantic_decision": "UNSUPPORTED",
            "requested_aspects": [],
            "supported_aspects": [],
            "missing_aspects": [],
            "evidence_sufficiency": "semantic_checker_unavailable",
            "evidence_gate_time": 0.0,
        })
        return base

    try:
        semantic = semantic_checker.evaluate(query=query, context=context)
    except Exception as error:
        base.update({
            "is_reliable": False,
            "reason": f"semantic_gate_error:{type(error).__name__}",
            "semantic_decision": "UNSUPPORTED",
            "requested_aspects": [],
            "supported_aspects": [],
            "missing_aspects": [],
            "evidence_sufficiency": "semantic_gate_error",
            "evidence_gate_time": 0.0,
        })
        return base

    decision = semantic["decision"]
    accepted = decision == "SUPPORTED"
    base.update({
        "is_reliable": accepted,
        "reason": (
            "semantic_gate_supported"
            if accepted
            else f"semantic_gate_{decision.casefold()}"
        ),
        "semantic_decision": decision,
        "requested_aspects": semantic["requested_aspects"],
        "supported_aspects": semantic["supported_aspects"],
        "missing_aspects": semantic["missing_aspects"],
        "evidence_sufficiency": decision.casefold(),
        "evidence_gate_time": semantic.get("latency_s", 0.0),
    })
    return base
