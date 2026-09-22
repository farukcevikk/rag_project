"""Backward-compatible imports for the canonical rag_core policies."""

try:
    from .rag_core.evidence import (
        GROUNDING_THRESHOLD,
        assess_context_reliability,
        detect_text_language,
        distinctive_query_terms,
        matching_tokens,
    )
    from .rag_core.retrieval_engine import (
        document_language,
        select_context,
    )
except ImportError:
    from rag_core.evidence import (
        GROUNDING_THRESHOLD,
        assess_context_reliability,
        detect_text_language,
        distinctive_query_terms,
        matching_tokens,
    )
    from rag_core.retrieval_engine import (
        document_language,
        select_context,
    )


def fuse_rankings(named_rankings, limit=10, rrf_k=60):
    """Legacy facade retaining process-local hash keys for older diagnostics."""
    documents = {}
    scores = {}
    source_ranks = {}
    for source, ranking in named_rankings:
        for rank, document in enumerate(ranking, 1):
            document_id = hash(document.page_content)
            documents[document_id] = document
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rank - 1 + rrf_k
            )
            ranks = source_ranks.setdefault(document_id, {})
            previous = ranks.get(source)
            ranks[source] = min(previous, rank) if previous else rank
    ordered_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [documents[item] for item in ordered_ids], scores, source_ranks


def select_reranked_context(
    ranked,
    max_documents=2,
    query_language=None,
    threshold=GROUNDING_THRESHOLD,
    policy="legacy_threshold",
):
    return select_context(
        ranked,
        max_documents,
        query_language,
        threshold,
        policy=policy,
    )


__all__ = [
    "GROUNDING_THRESHOLD",
    "assess_context_reliability",
    "detect_text_language",
    "distinctive_query_terms",
    "document_language",
    "fuse_rankings",
    "matching_tokens",
    "select_reranked_context",
]
