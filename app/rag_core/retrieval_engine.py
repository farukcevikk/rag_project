"""Hybrid retrieval with authoritative-query coverage and pluggable reranking."""

import hashlib
import time

import numpy as np

from .evidence import GROUNDING_THRESHOLD
from .models import RetrievalResult


def stable_document_id(document):
    return hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()


def retrieval_anchors(document):
    anchors = (getattr(document, "metadata", {}) or {}).get("_retrieval_anchors", [])
    if isinstance(anchors, str):
        anchors = [anchors]
    return [str(value).strip() for value in anchors if str(value).strip()]


def merge_document_anchors(existing, incoming):
    """Preserve child evidence when the same parent also arrives from BM25."""
    anchors = []
    for value in (*retrieval_anchors(existing), *retrieval_anchors(incoming)):
        if value not in anchors:
            anchors.append(value)
    if not anchors:
        return existing
    existing.metadata = dict(existing.metadata or {})
    existing.metadata["_retrieval_anchors"] = anchors
    return existing


def rerank_passages(document):
    """Return parent text for CrossEncoder precision; child anchors are diagnostics only.

    Previous child-only approach failed on kamera example: dense retrieval found
    correct parent but wrong child was selected because CrossEncoder lost parent context.
    Parent-level reranking ensures full section visibility for accurate selection.
    """
    return [document.page_content]


def document_heading(document):
    """Build the semantic breadcrumb that an atomic child does not contain."""
    metadata = getattr(document, "metadata", {}) or {}
    return " | ".join(filter(None, [
        metadata.get("Ana_Baslik", ""),
        metadata.get("Alt_Baslik", ""),
        metadata.get("Detay_Baslik", ""),
        metadata.get("Mikro_Baslik", ""),
    ]))


def cross_encoder_passage(title, passage):
    """Give the reranker both section identity and answer-bearing child text."""
    return f"{title}\n{passage}" if title else passage


def fuse_rankings(named_rankings, limit=10, rrf_k=60):
    documents = {}
    scores = {}
    source_ranks = {}
    for source, ranking in named_rankings:
        for rank, document in enumerate(ranking, 1):
            document_id = stable_document_id(document)
            if document_id in documents:
                documents[document_id] = merge_document_anchors(
                    documents[document_id], document
                )
            else:
                documents[document_id] = document
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                rank - 1 + rrf_k
            )
            ranks = source_ranks.setdefault(document_id, {})
            previous = ranks.get(source)
            ranks[source] = min(previous, rank) if previous else rank
    ordered_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [documents[item] for item in ordered_ids], scores, source_ranks


def authoritative_candidate_pool(named_rankings, authoritative_sources, limit):
    """Guarantee channel coverage for the resolved query before supplements.

    RRF can place a document found by only one channel behind documents that
    receive weak matches from several queries.  The resolved query is the
    model's standalone expression of intent, so its dense and lexical top-k
    candidates must not be displaced by the vague follow-up query.  Remaining
    capacity is filled from the all-query RRF order.
    """
    all_limit = sum(len(ranking) for _, ranking in named_rankings)
    all_ranked, fused_scores, source_ranks = fuse_rankings(
        named_rankings, limit=all_limit
    )
    authoritative = [
        (source, ranking)
        for source, ranking in named_rankings
        if source in authoritative_sources
    ]
    primary, _, _ = fuse_rankings(authoritative, limit=limit)
    candidates = list(primary)
    seen = {stable_document_id(document) for document in candidates}
    for document in all_ranked:
        if len(candidates) >= limit:
            break
        document_id = stable_document_id(document)
        if document_id not in seen:
            candidates.append(document)
            seen.add(document_id)
    return candidates, fused_scores, source_ranks


def document_language(document):
    language = str((getattr(document, "metadata", {}) or {}).get("Language", "")).casefold()
    return language if language in {"tr", "en"} else None


def select_context(
    ranked,
    max_documents,
    query_language,
    threshold,
    policy="legacy_threshold",
):
    if not ranked or max_documents <= 0:
        return []
    if policy == "legacy_threshold":
        eligible = [item for item in ranked if float(item[0]) > float(threshold)]
        fallback = list(ranked[:1])
    elif policy == "rank_only_top2":
        eligible = list(ranked)
        fallback = []
    else:
        raise ValueError(f"Bilinmeyen context selection policy: {policy}")
    if query_language:
        same_language = [
            item for item in eligible if document_language(item[1]) == query_language
        ]
        if same_language:
            eligible = same_language
    pool = eligible or fallback
    selected = []
    seen_headings = set()
    for item in pool:
        document = item[1]
        heading = document_heading(document).strip().casefold()
        identity = heading or stable_document_id(document)
        if identity in seen_headings:
            continue
        seen_headings.add(identity)
        selected.append(item)
        if len(selected) >= max_documents:
            break
    return selected


class HybridRetrievalEngine:
    def __init__(self, vector_retriever, bm25_retriever, reranker, config, language_detector):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.config = config
        self.language_detector = language_detector

    def retrieve(self, plan):
        retrieval_started = time.perf_counter()
        queries = list(plan.retrieval_queries)
        query_language = self.language_detector(queries[-1])
        vector_rankings = [self.vector_retriever.invoke(query) for query in queries]
        bm25_rankings = [self.bm25_retriever.invoke(query) for query in queries]
        retrieval_time = round(time.perf_counter() - retrieval_started, 2)

        named_rankings = [
            *((f"vector_q{index}", ranking) for index, ranking in enumerate(vector_rankings)),
            *((f"bm25_q{index}", ranking) for index, ranking in enumerate(bm25_rankings)),
        ]
        # The resolved dense and BM25 channels share one explicit parent-pool
        # budget. Channel coverage is preserved by authoritative_candidate_pool;
        # multiplying the configured value here made the real workload opaque.
        candidate_limit = self.config.rerank_candidates
        resolved_index = len(queries) - 1
        candidates, fused_scores, source_ranks = authoritative_candidate_pool(
            named_rankings,
            {
                f"vector_q{resolved_index}",
                f"bm25_q{resolved_index}",
            },
            limit=candidate_limit,
        )
        rank_only = self.config.context_selection_policy == "rank_only_top2"
        gate_threshold = None if rank_only else self.config.grounding_threshold
        gate_type = (
            "rank_only_no_relevance_gate"
            if rank_only
            else "cross_encoder_relevance_threshold"
        )
        evidence_sufficiency = (
            "not_evaluated_by_retrieval"
            if rank_only
            else "not_independently_measured"
        )
        if not candidates:
            return RetrievalResult(
                "", [], retrieval_time, 0.0, -999.0, [], [],
                gate_threshold,
                gate_type,
                evidence_sufficiency,
            )

        rerank_started = time.perf_counter()
        # Retrieval queries are recall probes; the final contextual query is
        # the standalone expression of the user's intent and therefore the
        # only query that should decide precision.  Taking a max across a vague
        # current message and its resolved form allowed accidental scores from
        # the vague message to override the resolved intent.
        rerank_query = queries[-1]
        # Score every distinct passage once.  The same overlapping child can be
        # attached to more than one fused candidate; duplicating it in the batch
        # adds work without adding evidence.
        passage_records = []
        passage_indexes = {}
        candidate_passage_indexes = []
        for document in candidates:
            indexes = []
            for passage in rerank_passages(document):
                title = document_heading(document)
                passage_key = (title, passage, document_language(document))
                index = passage_indexes.get(passage_key)
                if index is None:
                    index = len(passage_records)
                    passage_indexes[passage_key] = index
                    passage_records.append({
                        "text": passage,
                        "reranker_text": cross_encoder_passage(title, passage),
                        "title": title,
                        "language": document_language(document),
                    })
                indexes.append(index)
            candidate_passage_indexes.append(indexes)
        pairs = [[rerank_query, item["reranker_text"]] for item in passage_records]
        passage_scores = np.asarray(
            self.reranker.predict(pairs, show_progress_bar=False), dtype=float
        ).reshape(len(pairs))
        winning_passages = {}
        scores = []
        for document, indexes in zip(candidates, candidate_passage_indexes):
            best_index = max(indexes, key=lambda index: passage_scores[index])
            scores.append(passage_scores[best_index])
            winning_passages[stable_document_id(document)] = passage_records[best_index]["text"]
        scores = np.asarray(scores, dtype=float)
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        selected = select_context(
            ranked,
            max_documents=self.config.context_k,
            query_language=query_language,
            threshold=self.config.grounding_threshold,
            policy=self.config.context_selection_policy,
        )
        rerank_time = round(time.perf_counter() - rerank_started, 2)

        selected_documents = [
            self._document_metadata(
                rank,
                score,
                doc,
                fused_scores,
                source_ranks,
                winning_passages,
            )
            for rank, (score, doc) in enumerate(selected, 1)
        ]
        candidate_documents = [
            self._document_metadata(
                rank,
                score,
                doc,
                fused_scores,
                source_ranks,
                winning_passages,
            )
            for rank, (score, doc) in enumerate(ranked, 1)
        ]
        sections = []
        context_parts = []
        for index, (_, document) in enumerate(selected, 1):
            metadata = document.metadata
            headings = [
                metadata.get("Ana_Baslik", ""),
                metadata.get("Alt_Baslik", ""),
                metadata.get("Detay_Baslik", ""),
                metadata.get("Mikro_Baslik", ""),
            ]
            for heading in reversed(headings):
                if heading and heading not in sections:
                    sections.append(heading)
            evidence_passage = winning_passages[stable_document_id(document)]
            context_parts.append(
                f"[Source {index} | {' | '.join(filter(None, headings))}]\n{evidence_passage}"
            )
        return RetrievalResult(
            "\n\n---\n\n".join(context_parts),
            sections,
            retrieval_time,
            rerank_time,
            float(selected[0][0]),
            selected_documents,
            candidate_documents,
            gate_threshold,
            gate_type,
            evidence_sufficiency,
        )

    @staticmethod
    def _document_metadata(
        rank,
        score,
        document,
        fused_scores,
        source_ranks,
        winning_passages,
    ):
        document_id = stable_document_id(document)
        ranks = source_ranks.get(document_id, {})
        vector_ranks = [value for key, value in ranks.items() if key.startswith("vector_")]
        bm25_ranks = [value for key, value in ranks.items() if key.startswith("bm25_")]
        metadata = document.metadata
        return {
            "rank": rank,
            "reranker_score": round(float(score), 4),
            "hybrid_score": round(float(fused_scores.get(document_id, 0.0)), 6),
            "vector_rank": min(vector_ranks) if vector_ranks else None,
            "bm25_rank": min(bm25_ranks) if bm25_ranks else None,
            "retrieval_sources": ranks,
            "source_file": metadata.get("Source_File", ""),
            "main_section": metadata.get("Ana_Baslik", ""),
            "section": metadata.get("Alt_Baslik", ""),
            "detail_section": metadata.get("Detay_Baslik", ""),
            "language": document_language(document),
            "retrieval_anchor_count": len(retrieval_anchors(document)),
            # Parent text is both reranked and sent to generation. Child anchors
            # are preserved in metadata for retrieval diagnostics, not scoring.
            "content": winning_passages[document_id],
            "evidence_chars": len(winning_passages[document_id]),
            "parent_chars": len(document.page_content),
        }
