"""Single owner for the legacy UI and benchmark result schema."""

import time


def section_names(documents):
    names = []
    for document in documents or []:
        for key in ("detail_section", "section", "main_section"):
            value = document.get(key, "")
            if value and value not in names:
                names.append(value)
    return names


def build_turn_result(
    *,
    answer,
    basis,
    current_query,
    contextual_query,
    query_language,
    threshold,
    context="",
    sections=None,
    selected_documents=None,
    candidate_documents=None,
    reliability=None,
    rewrite_time=0.0,
    retrieval_time=0.0,
    rerank_time=0.0,
    generation=None,
    total_started=None,
    reason_override=None,
    gate_type="cross_encoder_relevance_threshold",
    evidence_sufficiency="not_independently_measured",
    runtime_profile="custom",
    reranker_backend="bge",
    context_selection_policy="legacy_threshold",
    evidence_gate_backend="legacy_ce",
    generation_grounding_mode="legacy",
    decomposition_time=0.0,
    decomposition_queries=None,
    decomposition_aspect_scores=None,
):
    sections = list(sections or [])
    selected_documents = list(selected_documents or [])
    candidate_documents = list(candidate_documents or selected_documents)
    reliability = reliability or {
        "is_reliable": False,
        "reason": reason_override or "not_applicable",
        "score_margin": None,
        "distinctive_term_count": 0,
        "distinctive_term_hits": 0,
        "top_distinctive_term_hits": 0,
        "retrieval_consensus": False,
        "consensus_support_rank": None,
    }
    best_score = (
        selected_documents[0].get("reranker_score")
        if selected_documents else None
    )
    intent = (
        "TEKNİK" if basis == "MANUAL"
        else "SOHBET" if basis == "CONVERSATION"
        else "KANIT_YETERSİZ"
    )
    generation_values = {
        "marker_compliance": None,
        "empty_output": False,
        "unsupported_technical_terms": [],
        "generation_time": 0.0,
        "time_to_first_token": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "finish_reason": "",
        "truncated": False,
        "tokens_per_second": 0.0,
        "ollama_load_time": 0.0,
        "thought_process": "",
        "structured_kind": None,
        "parser_failure": False,
        "parser_error": None,
        "streaming_mode": "not_applicable",
    }
    if generation is not None:
        generation_values = {
            "marker_compliance": generation.marker_compliance,
            "empty_output": generation.empty_output,
            "unsupported_technical_terms": generation.unsupported_technical_terms,
            "generation_time": generation.generation_time,
            "time_to_first_token": generation.time_to_first_token,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "finish_reason": generation.finish_reason,
            "truncated": generation.truncated,
            "tokens_per_second": generation.tokens_per_second,
            "ollama_load_time": generation.ollama_load_time,
            "thought_process": generation.thought_process,
            "structured_kind": generation.structured_kind,
            "parser_failure": generation.parser_failure,
            "parser_error": generation.parser_error,
            "streaming_mode": generation.streaming_mode,
        }
    return {
        "answer": answer,
        "intent": intent,
        "answer_basis": basis,
        "evidence_marker_compliance": generation_values["marker_compliance"],
        "empty_model_output": generation_values["empty_output"],
        "search_query": contextual_query,
        "evidence_query": contextual_query or current_query,
        "standalone_question": current_query,
        "query_language": query_language,
        "response_language": getattr(generation, "response_language", None),
        "best_distance": best_score,
        "avg_distance": best_score,
        "best_reranker_score": best_score,
        "best_reranker_relevance": best_score,
        "reranker_backend": reranker_backend,
        "runtime_profile": runtime_profile,
        "context_selection_policy": context_selection_policy,
        "threshold": threshold,
        "generation_gate_passed": reliability["is_reliable"],
        "gate_type": gate_type,
        "evidence_sufficiency": evidence_sufficiency,
        "semantic_decision": reliability.get("semantic_decision"),
        "grounding_decision": generation_values["structured_kind"],
        "requested_aspects": reliability.get("requested_aspects", []),
        "supported_aspects": reliability.get("supported_aspects", []),
        "missing_aspects": reliability.get("missing_aspects", []),
        "generation_grounding_mode": generation_grounding_mode,
        "evidence_gate_time": reliability.get("evidence_gate_time", 0.0),
        "parser_failure": generation_values["parser_failure"],
        "parser_error": generation_values["parser_error"],
        "streaming_mode": generation_values["streaming_mode"],
        "context_is_reliable": reliability["is_reliable"],
        "context_reliability_reason": reason_override or reliability["reason"],
        "reranker_score_margin": reliability["score_margin"],
        "distinctive_term_count": reliability["distinctive_term_count"],
        "distinctive_term_hits": reliability["distinctive_term_hits"],
        "top_distinctive_term_hits": reliability["top_distinctive_term_hits"],
        "retrieval_consensus": reliability["retrieval_consensus"],
        "consensus_support_rank": reliability["consensus_support_rank"],
        "context": context,
        "retrieved_sections": sections,
        "used_sections": sections if basis == "MANUAL" else [],
        "candidate_sections": section_names(candidate_documents),
        "retrieved_documents": selected_documents,
        "candidate_documents": candidate_documents,
        "retrieved_document_count": len(selected_documents),
        "unsupported_technical_terms": generation_values["unsupported_technical_terms"],
        "best_chat_score": 0.0,
        "rewrite_time": rewrite_time,
        "decomposition_time": decomposition_time,
        "decomposition_applied": len(decomposition_queries or []) == 2,
        "decomposition_queries": list(decomposition_queries or []),
        "decomposition_aspect_scores": list(decomposition_aspect_scores or []),
        "retrieval_time": retrieval_time,
        "rerank_time": rerank_time,
        "generation_time": generation_values["generation_time"],
        "time_to_first_token": generation_values["time_to_first_token"],
        "input_tokens": generation_values["input_tokens"],
        "output_tokens": generation_values["output_tokens"],
        "finish_reason": generation_values["finish_reason"],
        "truncated": generation_values["truncated"],
        "tokens_per_second": generation_values["tokens_per_second"],
        "ollama_load_time": generation_values["ollama_load_time"],
        "thought_process": generation_values["thought_process"],
        "total_time": round(time.perf_counter() - total_started, 3) if total_started else 0.0,
        "prompt": "",
    }
