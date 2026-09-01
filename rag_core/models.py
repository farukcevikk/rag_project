from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    current_query: str
    contextual_query: str
    rewrite_time: float

    @property
    def retrieval_queries(self):
        if self.contextual_query and self.contextual_query != self.current_query:
            return (self.current_query, self.contextual_query)
        return (self.current_query,)


@dataclass(frozen=True)
class RetrievalResult:
    context: str
    sections: list
    retrieval_time: float
    rerank_time: float
    best_score: float
    selected_documents: list
    candidate_documents: list
    gate_threshold: float = -2.5
    gate_type: str = "cross_encoder_relevance_threshold"
    evidence_sufficiency: str = "not_independently_measured"
