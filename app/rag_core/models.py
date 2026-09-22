from dataclasses import dataclass

from .evidence import GROUNDING_THRESHOLD


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
class DecompositionPlan:
    """A fail-closed plan for at most two independent retrieval intents."""

    original_query: str
    subqueries: tuple[str, ...]
    planning_time: float

    @property
    def is_decomposed(self):
        return len(self.subqueries) == 2


@dataclass(frozen=True)
class RetrievalResult:
    context: str
    sections: list
    retrieval_time: float
    rerank_time: float
    best_score: float
    selected_documents: list
    candidate_documents: list
    gate_threshold: float | None = GROUNDING_THRESHOLD
    gate_type: str = "cross_encoder_relevance_threshold"
    evidence_sufficiency: str = "not_independently_measured"
