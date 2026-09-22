from dataclasses import dataclass

from .evidence import GROUNDING_THRESHOLD


RERANKER_MODELS = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "qwen": "Qwen/Qwen3-Reranker-0.6B",
}
RUNTIME_PROFILE_FIELDS = (
    "reranker_backend",
    "context_selection_policy",
    "evidence_gate_backend",
    "generation_grounding_mode",
    "query_decomposition_enabled",
)
RUNTIME_PROFILES = {
    "production_bge": {
        "reranker_backend": "bge",
        "context_selection_policy": "legacy_threshold",
        "evidence_gate_backend": "legacy_ce",
        "generation_grounding_mode": "legacy",
        "query_decomposition_enabled": True,
    },
    "qwen_candidate": {
        "reranker_backend": "qwen",
        "context_selection_policy": "rank_only_top2",
        "evidence_gate_backend": "single_call_structured",
        "generation_grounding_mode": "strict_context",
        "query_decomposition_enabled": True,
    },
}
RUNTIME_PROFILE_MODELS = {
    "qwen_candidate": "qwen2.5:14b-instruct",
}
CONTEXT_SELECTION_POLICIES = {"legacy_threshold", "rank_only_top2"}
EVIDENCE_GATE_BACKENDS = {
    "legacy_ce",
    "conservative_hybrid",
    "single_call_structured",
}
GENERATION_GROUNDING_MODES = {"legacy", "strict_context"}


def runtime_profile_settings(profile_name):
    """Return one complete profile so migration flags cannot drift apart."""
    try:
        return dict(RUNTIME_PROFILES[profile_name])
    except KeyError as error:
        supported = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(
            f"Bilinmeyen runtime profile: {profile_name}. Desteklenenler: {supported}"
        ) from error


def identify_runtime_profile(config):
    values = {
        field: getattr(config, field)
        for field in RUNTIME_PROFILE_FIELDS
    }
    for profile_name, settings in RUNTIME_PROFILES.items():
        required_model = RUNTIME_PROFILE_MODELS.get(profile_name)
        if values == settings and (
            required_model is None or config.model_name == required_model
        ):
            return profile_name
    return "custom"


@dataclass(frozen=True)
class RuntimeConfig:
    """Canonical runtime settings shared by orchestration components."""

    model_name: str
    rewrite_model_name: str = "gpt-oss:20b"
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "bge-m3:latest"
    reranker_backend: str = "bge"
    cross_encoder_model: str | None = None
    cross_encoder_device: str = "cuda"
    cross_encoder_max_length: int = 512
    cross_encoder_dtype: str | None = "float16"
    context_selection_policy: str = "legacy_threshold"
    evidence_gate_backend: str = "legacy_ce"
    generation_grounding_mode: str = "legacy"
    semantic_gate_model: str = "qwen2.5:14b-instruct"
    semantic_gate_num_ctx: int = 3072
    semantic_gate_max_tokens: int = 192
    generation_num_ctx: int = 3072
    max_answer_tokens: int = 384
    keep_alive: str = "30m"
    history_messages: int = 3
    vector_k: int = 10
    bm25_k: int = 10
    rerank_candidates: int = 15
    context_k: int = 2
    child_chunk_size: int = 400
    child_chunk_overlap: int = 80
    grounding_threshold: float = GROUNDING_THRESHOLD
    sufficiency_review_ceiling: float = 0.30
    rewrite_bypass_threshold: float = 0.30
    query_decomposition_enabled: bool = True
    decomposition_max_subqueries: int = 2

    @property
    def runtime_profile(self):
        return identify_runtime_profile(self)

    def __post_init__(self):
        if self.reranker_backend not in RERANKER_MODELS:
            raise ValueError("reranker_backend yalnız bge veya qwen olabilir")
        if self.context_selection_policy not in CONTEXT_SELECTION_POLICIES:
            raise ValueError("context_selection_policy desteklenmiyor")
        if self.evidence_gate_backend not in EVIDENCE_GATE_BACKENDS:
            raise ValueError("evidence_gate_backend desteklenmiyor")
        if self.generation_grounding_mode not in GENERATION_GROUNDING_MODES:
            raise ValueError("generation_grounding_mode desteklenmiyor")
        if self.cross_encoder_model is None:
            object.__setattr__(
                self,
                "cross_encoder_model",
                RERANKER_MODELS[self.reranker_backend],
            )
        if (
            self.reranker_backend == "qwen"
            and self.context_selection_policy != "rank_only_top2"
        ):
            raise ValueError(
                "Qwen reranker yalnız rank_only_top2 context selection ile kullanılabilir"
            )
        if self.reranker_backend == "qwen" and self.evidence_gate_backend == "legacy_ce":
            raise ValueError(
                "Qwen relevance skoru legacy_ce evidence gate olarak kullanılamaz"
            )
        if (
            self.evidence_gate_backend == "single_call_structured"
            and self.context_selection_policy != "rank_only_top2"
        ):
            raise ValueError(
                "single_call_structured gate rank_only_top2 context gerektirir"
            )
        if (
            self.evidence_gate_backend == "conservative_hybrid"
            and self.context_selection_policy != "rank_only_top2"
        ):
            raise ValueError(
                "conservative_hybrid gate rank_only_top2 context gerektirir"
            )
        if (
            self.evidence_gate_backend in {"conservative_hybrid", "single_call_structured"}
            and self.generation_grounding_mode != "strict_context"
        ):
            raise ValueError(
                "deneysel gate strict_context generation modu gerektirir"
            )
        if self.generation_num_ctx < 512:
            raise ValueError("generation_num_ctx en az 512 olmalıdır")
        if self.semantic_gate_num_ctx != self.generation_num_ctx:
            raise ValueError("semantic gate ve generation num_ctx aynı olmalıdır")
        if self.semantic_gate_max_tokens < 1:
            raise ValueError("semantic_gate_max_tokens pozitif olmalıdır")
        if self.max_answer_tokens < 1:
            raise ValueError("max_answer_tokens pozitif olmalıdır")
        if self.cross_encoder_max_length < 1:
            raise ValueError("cross_encoder_max_length pozitif olmalıdır")
        if self.cross_encoder_dtype not in {None, "float32", "float16", "bfloat16"}:
            raise ValueError("cross_encoder_dtype desteklenmiyor")
        if self.context_k < 1 or self.rerank_candidates < self.context_k:
            raise ValueError("rerank_candidates, context_k değerinden küçük olamaz")
        if not 0 <= self.child_chunk_overlap < self.child_chunk_size:
            raise ValueError("child chunk overlap değeri geçersiz")
        if self.sufficiency_review_ceiling < self.grounding_threshold:
            raise ValueError(
                "sufficiency_review_ceiling, grounding_threshold değerinden küçük olamaz"
            )
        if not 0.0 <= self.rewrite_bypass_threshold <= 1.0:
            raise ValueError("rewrite_bypass_threshold 0 ile 1 arasında olmalıdır")
        if self.decomposition_max_subqueries != 2:
            raise ValueError("Mevcut decomposition sözleşmesi tam iki alt sorgu destekler")
