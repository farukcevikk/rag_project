from dataclasses import dataclass

from .evidence import GROUNDING_THRESHOLD


@dataclass(frozen=True)
class RuntimeConfig:
    """Canonical runtime settings shared by orchestration components."""

    model_name: str
    rewrite_model_name: str = "gpt-oss:20b"
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "bge-m3:latest"
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder_device: str = "cuda"
    generation_num_ctx: int = 3072
    max_answer_tokens: int = 384
    keep_alive: str = "30m"
    history_messages: int = 3
    vector_k: int = 10
    bm25_k: int = 10
    rerank_candidates: int = 10
    context_k: int = 2
    child_chunk_size: int = 400
    child_chunk_overlap: int = 80
    grounding_threshold: float = GROUNDING_THRESHOLD

    def __post_init__(self):
        if self.generation_num_ctx < 512:
            raise ValueError("generation_num_ctx en az 512 olmalıdır")
        if self.max_answer_tokens < 1:
            raise ValueError("max_answer_tokens pozitif olmalıdır")
        if self.context_k < 1 or self.rerank_candidates < self.context_k:
            raise ValueError("rerank_candidates, context_k değerinden küçük olamaz")
        if not 0 <= self.child_chunk_overlap < self.child_chunk_size:
            raise ValueError("child chunk overlap değeri geçersiz")
