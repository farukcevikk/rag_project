# rag_pipeline.py 

import time
from dataclasses import replace
import ollama # Langchain yerine doğrudan Ollama'nın resmi kütüphanesi 

try:
    from .rag_core.answering import (
        AnswerGenerator,
        EMPTY_ANSWER_FALLBACK,
        finalize_grounded_answer,
        grounding_refusal,
        unsupported_technical_terms,
    )
    from .rag_core.config import (
        RUNTIME_PROFILE_MODELS,
        RuntimeConfig,
        runtime_profile_settings,
    )
    from .rag_core.evidence import (
        GROUNDING_THRESHOLD,
        detect_text_language,
        evaluate_evidence_gate,
    )
    from .rag_core.models import QueryPlan
    from .rag_core.index_store import load_index
    from .rag_core.index_validation import (
        INDEX_MANIFEST_PATH,
        INDEX_SCHEMA_VERSION,
        validate_index_manifest,
    )
    from .rag_core.reranker_backends import (
        LazyCrossEncoder,
        TensorRTCrossEncoder,
    )
    from .rag_core.retrieval_engine import HybridRetrievalEngine
    from .rag_core.results import build_turn_result
    from .rag_core.semantic_gate import OllamaSemanticAnswerabilityChecker
    from .rag_core.routing import (
        bounded_dialogue as route_bounded_dialogue,
        fast_social_response,
        is_social_message,
        plan_decomposition,
        plan_query,
    )
except ImportError:
    # `streamlit run app/main.py` app/ dizinini doğrudan sys.path'e ekler.
    from rag_core.answering import (
        AnswerGenerator,
        EMPTY_ANSWER_FALLBACK,
        finalize_grounded_answer,
        grounding_refusal,
        unsupported_technical_terms,
    )
    from rag_core.config import (
        RUNTIME_PROFILE_MODELS,
        RuntimeConfig,
        runtime_profile_settings,
    )
    from rag_core.evidence import (
        GROUNDING_THRESHOLD,
        detect_text_language,
        evaluate_evidence_gate,
    )
    from rag_core.models import QueryPlan
    from rag_core.index_store import load_index
    from rag_core.index_validation import (
        INDEX_MANIFEST_PATH,
        INDEX_SCHEMA_VERSION,
        validate_index_manifest,
    )
    from rag_core.reranker_backends import (
        LazyCrossEncoder,
        TensorRTCrossEncoder,
    )
    from rag_core.retrieval_engine import HybridRetrievalEngine
    from rag_core.results import build_turn_result
    from rag_core.semantic_gate import OllamaSemanticAnswerabilityChecker
    from rag_core.routing import (
        bounded_dialogue as route_bounded_dialogue,
        fast_social_response,
        is_social_message,
        plan_decomposition,
        plan_query,
    )

# ===================================================== 
# PIPELINE 
# ===================================================== 

class RAGPipeline: 
    @classmethod
    def from_profile(cls, profile_name, **kwargs):
        """Construct a pipeline from an atomic production/candidate profile."""
        settings = runtime_profile_settings(profile_name)
        required_model = RUNTIME_PROFILE_MODELS.get(profile_name)
        if required_model is not None:
            configured_model = kwargs.get("model_name", required_model)
            if configured_model != required_model:
                raise ValueError(
                    f"{profile_name} profili model_name={required_model!r} gerektirir"
                )
            kwargs["model_name"] = required_model
        conflicts = {
            key: (kwargs[key], value)
            for key, value in settings.items()
            if key in kwargs and kwargs[key] != value
        }
        if conflicts:
            details = ", ".join(
                f"{key}={actual!r} (profile: {expected!r})"
                for key, (actual, expected) in conflicts.items()
            )
            raise ValueError(f"Runtime profile ile çelişen ayarlar: {details}")
        resolved = dict(settings)
        resolved.update(kwargs)
        return cls(**resolved)

    def __init__(
        self,
        model_name="gpt-oss:20b",
        rewrite_model_name="gpt-oss:20b",
        generation_num_ctx=3072,
        max_answer_tokens=None,
        warmup_num_ctx=None,
        cross_encoder_model=None,
        cross_encoder_max_length=512,
        cross_encoder_dtype=None,
        grounding_threshold=None,
        sufficiency_review_ceiling=None,
        rewrite_bypass_threshold=None,
        reranker_backend="bge",
        context_selection_policy="legacy_threshold",
        evidence_gate_backend="legacy_ce",
        generation_grounding_mode="legacy",
        semantic_answerability_checker=None,
        query_decomposition_enabled=True,
        ollama_host=None,
    ):
        print(f">>> Modeller yükleniyor (Seçilen Model: {model_name})") 
        self.model_name = model_name 
        self.rewrite_model = rewrite_model_name
        self.think_mode = "low" if model_name.casefold().startswith("gpt-oss") else False
        self.rewrite_think_mode = (
            "low" if rewrite_model_name.casefold().startswith("gpt-oss") else False
        )
        # The answer prompt already asks for concise responses. Keep the same
        # output ceiling across models so GPT-OSS cannot double the worst-case
        # generation work merely because reasoning is enabled at "low".
        default_answer_tokens = 384
        resolved_num_ctx = int(generation_num_ctx)
        resolved_warmup_num_ctx = int(
            resolved_num_ctx if warmup_num_ctx is None else warmup_num_ctx
        )
        resolved_answer_tokens = int(
            default_answer_tokens if max_answer_tokens is None else max_answer_tokens
        )
        if resolved_num_ctx < 512 or resolved_warmup_num_ctx < 512:
            raise ValueError("num_ctx en az 512 olmalıdır")
        if resolved_warmup_num_ctx != resolved_num_ctx:
            raise ValueError("Warmup ve generation num_ctx aynı olmalıdır")
        if resolved_answer_tokens < 1:
            raise ValueError("max_answer_tokens pozitif olmalıdır")
        config_kwargs = dict(
            model_name=model_name,
            rewrite_model_name=rewrite_model_name,
            generation_num_ctx=resolved_num_ctx,
            max_answer_tokens=resolved_answer_tokens,
            cross_encoder_max_length=int(cross_encoder_max_length),
            reranker_backend=reranker_backend,
            context_selection_policy=context_selection_policy,
            evidence_gate_backend=evidence_gate_backend,
            generation_grounding_mode=generation_grounding_mode,
            query_decomposition_enabled=bool(query_decomposition_enabled),
        )
        if ollama_host is not None:
            config_kwargs["ollama_host"] = ollama_host
        if cross_encoder_model is not None:
            config_kwargs["cross_encoder_model"] = cross_encoder_model
        if cross_encoder_dtype is not None:
            config_kwargs["cross_encoder_dtype"] = cross_encoder_dtype
        if grounding_threshold is not None:
            config_kwargs["grounding_threshold"] = float(grounding_threshold)
        if sufficiency_review_ceiling is not None:
            config_kwargs["sufficiency_review_ceiling"] = float(
                sufficiency_review_ceiling
            )
        if rewrite_bypass_threshold is not None:
            config_kwargs["rewrite_bypass_threshold"] = float(
                rewrite_bypass_threshold
            )
        self.config = RuntimeConfig(**config_kwargs)
        self.generation_num_ctx = self.config.generation_num_ctx
        self.warmup_num_ctx = resolved_warmup_num_ctx
        self.max_answer_tokens = self.config.max_answer_tokens
        
        # Langchain yerine saf Ollama client'ı başlatıyoruz 
        self.client = ollama.Client(host=self.config.ollama_host)
        if semantic_answerability_checker is not None:
            self.semantic_answerability_checker = semantic_answerability_checker
        elif self.config.evidence_gate_backend == "conservative_hybrid":
            self.semantic_answerability_checker = OllamaSemanticAnswerabilityChecker(
                self.client,
                model=self.config.semantic_gate_model,
                num_ctx=self.config.semantic_gate_num_ctx,
                max_tokens=self.config.semantic_gate_max_tokens,
                keep_alive=self.config.keep_alive,
            )
        else:
            self.semantic_answerability_checker = None
        (
            self.embeddings,
            self.chroma_retriever,
            self.bm25_retriever,
            manifest,
        ) = load_index(self.config)
        self.vectorstore = self.chroma_retriever.vectorstore
        self.index_validation = {
            "status": "verified",
            "schema_version": manifest.get("schema_version"),
        }
        # CrossEncoder is the single, batched precision layer. Model allocation
        # is deferred until warmup so imports do not allocate GPU memory.
        self.cross_encoder = LazyCrossEncoder(
            self.config.cross_encoder_model,
            max_length=self.config.cross_encoder_max_length,
            device=self.config.cross_encoder_device,
            dtype=self.config.cross_encoder_dtype,
        )
        
        # SEÇENEK 2: TensorRT/GPU versiyonu (ONNX dosyası gerektirir)
        #self.cross_encoder = TensorRTCrossEncoder(
        #    model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
        #    onnx_path="cross_encoder.onnx"
        #)
        
        # --------------------------------------------------------- 

        self.pipeline_config = { 
            "architecture": "Jetson RAG (Hybrid Retrieval + Batched CrossEncoder)",
            "architecture_version": "vNext-boundary-1",
            "runtime_profile": self.config.runtime_profile,
            "rewrite_model": f"{self.rewrite_model} (history only)",
            "history_user_turns": self.config.history_messages,
            "has_heuristic_bypass": self.config.context_selection_policy == "legacy_threshold",
            "routing": "exact social fast path + LLM history resolution + fail-closed evidence gate",
            "query_decomposition": (
                "LLM-planned max-two atomic retrievals"
                if self.config.query_decomposition_enabled else "disabled"
            ),
            "chunk_size": self.config.child_chunk_size,
            "chunk_overlap": self.config.child_chunk_overlap,
            "vector_search_k": self.config.vector_k,
            "rerank_top_k": self.config.context_k,
            "cross_encoder_model": self.config.cross_encoder_model,
            "reranker_backend": self.config.reranker_backend,
            "cross_encoder_max_length": self.config.cross_encoder_max_length,
            "cross_encoder_dtype": self.config.cross_encoder_dtype or "model_default",
            "context_selection": (
                "reranker ordered top 2 parent passages without score filtering"
                if self.config.context_selection_policy == "rank_only_top2"
                else "trusted top 2 parent passages with legacy score filtering"
            ),
            "context_selection_policy": self.config.context_selection_policy,
            "followup_state": "bounded conversation to minimal standalone query",
            "followup_resolution": (
                "raw retrieval first; history resolver independent of reranker score"
                if self.config.context_selection_policy == "rank_only_top2"
                else "raw retrieval first; rewrite only for weak legacy score results"
            ),
            "language_routing": "trusted_same_language_first_with_cross_language_fallback",
            "max_answer_tokens": self.max_answer_tokens,
            "generation_num_ctx": self.generation_num_ctx,
            "condense_num_ctx": self.generation_num_ctx,
            "warmup_num_ctx": self.warmup_num_ctx,
            "think_mode": self.think_mode,
            "grounding_threshold": self.config.grounding_threshold,
            "sufficiency_review_ceiling": self.config.sufficiency_review_ceiling,
            "rewrite_bypass_threshold": self.config.rewrite_bypass_threshold,
            "evidence_gate": self.config.evidence_gate_backend,
            "evidence_gate_backend": self.config.evidence_gate_backend,
            "generation_grounding_mode": self.config.generation_grounding_mode,
            "semantic_gate_model": (
                self.config.semantic_gate_model
                if self.config.evidence_gate_backend == "conservative_hybrid"
                else None
            ),
            "streaming_mode": (
                "buffered_structured"
                if self.config.evidence_gate_backend == "single_call_structured"
                else "live"
            ),
            "index_validation": self.index_validation,
        } 
    def warmup(self):
        print("\n[INFO] Sistem Isınıyor: Modeller GPU'ya yükleniyor (Cold Start)...")
        start_time = time.perf_counter()
        
        # 1. Embedding Modelini Uyandır
        self.embeddings.embed_query("hello")
        
        # 2. Load the CrossEncoder outside the first operator request.
        if hasattr(self, "cross_encoder"):
            try:
                self.cross_encoder.predict(
                    [["hello", "world"]], show_progress_bar=False
                )
            except Exception:
                pass
        
        # Resolver only runs when bounded history exists. Keep it resident so a
        # follow-up does not pay a cold load; generation is warmed last.
        rewrite_model = getattr(self, "rewrite_model", self.model_name)
        rewrite_think_mode = getattr(self, "rewrite_think_mode", False)
        keep_alive = getattr(getattr(self, "config", None), "keep_alive", "30m")
        if rewrite_model != self.model_name:
            try:
                self.client.chat(
                    model=rewrite_model,
                    messages=[{"role": "user", "content": "hi"}],
                    think=rewrite_think_mode,
                    options={"num_predict": 1, "num_ctx": self.warmup_num_ctx},
                    keep_alive=keep_alive,
                )
            except Exception:
                pass

        try:
            self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                think=getattr(self, "think_mode", False),
                options={"num_predict": 1, "num_ctx": self.warmup_num_ctx},
                keep_alive=getattr(getattr(self, "config", None), "keep_alive", "30m"),
            )
        except Exception as e:
            pass
            
        elapsed = round(time.perf_counter() - start_time, 2)
        print(f"[INFO] Warmup Tamamlandı! ({elapsed} sn) - Bütün modeller VRAM'de hazır.\n")

    @staticmethod
    def _bounded_dialogue(history, max_user_turns=3):
        return route_bounded_dialogue(history, max_user_turns=max_user_turns)

    def condense_question(self, question, history):
        """Legacy tuple adapter around the canonical query planner."""
        plan = plan_query(
            question,
            history,
            self.client,
            getattr(self, "rewrite_model", self.model_name),
            self.generation_num_ctx,
            think_mode=getattr(self, "rewrite_think_mode", False),
            keep_alive=getattr(getattr(self, "config", None), "keep_alive", "30m"),
            max_user_turns=getattr(getattr(self, "config", None), "history_messages", 3),
        )
        return plan.current_query, plan.contextual_query, plan.rewrite_time

    def retrieve_context(self, query, contextual_query=None):
        """Legacy tuple adapter around the canonical retrieval engine."""
        config = getattr(self, "config", RuntimeConfig(model_name=getattr(self, "model_name", "test")))
        plan = QueryPlan(query, contextual_query or query, 0.0)
        result = HybridRetrievalEngine(
            self.chroma_retriever,
            self.bm25_retriever,
            self.cross_encoder,
            config,
            detect_text_language,
        ).retrieve(plan)
        self.last_candidate_documents = result.candidate_documents
        self.last_grounding_threshold = result.gate_threshold
        self.last_gate_type = result.gate_type
        self.last_evidence_sufficiency = result.evidence_sufficiency
        self.last_reranker_backend = config.reranker_backend
        self.last_context_selection_policy = config.context_selection_policy
        return (
            result.context,
            result.sections,
            result.retrieval_time,
            result.rerank_time,
            result.best_score,
            result.selected_documents,
        )

    def decompose_question(self, question):
        """Return a validated, fail-closed two-intent retrieval plan."""
        config = getattr(
            self,
            "config",
            RuntimeConfig(model_name=getattr(self, "model_name", "test")),
        )
        if not config.query_decomposition_enabled:
            from_model = None
        else:
            from_model = plan_decomposition(
                question,
                self.client,
                getattr(
                    self,
                    "rewrite_model",
                    getattr(self, "model_name", config.model_name),
                ),
                getattr(self, "generation_num_ctx", config.generation_num_ctx),
                think_mode=getattr(self, "rewrite_think_mode", False),
                keep_alive=config.keep_alive,
            )
        return from_model

    @staticmethod
    def _document_identity(document):
        return tuple(document.get(key, "") for key in (
            "source_file", "main_section", "section", "detail_section", "content"
        ))

    def retrieve_decomposed_context(self, subqueries):
        """Retrieve each intent independently while preserving a two-doc budget."""
        selected = []
        candidates = []
        seen_selected = set()
        retrieval_time = 0.0
        rerank_time = 0.0
        aspect_scores = []
        gate_threshold = None
        gate_type = None
        evidence_sufficiency = None

        for subquery in subqueries:
            (
                _context,
                _sections,
                query_retrieval_time,
                query_rerank_time,
                query_best_score,
                query_documents,
            ) = self.retrieve_context(subquery, subquery)
            query_candidates = list(getattr(
                self, "last_candidate_documents", query_documents
            ))
            for document in query_candidates:
                candidates.append({**document, "decomposition_query": subquery})
            chosen = next(
                (
                    document for document in query_documents
                    if self._document_identity(document) not in seen_selected
                ),
                query_documents[0] if query_documents else None,
            )
            if chosen is not None:
                identity = self._document_identity(chosen)
                if identity not in seen_selected:
                    seen_selected.add(identity)
                    selected.append({**chosen, "decomposition_query": subquery})
                aspect_scores.append(float(chosen.get("reranker_score", query_best_score)))
            else:
                aspect_scores.append(float(query_best_score))
            retrieval_time += query_retrieval_time
            rerank_time += query_rerank_time
            gate_threshold = getattr(self, "last_grounding_threshold", gate_threshold)
            gate_type = getattr(self, "last_gate_type", gate_type)
            evidence_sufficiency = getattr(
                self, "last_evidence_sufficiency", evidence_sufficiency
            )

        deduplicated_candidates = []
        seen_candidates = set()
        for document in [*selected, *candidates]:
            identity = self._document_identity(document)
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
            deduplicated_candidates.append(document)
        self.last_candidate_documents = deduplicated_candidates
        self.last_grounding_threshold = gate_threshold
        self.last_gate_type = gate_type or "cross_encoder_relevance_threshold"
        self.last_evidence_sufficiency = (
            evidence_sufficiency or "not_independently_measured"
        )
        self.last_decomposition_aspect_scores = aspect_scores

        sections = []
        context_parts = []
        for index, document in enumerate(selected[:2], 1):
            headings = [
                document.get("main_section", ""),
                document.get("section", ""),
                document.get("detail_section", ""),
            ]
            for heading in reversed(headings):
                if heading and heading not in sections:
                    sections.append(heading)
            context_parts.append(
                f"[Source {index} | {' | '.join(filter(None, headings))}]\n"
                f"{document.get('content', '')}"
            )
        return (
            "\n\n---\n\n".join(context_parts),
            sections,
            round(retrieval_time, 2),
            round(rerank_time, 2),
            min(aspect_scores) if aspect_scores else -999.0,
            selected[:2],
        )

    def answer_question(
        self,
        user_input,
        history_messages=None,
        stream_callback=None,
        thinking_callback=None,
    ):
        history_messages = history_messages or []
        total_start = time.perf_counter()
        runtime_profile = getattr(
            getattr(self, "config", None), "runtime_profile", "custom"
        )

        social_answer = fast_social_response(user_input, history_messages)
        if social_answer is not None:
            if stream_callback:
                stream_callback(social_answer)
            standalone_query = " ".join(str(user_input or "").split())
            return build_turn_result(
                answer=social_answer,
                basis="CONVERSATION",
                current_query=standalone_query,
                contextual_query=standalone_query,
                query_language=detect_text_language(standalone_query),
                threshold=GROUNDING_THRESHOLD,
                total_started=total_start,
                reason_override="social_fast_path",
                runtime_profile=runtime_profile,
                reranker_backend=getattr(
                    getattr(self, "config", None), "reranker_backend", "bge"
                ),
                context_selection_policy=getattr(
                    getattr(self, "config", None),
                    "context_selection_policy",
                    "legacy_threshold",
                ),
                evidence_gate_backend=getattr(
                    getattr(self, "config", None),
                    "evidence_gate_backend",
                    "legacy_ce",
                ),
                generation_grounding_mode=getattr(
                    getattr(self, "config", None),
                    "generation_grounding_mode",
                    "legacy",
                ),
            )

        # Search the user's words before allowing conversation history to alter
        # reranker precision. A strong raw result means the question already has
        # enough retrieval identity; the resolver would only add latency and can
        # poison an otherwise correct ranking. This score controls query routing,
        # never evidence acceptance or sufficiency.
        standalone_query = " ".join(str(user_input or "").split())
        contextual_query = standalone_query
        rewrite_time = 0.0
        has_user_history = bool(self._bounded_dialogue(
            history_messages,
            max_user_turns=getattr(
                getattr(self, "config", None), "history_messages", 3
            ),
        ))
        decomposition_plan = None
        if not has_user_history:
            decomposition_plan = self.decompose_question(contextual_query)

        if decomposition_plan is not None and decomposition_plan.is_decomposed:
            (
                context,
                sections,
                retrieval_time,
                rerank_time,
                best_evidence_score,
                retrieved_documents,
            ) = self.retrieve_decomposed_context(decomposition_plan.subqueries)
        else:
            (
                context,
                sections,
                retrieval_time,
                rerank_time,
                best_evidence_score,
                retrieved_documents,
            ) = self.retrieve_context(standalone_query, standalone_query)

        rewrite_bypass_threshold = getattr(
            getattr(self, "config", None), "rewrite_bypass_threshold", 0.30
        )
        context_selection_policy = getattr(
            getattr(self, "config", None),
            "context_selection_policy",
            "legacy_threshold",
        )
        if context_selection_policy == "rank_only_top2":
            raw_result_is_weak = True
        else:
            raw_result_is_weak = (
                best_evidence_score is None
                or float(best_evidence_score) < rewrite_bypass_threshold
            )
        if has_user_history and raw_result_is_weak:
            _, rewritten_query, rewrite_time = self.condense_question(
                standalone_query, history_messages
            )
            if rewritten_query and rewritten_query != standalone_query:
                contextual_query = rewritten_query
                (
                    context,
                    sections,
                    rewritten_retrieval_time,
                    rewritten_rerank_time,
                    best_evidence_score,
                    retrieved_documents,
                ) = self.retrieve_context(standalone_query, contextual_query)
                retrieval_time = round(
                    retrieval_time + rewritten_retrieval_time, 2
                )
                rerank_time = round(rerank_time + rewritten_rerank_time, 2)

        if has_user_history:
            decomposition_plan = self.decompose_question(contextual_query)
            if decomposition_plan is not None and decomposition_plan.is_decomposed:
                (
                    context,
                    sections,
                    decomposed_retrieval_time,
                    decomposed_rerank_time,
                    best_evidence_score,
                    retrieved_documents,
                ) = self.retrieve_decomposed_context(decomposition_plan.subqueries)
                retrieval_time = round(
                    retrieval_time + decomposed_retrieval_time, 2
                )
                rerank_time = round(rerank_time + decomposed_rerank_time, 2)

        decomposition_time = (
            decomposition_plan.planning_time if decomposition_plan is not None else 0.0
        )
        decomposition_queries = (
            decomposition_plan.subqueries if decomposition_plan is not None else ()
        )
        decomposition_result_kwargs = {
            "decomposition_time": decomposition_time,
            "decomposition_queries": decomposition_queries,
            "decomposition_aspect_scores": (
                getattr(self, "last_decomposition_aspect_scores", [])
                if decomposition_queries else []
            ),
        }

        grounding_threshold = getattr(
            self,
            "last_grounding_threshold",
            getattr(getattr(self, "config", None), "grounding_threshold", GROUNDING_THRESHOLD),
        )
        gate_type = getattr(
            self, "last_gate_type", "cross_encoder_relevance_threshold"
        )
        evidence_sufficiency = getattr(
            self, "last_evidence_sufficiency", "not_independently_measured"
        )
        evidence_query = contextual_query or standalone_query
        query_language = detect_text_language(evidence_query)
        evidence_gate_backend = getattr(
            getattr(self, "config", None), "evidence_gate_backend", "legacy_ce"
        )
        reliability = evaluate_evidence_gate(
            backend=evidence_gate_backend,
            query=evidence_query,
            context=context,
            retrieved_documents=retrieved_documents,
            best_score=best_evidence_score,
            threshold=grounding_threshold,
            semantic_checker=getattr(self, "semantic_answerability_checker", None),
        )
        context_is_reliable = reliability["is_reliable"]
        social_message = False

        if evidence_gate_backend == "single_call_structured":
            if not context_is_reliable:
                answer = grounding_refusal(user_input)
                if stream_callback:
                    stream_callback(answer)
                return build_turn_result(
                    answer=answer,
                    basis="NONE",
                    current_query=standalone_query,
                    contextual_query=contextual_query,
                    query_language=query_language,
                    threshold=None,
                    context=context,
                    sections=sections,
                    selected_documents=retrieved_documents,
                    candidate_documents=getattr(
                        self, "last_candidate_documents", retrieved_documents
                    ),
                    reliability=reliability,
                    rewrite_time=rewrite_time,
                    retrieval_time=retrieval_time,
                    rerank_time=rerank_time,
                    total_started=total_start,
                    gate_type="structural_pre_gate_then_single_call_structured",
                    evidence_sufficiency="structural_reject",
                    runtime_profile=runtime_profile,
                    reranker_backend=getattr(
                        getattr(self, "config", None), "reranker_backend", "bge"
                    ),
                    context_selection_policy=context_selection_policy,
                    evidence_gate_backend=evidence_gate_backend,
                    generation_grounding_mode=getattr(
                        getattr(self, "config", None),
                        "generation_grounding_mode",
                        "legacy",
                    ),
                    **decomposition_result_kwargs,
                )

            structured_sections = [
                next(
                    (
                        document.get(key, "")
                        for key in ("detail_section", "section", "main_section")
                        if document.get(key, "")
                    ),
                    "Kılavuz",
                )
                for document in retrieved_documents
            ]
            generation = AnswerGenerator(
                self.client,
                self.config,
                think_mode=self.think_mode,
            ).generate_uncertain_fallback(
                user_input,
                history_messages,
                context,
                structured_sections,
            )
            if thinking_callback and generation.thought_process:
                thinking_callback(generation.thought_process)
            if stream_callback:
                stream_callback(generation.answer)
            final_reliability = dict(reliability)
            final_reliability["is_reliable"] = generation.basis == "MANUAL"
            final_reliability["semantic_decision"] = generation.structured_kind
            if generation.parser_failure:
                final_reliability["reason"] = "single_call_parser_failure"
                sufficiency = "parser_failure"
            else:
                final_reliability["reason"] = (
                    f"single_call_{generation.structured_kind or 'none'}"
                )
                sufficiency = generation.structured_kind or "none"
            return build_turn_result(
                answer=generation.answer,
                basis=generation.basis,
                current_query=standalone_query,
                contextual_query=contextual_query,
                query_language=query_language,
                threshold=None,
                context=context,
                sections=structured_sections,
                selected_documents=retrieved_documents,
                candidate_documents=getattr(
                    self, "last_candidate_documents", retrieved_documents
                ),
                reliability=final_reliability,
                rewrite_time=rewrite_time,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                generation=generation,
                total_started=total_start,
                gate_type="structural_pre_gate_then_single_call_structured",
                evidence_sufficiency=sufficiency,
                runtime_profile=runtime_profile,
                reranker_backend=getattr(
                    getattr(self, "config", None), "reranker_backend", "bge"
                ),
                context_selection_policy=context_selection_policy,
                evidence_gate_backend=evidence_gate_backend,
                generation_grounding_mode=getattr(
                    getattr(self, "config", None),
                    "generation_grounding_mode",
                    "legacy",
                ),
                **decomposition_result_kwargs,
            )

        if evidence_gate_backend == "conservative_hybrid" and not context_is_reliable:
            answer = grounding_refusal(user_input)
            if stream_callback:
                stream_callback(answer)
            return build_turn_result(
                answer=answer,
                basis="NONE",
                current_query=standalone_query,
                contextual_query=contextual_query,
                query_language=query_language,
                threshold=None,
                context=context,
                sections=sections,
                selected_documents=retrieved_documents,
                candidate_documents=getattr(
                    self, "last_candidate_documents", retrieved_documents
                ),
                reliability=reliability,
                rewrite_time=rewrite_time,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                total_started=total_start,
                gate_type="conservative_hybrid",
                evidence_sufficiency=reliability["evidence_sufficiency"],
                runtime_profile=runtime_profile,
                reranker_backend=getattr(
                    getattr(self, "config", None), "reranker_backend", "bge"
                ),
                context_selection_policy=context_selection_policy,
                evidence_gate_backend=evidence_gate_backend,
                generation_grounding_mode=getattr(
                    getattr(self, "config", None),
                    "generation_grounding_mode",
                    "legacy",
                ),
                **decomposition_result_kwargs,
            )

        def uncertain_candidate_context(max_documents=3):
            documents = []
            sections_for_documents = []
            context_parts = []
            seen = set()
            candidates = getattr(
                self, "last_candidate_documents", retrieved_documents
            )
            for document in candidates:
                headings = [
                    document.get("main_section", ""),
                    document.get("section", ""),
                    document.get("detail_section", ""),
                ]
                breadcrumb = " | ".join(filter(None, headings))
                identity = breadcrumb.casefold() or document.get("content", "")
                if identity in seen:
                    continue
                seen.add(identity)
                documents.append(document)
                sections_for_documents.append(
                    next((value for value in reversed(headings) if value), "Kılavuz")
                )
                context_parts.append(
                    f"[Passage {len(documents)} | {breadcrumb}]\n"
                    f"{document.get('content', '')}"
                )
                if len(documents) >= max_documents:
                    break
            return documents, sections_for_documents, "\n\n---\n\n".join(context_parts)

        # CrossEncoder düşük-güven bandında önce kısa ve buffer'lanan bir karar
        # al. Yalnız grounded/conversation kararından sonra ayrı cevap çağrısı
        # canlı akar; doğrulanmamış veya yarıda kesilmiş JSON kullanıcıya
        # gösterilmez.
        sufficiency_review_ceiling = getattr(
            getattr(self, "config", None), "sufficiency_review_ceiling", 0.30
        )
        needs_sufficiency_review = evidence_gate_backend == "legacy_ce" and (
            not context_is_reliable
            or best_evidence_score is None
            or best_evidence_score < sufficiency_review_ceiling
        )
        if needs_sufficiency_review and not social_message:
            review_documents, review_sections, review_context = uncertain_candidate_context()
            fallback_config = replace(
                self.config,
                model_name=self.rewrite_model,
                # The sufficiency decision has its own 128-token JSON budget.
                # Do not carry the legacy 192-token combined-call limit into
                # the separate, user-visible answer generation.
                max_answer_tokens=self.config.max_answer_tokens,
            )
            fallback_generator = AnswerGenerator(
                self.client,
                fallback_config,
                think_mode=self.rewrite_think_mode,
            )
            # Retrieval has already resolved conversation-dependent references
            # such as "bunu" or "ilk konu".  Reusing the raw user message here
            # forced the sufficiency classifier to perform that resolution a
            # second time and could reject directly supporting evidence.
            sufficiency_query = contextual_query or standalone_query
            decision = fallback_generator.classify_uncertain(
                sufficiency_query,
                history_messages,
                review_context,
                review_sections,
                allow_multiple_sources=bool(decomposition_queries),
            )
            if thinking_callback and decision.thought_process:
                thinking_callback(decision.thought_process)

            if decision.structured_kind in {"grounded", "conversation"}:
                if decision.structured_kind == "grounded":
                    chosen_indices = decision.source_indices or (decision.source_index,)
                    chosen_sections = [
                        review_sections[index - 1] for index in chosen_indices
                    ]
                    generation_sections = ["; ".join(chosen_sections)]
                    generation_context = review_context
                    generation_is_reliable = True
                    generation_is_social = False
                else:
                    generation_sections = []
                    generation_context = ""
                    generation_is_reliable = False
                    generation_is_social = True
                live_generation = fallback_generator.generate(
                    question=user_input,
                    history=history_messages,
                    context=generation_context,
                    sections=generation_sections,
                    context_is_reliable=generation_is_reliable,
                    social_message=generation_is_social,
                    contextual_query=contextual_query,
                    current_query=standalone_query,
                    stream_callback=stream_callback,
                    thinking_callback=thinking_callback,
                )
                combined_thinking = "\n".join(filter(None, (
                    decision.thought_process,
                    live_generation.thought_process,
                )))
                fallback = replace(
                    live_generation,
                    generation_time=round(
                        decision.generation_time + live_generation.generation_time, 3
                    ),
                    time_to_first_token=(
                        round(
                            decision.generation_time
                            + live_generation.time_to_first_token,
                            3,
                        )
                        if live_generation.time_to_first_token is not None else None
                    ),
                    input_tokens=decision.input_tokens + live_generation.input_tokens,
                    output_tokens=decision.output_tokens + live_generation.output_tokens,
                    ollama_load_time=round(
                        decision.ollama_load_time + live_generation.ollama_load_time, 3
                    ),
                    thought_process=combined_thinking,
                    structured_kind=decision.structured_kind,
                    parser_failure=decision.parser_failure,
                    parser_error=decision.parser_error,
                    streaming_mode=(
                        "live_after_sufficiency_decision"
                        if stream_callback else "buffered_after_sufficiency_decision"
                    ),
                    source_index=decision.source_index,
                    source_indices=decision.source_indices,
                )
            else:
                fallback = decision
                if stream_callback:
                    stream_callback(fallback.answer)
            final_reliability = dict(reliability)
            final_reliability["is_reliable"] = fallback.basis == "MANUAL"
            final_reliability["semantic_decision"] = decision.structured_kind
            if fallback.basis == "MANUAL":
                final_reliability["reason"] = "llm_sufficiency_review_pass"
            elif fallback.basis == "CONVERSATION":
                final_reliability["reason"] = "conversation_fallback"
            else:
                final_reliability["reason"] = "llm_sufficiency_review_reject"
            return build_turn_result(
                answer=fallback.answer,
                basis=fallback.basis,
                current_query=standalone_query,
                contextual_query=contextual_query,
                query_language=query_language,
                threshold=grounding_threshold,
                context=review_context,
                sections=review_sections,
                selected_documents=review_documents,
                candidate_documents=getattr(
                    self, "last_candidate_documents", retrieved_documents
                ),
                reliability=final_reliability,
                rewrite_time=rewrite_time,
                retrieval_time=retrieval_time,
                rerank_time=rerank_time,
                generation=fallback,
                total_started=total_start,
                reason_override=(
                    "conversation_fallback"
                    if fallback.basis == "CONVERSATION"
                    else final_reliability["reason"]
                ),
                gate_type=f"{gate_type}_then_llm_sufficiency_review",
                runtime_profile=runtime_profile,
                evidence_sufficiency=(
                    "llm_review_pass"
                    if fallback.basis == "MANUAL"
                    else (
                        "not_applicable_conversation"
                        if fallback.basis == "CONVERSATION"
                        else "llm_review_reject"
                    )
                ),
                reranker_backend=getattr(
                    getattr(self, "config", None), "reranker_backend", "bge"
                ),
                context_selection_policy=context_selection_policy,
                evidence_gate_backend=evidence_gate_backend,
                generation_grounding_mode=getattr(
                    getattr(self, "config", None),
                    "generation_grounding_mode",
                    "legacy",
                ),
                **decomposition_result_kwargs,
            )

        generation = AnswerGenerator(
            self.client,
            getattr(self, "config", RuntimeConfig(
                model_name=self.model_name,
                generation_num_ctx=self.generation_num_ctx,
                max_answer_tokens=self.max_answer_tokens,
            )),
            think_mode=self.think_mode,
        ).generate(
            question=user_input,
            history=history_messages,
            context=context,
            sections=sections,
            context_is_reliable=context_is_reliable,
            social_message=social_message,
            contextual_query=contextual_query,
            current_query=standalone_query,
            stream_callback=stream_callback,
            thinking_callback=thinking_callback,
        )
        return build_turn_result(
            answer=generation.answer,
            basis=generation.basis,
            current_query=standalone_query,
            contextual_query=contextual_query,
            query_language=query_language,
            threshold=grounding_threshold,
            context=context,
            sections=sections,
            selected_documents=retrieved_documents,
            candidate_documents=getattr(self, "last_candidate_documents", retrieved_documents),
            reliability=reliability,
            rewrite_time=rewrite_time,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation=generation,
            total_started=total_start,
            gate_type=(
                "conservative_hybrid"
                if evidence_gate_backend == "conservative_hybrid"
                else gate_type
            ),
            evidence_sufficiency=(
                reliability["evidence_sufficiency"]
                if evidence_gate_backend == "conservative_hybrid"
                else evidence_sufficiency
            ),
            runtime_profile=runtime_profile,
            reranker_backend=getattr(
                getattr(self, "config", None), "reranker_backend", "bge"
            ),
            context_selection_policy=context_selection_policy,
            evidence_gate_backend=evidence_gate_backend,
            generation_grounding_mode=getattr(
                getattr(self, "config", None),
                "generation_grounding_mode",
                "legacy",
            ),
            **decomposition_result_kwargs,
        )
