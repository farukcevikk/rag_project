# rag_pipeline.py 

import time 
import ollama # Langchain yerine doğrudan Ollama'nın resmi kütüphanesi 

try:
    from .rag_core.answering import (
        AnswerGenerator,
        EMPTY_ANSWER_FALLBACK,
        finalize_grounded_answer,
        unsupported_technical_terms,
    )
    from .rag_core.config import RuntimeConfig
    from .rag_core.evidence import (
        GROUNDING_THRESHOLD,
        assess_context_reliability,
        detect_text_language,
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
    from .rag_core.routing import (
        bounded_dialogue as route_bounded_dialogue,
        fast_social_response,
        is_social_message,
        plan_query,
    )
except ImportError:
    # `streamlit run app/main.py` app/ dizinini doğrudan sys.path'e ekler.
    from rag_core.answering import (
        AnswerGenerator,
        EMPTY_ANSWER_FALLBACK,
        finalize_grounded_answer,
        unsupported_technical_terms,
    )
    from rag_core.config import RuntimeConfig
    from rag_core.evidence import (
        GROUNDING_THRESHOLD,
        assess_context_reliability,
        detect_text_language,
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
    from rag_core.routing import (
        bounded_dialogue as route_bounded_dialogue,
        fast_social_response,
        is_social_message,
        plan_query,
    )

# ===================================================== 
# YARDIMCI FONKSİYONLAR
# ===================================================== 

OUT_OF_SCOPE_ANSWER = (
    "Bu asistan 1PAVI kullanım kılavuzu kapsamında yanıt verir. "
    "Bu soru için güvenilir bir kılavuz kanıtım yok."
)

# ===================================================== 
# PIPELINE 
# ===================================================== 

class RAGPipeline: 
    def __init__(
        self,
        model_name="llama3.1",
        rewrite_model_name="gpt-oss:20b",
        generation_num_ctx=3072,
        max_answer_tokens=None,
        warmup_num_ctx=None,
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
        self.config = RuntimeConfig(
            model_name=model_name,
            rewrite_model_name=rewrite_model_name,
            generation_num_ctx=resolved_num_ctx,
            max_answer_tokens=resolved_answer_tokens,
        )
        self.generation_num_ctx = self.config.generation_num_ctx
        self.warmup_num_ctx = resolved_warmup_num_ctx
        self.max_answer_tokens = self.config.max_answer_tokens
        
        # Langchain yerine saf Ollama client'ı başlatıyoruz 
        self.client = ollama.Client(host=self.config.ollama_host)
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
            max_length=512,
            device=self.config.cross_encoder_device,
        )
        
        # SEÇENEK 2: TensorRT/GPU versiyonu (ONNX dosyası gerektirir)
        #self.cross_encoder = TensorRTCrossEncoder(
        #    model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
        #    onnx_path="cross_encoder.onnx"
        #)
        
        # --------------------------------------------------------- 

        self.pipeline_config = { 
            "architecture": "Jetson RAG (Hybrid Retrieval + Batched CrossEncoder)",
            "rewrite_model": f"{self.rewrite_model} (history only)",
            "history_user_turns": self.config.history_messages,
            "has_heuristic_bypass": True,
            "routing": "exact social fast path + LLM history resolution + fail-closed evidence gate",
            "chunk_size": self.config.child_chunk_size,
            "chunk_overlap": self.config.child_chunk_overlap,
            "vector_search_k": self.config.vector_k,
            "rerank_top_k": self.config.context_k,
            "context_selection": "dual-query recall + resolved-query precision + trusted top 2 atomic passages",
            "followup_state": "bounded conversation to minimal standalone query",
            "followup_resolution": "one local LLM rewrite for every non-social turn with history",
            "language_routing": "trusted_same_language_first_with_cross_language_fallback",
            "max_answer_tokens": self.max_answer_tokens,
            "generation_num_ctx": self.generation_num_ctx,
            "condense_num_ctx": self.generation_num_ctx,
            "warmup_num_ctx": self.warmup_num_ctx,
            "think_mode": self.think_mode,
            "grounding_threshold": self.config.grounding_threshold,
            "evidence_gate": "cross_encoder_relevance_threshold_fail_closed",
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
        return (
            result.context,
            result.sections,
            result.retrieval_time,
            result.rerank_time,
            result.best_score,
            result.selected_documents,
        )

    def answer_question(self, user_input, history_messages=None, stream_callback=None):
        history_messages = history_messages or []
        total_start = time.perf_counter()

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
            )

        standalone_query, contextual_query, rewrite_time = self.condense_question(
            user_input, history_messages
        )
        context, sections, retrieval_time, rerank_time, best_evidence_score, retrieved_documents = self.retrieve_context(
            standalone_query, contextual_query
        )

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
        reliability = assess_context_reliability(
            evidence_query,
            best_evidence_score,
            retrieved_documents,
            threshold=grounding_threshold,
        )
        context_is_reliable = reliability["is_reliable"]
        social_message = False

        def bypass_generation(answer, basis):
            if basis == "MANUAL" and sections and "Kaynak/Source:" not in answer:
                answer = f"{answer}\n\nKaynak/Source: {sections[0]}"
            if stream_callback:
                stream_callback(answer)
            return build_turn_result(
                answer=answer,
                basis=basis,
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
                total_started=total_start,
                gate_type=gate_type,
                evidence_sufficiency=evidence_sufficiency,
            )

        # Güvenilir ürün kanıtı ve sosyal konuşma sinyali yoksa küçük modele
        # genel bilgi uydurtma. Bu yol aynı zamanda gereksiz generation'ı atlar.
        if not context_is_reliable and not social_message:
            return bypass_generation(OUT_OF_SCOPE_ANSWER, "NONE")

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
            gate_type=gate_type,
            evidence_sufficiency=evidence_sufficiency,
        )
