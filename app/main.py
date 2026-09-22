import streamlit as st
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from rag_core.logging_config import configure_chat_logger


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
LOG_DIR = PROJECT_DIR / "log"
CHAT_LOG_PATH = LOG_DIR / "chat_logs.log"
FEEDBACK_LOG_PATH = LOG_DIR / "user_feedback_logs.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)
chat_logger = configure_chat_logger(CHAT_LOG_PATH)

# Dependency logging is configured before this import initializes the pipeline.
from rag_pipeline import RAGPipeline  # noqa: E402

# =====================================================
# UI & AYARLAR (SIDEBAR)
# =====================================================
st.set_page_config(
    page_title="1pilot",
    page_icon="🚗"
)

# Sol tarafa bir menü ekliyoruz
with st.sidebar:
    st.header("⚙️ Sistem Ayarları")
    selected_model = st.selectbox(
        "Aktif LLM Modelini Seçin:",
        ["llama3.1", "qwen2:7b", "qwen3:8b", "gpt-oss:20b", "qwen2.5:3b", "gemma3:12b", "qwen3.8:27b", "qwen2.5:14b-instruct"],
        index=3,
    )
    st.caption("gpt-oss:20b daha yüksek yanıt kalitesi için varsayılandır; qwen2.5:3b daha hızlı alternatiftir.")

# Modeli Streamlit'in önbelleğinde (cache) tutuyoruz
@st.cache_resource
def load_pipeline(model_name):
    # Bu fonksiyon sadece model değiştiğinde 1 kez çalışır
    pipeline = RAGPipeline.from_profile("production_bge", model_name=model_name)
    pipeline.warmup()
    
    return pipeline

# Seçilen modele göre pipeline'ı başlat
rag_pipeline = load_pipeline(selected_model)


# =====================================================
# FEEDBACK
# =====================================================

log_lock = threading.Lock()

def log_feedback(question, answer, source_sections, is_positive):
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "source_sections": source_sections,
        "is_positive": is_positive
    }

    with log_lock:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


# =====================================================
# UI
# =====================================================

st.title("1pilot")

# =====================================================
# SESSION
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# MESSAGE HISTORY
# =====================================================

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # DEBUG EXPANDER
        if msg["role"] == "assistant" and "metadata" in msg:
            debug = msg["metadata"].get("debug_info", {})

            if debug.get("thought_process"):
                with st.expander("🧠 Düşünme Süreci", expanded=False):
                    st.markdown(debug["thought_process"])

            with st.expander("Kullanılan Kaynaklar & Performans"):
                st.write(f"**Intent:** {debug.get('intent')}")
                st.write(f"**Sorgu:** {debug.get('search_query')}")
                st.write(f"**Standalone Question:** {debug.get('standalone_question')}")
                st.write(f"**Query Decomposition:** {debug.get('decomposition_applied', False)}")
                if debug.get("decomposition_queries"):
                    st.write(f"**Atomic Queries:** {debug.get('decomposition_queries')}")
                    st.write(f"**Aspect Relevance:** {debug.get('decomposition_aspect_scores', [])}")
                st.write(f"**Query Language:** {debug.get('query_language')}")
                st.write(f"**Response Language:** {debug.get('response_language')}")
                if debug.get("evidence_query") != debug.get("standalone_question"):
                    st.write(f"**Evidence Query:** {debug.get('evidence_query')}")
                st.write(f"**Reranker Relevance:** {debug.get('best_reranker_relevance')}")
                st.write(f"**Runtime Profile:** {debug.get('runtime_profile')}")
                st.write(f"**Reranker Backend:** {debug.get('reranker_backend')}")
                st.write(f"**Context Selection:** {debug.get('context_selection_policy')}")
                if debug.get("threshold") is not None:
                    st.write(f"**Legacy CE Threshold:** {debug.get('threshold')}")
                st.write(f"**Generation Gate:** {debug.get('context_reliability_reason')}")
                st.write(f"**Gate Type:** {debug.get('gate_type')}")
                st.write(f"**Evidence Sufficiency:** {debug.get('evidence_sufficiency')}")
                if debug.get("semantic_decision"):
                    st.write(f"**Semantic Decision:** {debug.get('semantic_decision')}")
                if debug.get("grounding_decision"):
                    st.write(f"**Grounding Decision:** {debug.get('grounding_decision')}")
                st.write(f"**Streaming Mode:** {debug.get('streaming_mode')}")
                if debug.get("parser_failure"):
                    st.warning(
                        f"Structured parser failure: {debug.get('parser_error') or 'unknown'}"
                    )
                st.write(f"**Evidence Gate:** {debug.get('evidence_gate_time')} sn")
                st.write(f"**Reranker Margin:** {debug.get('reranker_score_margin')}")
                st.write(
                    f"**Distinctive Term Hits:** {debug.get('distinctive_term_hits')}"
                    f"/{debug.get('distinctive_term_count')}"
                )
                st.write(f"**Retrieval Consensus:** {debug.get('retrieval_consensus')}")
                st.write(f"**Consensus Support Rank:** {debug.get('consensus_support_rank')}")
                st.write(f"**Context Documents:** {debug.get('retrieved_document_count')}")
                violations = debug.get("unsupported_technical_terms", [])
                if violations:
                    st.warning(f"Unsupported Technical Terms: {violations}")
                st.write(f"**Rewrite:** {debug.get('rewrite_time')} sn")
                st.write(f"**Decomposition:** {debug.get('decomposition_time', 0.0)} sn")
                st.write(f"**Retrieval:** {debug.get('retrieval_time')} sn")
                st.write(f"**Rerank:** {debug.get('rerank_time')} sn")
                st.write(f"**Model TTFT (ilk üretilen token):** {debug.get('time_to_first_token')} sn")
                st.write(f"**Generation:** {debug.get('generation_time')} sn")
                st.write(f"**Token/s:** {debug.get('tokens_per_second')}")
                st.write(f"**Input Tokens:** {debug.get('input_tokens')}")
                st.write(f"**Output Tokens:** {debug.get('output_tokens')}")
                st.write(f"**Finish Reason:** {debug.get('finish_reason') or 'unknown'}")
                st.write(f"**Truncated:** {debug.get('truncated', False)}")
                st.write(f"**Total:** {debug.get('total_time')} sn")
                st.write("**Başlıklar:**")
                st.write(debug.get("retrieved_sections", []))
                candidate_sections = debug.get("candidate_sections", [])
                if candidate_sections and candidate_sections != debug.get("retrieved_sections", []):
                    st.write("**Kullanılmayan Aday Başlıklar:**")
                    st.write(candidate_sections)


        # FEEDBACK
        if msg["role"] == "assistant" and msg.get("feedback") is None:
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    log_feedback(
                        msg["metadata"]["query"], msg["content"],
                        msg["metadata"]["debug_info"].get("retrieved_sections", []), True,
                    )
                    st.session_state.messages[i]["feedback"] = True
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    log_feedback(
                        msg["metadata"]["query"], msg["content"],
                        msg["metadata"]["debug_info"].get("retrieved_sections", []), False,
                    )
                    st.session_state.messages[i]["feedback"] = False
                    st.rerun()
        elif msg["role"] == "assistant" and msg.get("feedback") is True:
            st.caption("✅ Geri bildiriminiz kaydedildi")
        elif msg["role"] == "assistant" and msg.get("feedback") is False:
            st.caption("🔧 Geri bildiriminiz kaydedildi")


# =====================================================
# CHAT INPUT
# =====================================================

user_input = st.chat_input("Toyota kılavuzu hakkında bir soru sorun...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.expander("🧠 Düşünme Süreci", expanded=False):
            thinking_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.info("Kaynaklar aranıyor...")
        last_render = [0.0]
        last_thinking_render = [0.0]
        stream_started = [False]

        def render_stream(partial_text):
            if not stream_started[0]:
                stream_started[0] = True
                status_placeholder.empty()
            now = time.monotonic()
            if now - last_render[0] >= 0.04:
                response_placeholder.markdown(partial_text + "▌")
                last_render[0] = now

        def render_thinking(partial_thinking):
            now = time.monotonic()
            if now - last_thinking_render[0] >= 0.04:
                thinking_placeholder.markdown(partial_thinking + "▌")
                last_thinking_render[0] = now

        result = rag_pipeline.answer_question(
            user_input=user_input,
            history_messages=st.session_state.messages[:-1],
            stream_callback=render_stream,
            thinking_callback=render_thinking,
        )
        status_placeholder.empty()

    thought_process = str(result.get("thought_process") or "").strip()
    if thought_process:
        thinking_placeholder.markdown(thought_process)
    else:
        thinking_placeholder.caption("Seçili model bu yanıtta düşünme metni üretmedi.")
    response = result["answer"]

    response_placeholder.markdown(response)

    debug_info = {
        "intent": result["intent"],
        "search_query": result["search_query"],
        "standalone_question": result.get("standalone_question", result["search_query"]),
        "decomposition_applied": result.get("decomposition_applied", False),
        "decomposition_queries": result.get("decomposition_queries", []),
        "decomposition_time": result.get("decomposition_time", 0.0),
        "decomposition_aspect_scores": result.get("decomposition_aspect_scores", []),
        "evidence_query": result.get("evidence_query", result["search_query"]),
        "query_language": result.get("query_language"),
        "response_language": result.get("response_language"),
        "best_distance": result.get("best_distance", "N/A"),
        "best_reranker_score": result.get(
            "best_reranker_score", result.get("best_distance", "N/A")
        ),
        "best_reranker_relevance": result.get(
            "best_reranker_relevance", result.get("best_reranker_score", "N/A")
        ),
        "reranker_backend": result.get("reranker_backend", "bge"),
        "runtime_profile": result.get("runtime_profile", "custom"),
        "context_selection_policy": result.get(
            "context_selection_policy", "legacy_threshold"
        ),
        "threshold": result.get("threshold"),
        "gate_type": result.get("gate_type"),
        "evidence_sufficiency": result.get("evidence_sufficiency"),
        "semantic_decision": result.get("semantic_decision"),
        "grounding_decision": result.get("grounding_decision"),
        "parser_failure": result.get("parser_failure", False),
        "parser_error": result.get("parser_error"),
        "streaming_mode": result.get("streaming_mode", "live"),
        "evidence_gate_time": result.get("evidence_gate_time", 0.0),
        "context_reliability_reason": result.get("context_reliability_reason", "N/A"),
        "context_is_reliable": result.get("context_is_reliable", False),
        "reranker_score_margin": result.get("reranker_score_margin"),
        "distinctive_term_count": result.get("distinctive_term_count", 0),
        "distinctive_term_hits": result.get("distinctive_term_hits", 0),
        "top_distinctive_term_hits": result.get("top_distinctive_term_hits", 0),
        "retrieval_consensus": result.get("retrieval_consensus", False),
        "consensus_support_rank": result.get("consensus_support_rank"),
        "retrieved_document_count": result.get("retrieved_document_count", 0),
        "unsupported_technical_terms": result.get("unsupported_technical_terms", []),
        "rewrite_time": result["rewrite_time"],
        "retrieval_time": result["retrieval_time"],
        "rerank_time": result["rerank_time"],
        "time_to_first_token": result.get("time_to_first_token"),
        "generation_time": result["generation_time"],
        "tokens_per_second": result.get("tokens_per_second", 0.0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "finish_reason": result.get("finish_reason", ""),
        "truncated": result.get("truncated", False),
        "total_time": result["total_time"],
        "retrieved_sections": result.get("used_sections", result["retrieved_sections"]),
        "candidate_sections": result.get("candidate_sections", result["retrieved_sections"]),
        "context": result["context"],
        "thought_process": thought_process,
    }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "metadata": {
                "query": result["search_query"],
                "context": result["context"],
                "answer_basis": result.get("answer_basis"),
                "used_sections": result.get("used_sections", []),
                "debug_info": debug_info
            },
            "feedback": None
        }
    )

    chat_logger.info(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": user_input,
                "answer": response,
                "intent": result["intent"],
                "search_query": result["search_query"],
                "standalone_question": result.get(
                    "standalone_question", result["search_query"]
                ),
                "decomposition_applied": result.get("decomposition_applied", False),
                "decomposition_queries": result.get("decomposition_queries", []),
                "decomposition_time": result.get("decomposition_time", 0.0),
                "decomposition_aspect_scores": result.get(
                    "decomposition_aspect_scores", []
                ),
                "evidence_query": result.get("evidence_query", result["search_query"]),
                "query_language": result.get("query_language"),
                "response_language": result.get("response_language"),
                "answer_basis": result.get("answer_basis"),
                "retrieved_sections": result["retrieved_sections"],
                "used_sections": result.get("used_sections", []),
                "candidate_sections": result.get(
                    "candidate_sections", result["retrieved_sections"]
                ),
                "best_reranker_relevance": result.get("best_reranker_relevance"),
                "reranker_backend": result.get("reranker_backend"),
                "runtime_profile": result.get("runtime_profile"),
                "context_selection_policy": result.get("context_selection_policy"),
                "gate_type": result.get("gate_type"),
                "evidence_sufficiency": result.get("evidence_sufficiency"),
                "semantic_decision": result.get("semantic_decision"),
                "grounding_decision": result.get("grounding_decision"),
                "context_reliability_reason": result.get(
                    "context_reliability_reason"
                ),
                "context_is_reliable": result.get("context_is_reliable", False),
                "parser_failure": result.get("parser_failure", False),
                "parser_error": result.get("parser_error"),
                "streaming_mode": result.get("streaming_mode"),
                "rewrite_time": result["rewrite_time"],
                "retrieval_time": result["retrieval_time"],
                "rerank_time": result["rerank_time"],
                "evidence_gate_time": result.get("evidence_gate_time", 0.0),
                "time_to_first_token": result.get("time_to_first_token"),
                "generation_time": result["generation_time"],
                "finish_reason": result.get("finish_reason", ""),
                "truncated": result.get("truncated", False),
                "total_time": result["total_time"]
            },
            ensure_ascii=False
        )
    )

    st.rerun()
