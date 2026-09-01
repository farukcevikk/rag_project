import streamlit as st
import logging
import json
import threading
import time
import re # DEĞİŞİKLİK: Regex kütüphanesini ekledik
from datetime import datetime
from pathlib import Path

# Doğrudan sınıfı import ediyoruz
from rag_pipeline import RAGPipeline


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
LOG_DIR = PROJECT_DIR / "log"
CHAT_LOG_PATH = LOG_DIR / "chat_logs.log"
FEEDBACK_LOG_PATH = LOG_DIR / "user_feedback_logs.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# UI & AYARLAR (SIDEBAR)
# =====================================================
st.set_page_config(
    page_title="Toyota RAG Asistanı",
    page_icon="🚗"
)

# Sol tarafa bir menü ekliyoruz
with st.sidebar:
    st.header("⚙️ Sistem Ayarları")
    selected_model = st.selectbox(
        "Aktif LLM Modelini Seçin:",
        ["llama3.1", "qwen3:8b", "gpt-oss:20b", "qwen2.5:3b", "gemma3:12b"],
        index=3,  # Jetson için daha düşük gecikmeli varsayılan
    )
    st.caption("Jetson için qwen2.5:3b hızlı varsayılandır; daha büyük modeller daha kaliteli fakat belirgin biçimde yavaştır.")

# Modeli Streamlit'in önbelleğinde (cache) tutuyoruz
@st.cache_resource
def load_pipeline(model_name):
    # Bu fonksiyon sadece model değiştiğinde 1 kez çalışır
    pipeline = RAGPipeline(model_name=model_name)
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
# LOGGING
# =====================================================

logging.basicConfig(
    filename=str(CHAT_LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# =====================================================
# UI
# =====================================================

st.title("1pavi Copilot")

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

            with st.expander("Kullanılan Kaynaklar & Performans"):
                # Mentörününki gibi düşünce sürecini buraya ekliyoruz!
                if debug.get("thought_process"):
                    st.markdown("**🧠 Düşünce Süreci:**")
                    st.info(debug.get("thought_process"))
                    st.divider()

                st.write(f"**Intent:** {debug.get('intent')}")
                st.write(f"**Sorgu:** {debug.get('search_query')}")
                st.write(f"**Standalone Question:** {debug.get('standalone_question')}")
                st.write(f"**Query Language:** {debug.get('query_language')}")
                if debug.get("evidence_query") != debug.get("standalone_question"):
                    st.write(f"**Evidence Query:** {debug.get('evidence_query')}")
                st.write(f"**Evidence Score:** {debug.get('best_reranker_score')}")
                st.write(f"**Threshold:** {debug.get('threshold')}")
                st.write(f"**Generation Gate:** {debug.get('context_reliability_reason')}")
                st.write(f"**Gate Type:** {debug.get('gate_type')}")
                st.write(f"**Evidence Sufficiency:** {debug.get('evidence_sufficiency')}")
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
                st.write(f"**Retrieval:** {debug.get('retrieval_time')} sn")
                st.write(f"**Rerank:** {debug.get('rerank_time')} sn")
                st.write(f"**TTFT (ilk görünür token):** {debug.get('time_to_first_token')} sn")
                st.write(f"**Generation:** {debug.get('generation_time')} sn")
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
        status_placeholder = st.empty()
        status_placeholder.info("Kaynaklar aranıyor...")
        last_render = [0.0]
        stream_started = [False]

        def render_stream(partial_text):
            if not stream_started[0]:
                stream_started[0] = True
                status_placeholder.empty()
            now = time.monotonic()
            if now - last_render[0] >= 0.04:
                response_placeholder.markdown(partial_text + "▌")
                last_render[0] = now

        result = rag_pipeline.answer_question(
            user_input=user_input,
            history_messages=st.session_state.messages[:-1],
            stream_callback=render_stream,
        )
        status_placeholder.empty()

    # ---------------------------------------------------------
    # DEĞİŞİKLİK: GELEN CEVABI XML ETİKETLERİNDEN AYIKLIYORUZ
    # ---------------------------------------------------------
    raw_response = result["answer"]
    
    # 1. Düşünce kısmını (<thought>) yakala ve UI için ayır
    match_thought = re.search(r"<thought>(.*?)</thought>", raw_response, re.DOTALL | re.IGNORECASE)
    thought_process = match_thought.group(1).strip() if match_thought else ""
    
    # 2. Ana cevaptan <thought>...</thought> bloğunu komple sil
    clean_response = re.sub(r"<thought>.*?</thought>", "", raw_response, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Kalan metindeki <response> ve </response> kelimelerini (model tag'i kapatmayı unutsa bile) zorla sil
    clean_response = clean_response.replace("<response>", "").replace("</response>", "").replace("<response\n", "").strip()
    
    response = clean_response
    # ---------------------------------------------------------
    # ---------------------------------------------------------

    response_placeholder.markdown(response)

    debug_info = {
        "intent": result["intent"],
        "search_query": result["search_query"],
        "standalone_question": result.get("standalone_question", result["search_query"]),
        "evidence_query": result.get("evidence_query", result["search_query"]),
        "query_language": result.get("query_language"),
        "best_distance": result.get("best_distance", "N/A"),
        "best_reranker_score": result.get(
            "best_reranker_score", result.get("best_distance", "N/A")
        ),
        "threshold": result.get("threshold", "N/A"),
        "gate_type": result.get("gate_type"),
        "evidence_sufficiency": result.get("evidence_sufficiency"),
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
        "finish_reason": result.get("finish_reason", ""),
        "truncated": result.get("truncated", False),
        "total_time": result["total_time"],
        "retrieved_sections": result.get("used_sections", result["retrieved_sections"]),
        "candidate_sections": result.get("candidate_sections", result["retrieved_sections"]),
        "context": result["context"],
        "thought_process": thought_process # Düşünce sürecini debug sekmesi için kaydediyoruz
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

    logging.info(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": user_input,
                "answer": response,
                "intent": result["intent"],
                "search_query": result["search_query"],
                "retrieved_sections": result["retrieved_sections"],
                "rewrite_time": result["rewrite_time"],
                "retrieval_time": result["retrieval_time"],
                "rerank_time": result["rerank_time"],
                "generation_time": result["generation_time"],
                "finish_reason": result.get("finish_reason", ""),
                "truncated": result.get("truncated", False),
                "total_time": result["total_time"]
            },
            ensure_ascii=False
        )
    )

    st.rerun()
