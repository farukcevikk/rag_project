"""Minimal conversation routing and LLM-based query planning.

Language semantics belong to the model. Python code here only provides a tiny
exact-match social fast path, bounded history, schema validation, and a safe
fallback when the resolver is unavailable.
"""

import json
import time

from .models import QueryPlan


STANDALONE_QUERY_PROMPT = """Rewrite CURRENT_MESSAGE as a minimal standalone
retrieval query. Use bounded conversation only to resolve a subject, object,
comparison target, ordinal reference, or requested transformation that
CURRENT_MESSAGE omits. If CURRENT_MESSAGE already names its subject and request
clearly, return it unchanged. Preserve the current request; do not carry an older
request, greeting, personal detail, or answer into the query. Keep the current
language. Do not answer the question.

Examples:
HISTORY: USER: Alpha cihazını açıkla.
CURRENT_MESSAGE: Hangi türleri destekler?
OUTPUT: {"standalone_query":"Alpha cihazı hangi türleri destekler?"}

HISTORY: USER: Alpha cihazını açıkla.
CURRENT_MESSAGE: Beta sistemi nedir?
OUTPUT: {"standalone_query":"Beta sistemi nedir?"}

HISTORY: USER: Alpha ve Beta nedir? Tablo yap.
CURRENT_MESSAGE: Bunlar nasıl haberleşiyor?
OUTPUT: {"standalone_query":"Alpha ve Beta nasıl haberleşiyor?"}

HISTORY: USER: Alpha varyantını açıkla.
CURRENT_MESSAGE: Peki diğer varyant?
OUTPUT: {"standalone_query":"Diğer Alpha varyantını açıkla."}

HISTORY: USER: Alpha konusu. USER: Beta konusu.
CURRENT_MESSAGE: İlk konuyu özetle.
OUTPUT: {"standalone_query":"Alpha konusunu özetle."}

Return only JSON: {"standalone_query":"..."}."""


# Deliberately small and exact. This is a GPU-saving shortcut, not a natural
# language intent classifier. Compound messages must continue to the RAG path.
SOCIAL_FAST_RESPONSES = {
    "merhaba": "Merhaba, nasıl yardımcı olabilirim?",
    "selam": "Merhaba, nasıl yardımcı olabilirim?",
    "günaydın": "Günaydın, nasıl yardımcı olabilirim?",
    "iyi günler": "İyi günler, nasıl yardımcı olabilirim?",
    "iyi akşamlar": "İyi akşamlar, nasıl yardımcı olabilirim?",
    "iyi geceler": "İyi geceler.",
    "nasılsın": "İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
    "nasılsınız": "İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
    "nasıl gidiyor": "İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
    "teşekkürler": "Rica ederim. Başka nasıl yardımcı olabilirim?",
    "teşekkür ederim": "Rica ederim. Başka nasıl yardımcı olabilirim?",
    "sağ ol": "Rica ederim. Başka nasıl yardımcı olabilirim?",
    "sağ olun": "Rica ederim. Başka nasıl yardımcı olabilirim?",
    "görüşürüz": "Görüşmek üzere.",
    "hoşça kal": "Görüşmek üzere.",
    "hello": "Hello, how can I help?",
    "hi": "Hello, how can I help?",
    "how are you": "I'm well, thank you. How can I help?",
    "thanks": "You're welcome. How else can I help?",
    "thank you": "You're welcome. How else can I help?",
    "goodbye": "Goodbye.",
    "bye": "Goodbye.",
}


def normalize_social_message(message):
    """Normalize only casing, surrounding whitespace, and terminal punctuation."""
    return " ".join(str(message or "").casefold().split()).strip(".,!?;: ")


def is_social_message(message):
    return normalize_social_message(message) in SOCIAL_FAST_RESPONSES


def fast_social_response(message, history=None):
    del history  # The exact fast path deliberately has no semantic memory.
    return SOCIAL_FAST_RESPONSES.get(normalize_social_message(message))


def bounded_dialogue(history, max_user_turns=3):
    """Return a size-bounded, role-filtered suffix for the local resolver."""
    valid = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        valid.append({
            "role": role,
            "content": " ".join(str(message.get("content", "")).split())[:500],
        })
    user_indexes = [index for index, item in enumerate(valid) if item["role"] == "user"]
    if not user_indexes:
        return []
    first_user_position = user_indexes[max(0, len(user_indexes) - max_user_turns)]
    selected = valid[first_user_position:]
    user_turn = 0
    for item in selected:
        if item["role"] == "user":
            user_turn += 1
            item["user_turn"] = user_turn
    return selected


def plan_query(
    question,
    history,
    client,
    model_name,
    num_ctx,
    think_mode=False,
    keep_alive="30m",
    max_user_turns=3,
):
    """Let the local LLM produce one minimal standalone retrieval query."""
    started = time.perf_counter()
    question = " ".join(str(question or "").split())
    dialogue = bounded_dialogue(history, max_user_turns=max_user_turns)
    if not dialogue:
        return QueryPlan(question, question, 0.0)

    resolver_input = json.dumps(
        {
            "bounded_conversation": dialogue,
            "current_message": question,
        },
        ensure_ascii=False,
    )
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": STANDALONE_QUERY_PROMPT},
                {"role": "user", "content": resolver_input},
            ],
            format="json",
            think=think_mode,
            options={
                "temperature": 0.0,
                "num_ctx": num_ctx,
                "num_predict": 128,
            },
            keep_alive=keep_alive,
        )
        payload = json.loads(response["message"]["content"])
        if not isinstance(payload, dict):
            raise ValueError("Resolver JSON nesnesi döndürmedi")
        contextual = payload.get("standalone_query")
        if not isinstance(contextual, str):
            raise ValueError("Resolver standalone sorgu döndürmedi")
        contextual = " ".join(contextual.split()).strip()
        if not contextual or len(contextual) > 800:
            raise ValueError("Resolver standalone sorgusu geçersiz")
    except Exception:
        # Never guess which prior turn was intended. Searching the current
        # question alone is predictable; the evidence gate then fails closed.
        contextual = question

    return QueryPlan(question, contextual, round(time.perf_counter() - started, 3))
