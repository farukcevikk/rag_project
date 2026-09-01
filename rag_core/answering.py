"""Prompt construction, local generation, and answer finalization."""

import re
import time
from dataclasses import dataclass

from .evidence import detect_text_language


EMPTY_ANSWER_FALLBACK = "Şu anda yanıt oluşturamadım. Lütfen tekrar deneyin."
GROUNDING_REFUSAL_TR = (
    "Bu soru için getirilen kılavuz bölümlerinde yeterli ve doğrudan kanıt bulunamadı."
)
GROUNDING_REFUSAL_EN = (
    "The retrieved guide sections do not contain sufficient direct evidence for this question."
)

SYSTEM_PROMPT = """Sen 1PAVI kullanım kılavuzu asistanısın.

Kurallar:
- 1PAVI hakkındaki her iddia, sayı, alan, rota ve işlem adımı yalnız İÇ KANITTA açıkça bulunuyorsa söylenebilir. Tahmin etme.
- Kısaltmaları yalnız İÇ KANITTA açıkça verilen biçimde kullan; kaynakta bulunmayan açılım üretme.
- Kaynakta açıkça geçmeyen protokol, aracı bileşen, veri alanı, tablo adı veya iletişim yönü ekleme.
- İÇ KANIT yalnız mantıksal iş akışını anlatıyorsa fiziksel haberleşme protokolünü bildiğini varsayma.
- Zorunluluk sorularında zorunlu/required ve isteğe bağlı/optional alanları ayrı belirt; alternatifleri birlikte zorunluymuş gibi sunma.
- İÇ KANITTA açıkça bulunmayan garanti, başarı veya sonuç cümlesi ekleme.
- İÇ KANIT sorunun cevabını açıkça içermiyorsa tahmin yürütme; [[BASIS:NONE]] kullan.
- Konuşma geçmişi yalnız söylem bağlamıdır; 1PAVI gerçeği için kanıt değildir.
- İÇ KANIT ve aşağıdaki işaretler özel protokoldür; kullanıcıya gösterme veya açıklama.

Kullanıcının dilinde, doğrudan ve 120 kelimeden kısa yanıt ver. Kaynak yazma.
Yanıtın sonuna mutlaka tam olarak bir işaret ekle:
[[BASIS:MANUAL]], [[BASIS:CONVERSATION]] veya [[BASIS:NONE]]."""

SOCIAL_SYSTEM_PROMPT = """Kısa ve doğal bir sosyal sohbet asistanısın.
Yalnız kullanıcının selamlaşma, tanışma, hal-hatır, teşekkür veya veda mesajına karşılık ver.
Konuşma geçmişinde kullanıcı kendisi hakkında açıkça bilgi verdiyse ilgili soruda bu bilgiyi kullan; tahmin etme.
Kullanıcı önceki cevabı kısaltma, özetleme veya biçimlendirme isterse yalnız o cevabı dönüştür; yeni bilgi ekleme.
Ürün, kılavuz, kanıt veya iç sistemlerden söz etme. Bilgi sorusu yanıtlama.
Kullanıcının dilinde ve iki cümleden kısa yaz.
Yanıtın sonuna tam olarak [[BASIS:CONVERSATION]] ekle; bu işareti açıklama."""


@dataclass(frozen=True)
class GenerationOutcome:
    answer: str
    basis: str
    marker_compliance: bool
    empty_output: bool
    unsupported_technical_terms: list
    generation_time: float
    time_to_first_token: float | None
    input_tokens: int
    output_tokens: int
    finish_reason: str
    truncated: bool
    tokens_per_second: float
    ollama_load_time: float


def finalize_grounded_answer(answer, sections, context_is_reliable):
    basis_match = re.search(r"\[\[BASIS:(MANUAL|CONVERSATION|NONE)\]\]", answer, re.IGNORECASE)
    basis = basis_match.group(1).upper() if basis_match else "NONE"
    clean = re.sub(
        r"\s*\[\[BASIS:(?:MANUAL|CONVERSATION|NONE)\]\]\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    clean = re.sub(r"(?im)^\s*\[Source\s+\d+\s*\|[^\]\n]*\]\s*", "", clean).strip()
    clean = re.sub(r"\n*Kaynak/Source\s*:.*$", "", clean, flags=re.IGNORECASE | re.DOTALL).strip()
    clean = re.sub(
        r"(?im)^\s*(?:MANUAL[_ ]CONTEXT[_ ]STATUS|INTERNAL EVIDENCE|İÇ KANIT)\s*(?::|—).*?(?:\n|$)",
        "",
        clean,
    ).strip()
    if not clean:
        clean = EMPTY_ANSWER_FALLBACK
        basis = "NONE"
    if basis == "MANUAL" and context_is_reliable and sections and clean:
        clean = f"{clean}\n\nKaynak/Source: {sections[0]}"
    return clean, basis, basis_match is not None


def unsupported_technical_terms(answer, context, user_input=""):
    candidate_pattern = re.compile(
        r"(?<!\w)(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9_]{1,}(?:-[A-Za-z0-9_]+)*|`[^`]+`)(?!\w)"
    )
    allowed_text = f"{context}\n{user_input}".casefold()
    ignored = {"manual", "conversation", "none", "basis", "source"}
    unsupported = set()
    for token in candidate_pattern.findall(str(answer or "")):
        normalized = token.strip("`").casefold()
        if normalized not in ignored and normalized not in allowed_text:
            unsupported.add(token.strip("`"))
    return sorted(unsupported, key=str.casefold)


def response_metric(response, name, default=0):
    value = getattr(response, name, None)
    if value is None and hasattr(response, "get"):
        value = response.get(name)
    return default if value is None else value


def grounding_refusal(question):
    return (
        GROUNDING_REFUSAL_EN
        if detect_text_language(question) == "en"
        else GROUNDING_REFUSAL_TR
    )


class AnswerGenerator:
    def __init__(self, client, config, think_mode=False):
        self.client = client
        self.config = config
        self.think_mode = think_mode

    def generate(
        self,
        question,
        history,
        context,
        sections,
        context_is_reliable,
        social_message,
        contextual_query,
        current_query,
        stream_callback=None,
    ):
        if social_message:
            messages = [{"role": "system", "content": SOCIAL_SYSTEM_PROMPT}]
            generation_history = history[-2:]
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.append({
                "role": "system",
                "content": f"İÇ KANIT — güvenilir 1PAVI kılavuz bölümleri:\n{context}",
            })
            generation_history = []
        for message in generation_history:
            role = message.get("role")
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": message.get("content", "")[:700]})
        if not social_message and contextual_query != current_query:
            messages.append({
                "role": "system",
                "content": f"Çözümlenmiş arama niyeti (kanıt değildir): {contextual_query}",
            })
        messages.append({"role": "user", "content": question})

        options = {
            "temperature": 0.0,
            "num_ctx": self.config.generation_num_ctx,
            "num_predict": self.config.max_answer_tokens,
            "repeat_penalty": 1.1,
            "top_k": 10,
            "top_p": 0.8,
        }
        started = time.perf_counter()
        first_token = None
        metric_response = None
        if stream_callback:
            generated_parts = []
            chunks = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                stream=True,
                think=self.think_mode,
                options=options,
                keep_alive=self.config.keep_alive,
            )
            for chunk in chunks:
                metric_response = chunk
                token = chunk["message"]["content"]
                if not token:
                    continue
                if first_token is None:
                    first_token = time.perf_counter() - started
                generated_parts.append(token)
                visible = re.sub(
                    r"\s*\[\[BASIS:.*$",
                    "",
                    "".join(generated_parts),
                    flags=re.IGNORECASE | re.DOTALL,
                )
                stream_callback(visible)
            generated_text = "".join(generated_parts)
        else:
            metric_response = self.client.chat(
                model=self.config.model_name,
                messages=messages,
                think=self.think_mode,
                options=options,
                keep_alive=self.config.keep_alive,
            )
            generated_text = metric_response["message"]["content"]

        answer, basis, marker_compliance = finalize_grounded_answer(
            generated_text.strip(), sections, context_is_reliable
        )
        violations = unsupported_technical_terms(generated_text, context, question)
        output_tokens = int(response_metric(metric_response, "eval_count", 0))
        input_tokens = int(response_metric(metric_response, "prompt_eval_count", 0))
        eval_duration = int(response_metric(metric_response, "eval_duration", 0))
        finish_reason = str(response_metric(metric_response, "done_reason", "") or "")
        truncated = finish_reason.casefold() == "length" or (
            not finish_reason and output_tokens >= self.config.max_answer_tokens
        )
        expected_basis = "CONVERSATION" if social_message else "MANUAL"
        if truncated or (marker_compliance and basis != expected_basis):
            # Explicit NONE and a hard token cutoff are fail-closed. Missing
            # markers remain observable but cannot be a hard gate because some
            # local models finish valid grounded answers without the marker.
            answer = grounding_refusal(question)
            basis = "NONE"
        elif not marker_compliance:
            basis = expected_basis
        if basis == "MANUAL" and sections and "Kaynak/Source:" not in answer:
            answer = f"{answer}\n\nKaynak/Source: {sections[0]}"
        tokens_per_second = (
            round(output_tokens / (eval_duration / 1e9), 2)
            if output_tokens and eval_duration else 0.0
        )
        return GenerationOutcome(
            answer=answer,
            basis=basis,
            marker_compliance=marker_compliance,
            empty_output=not bool(generated_text.strip()),
            unsupported_technical_terms=violations,
            generation_time=round(time.perf_counter() - started, 3),
            time_to_first_token=round(first_token, 3) if first_token is not None else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            truncated=truncated,
            tokens_per_second=tokens_per_second,
            ollama_load_time=round(int(response_metric(metric_response, "load_duration", 0)) / 1e9, 3),
        )
