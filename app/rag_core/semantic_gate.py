"""Local semantic answerability gate, independent of reranker scores."""

from __future__ import annotations

import json
import time


SEMANTIC_GATE_PROMPT = """You are a strict evidence-sufficiency classifier for a
technical manual assistant. Decide whether the supplied EVIDENCE explicitly
supports every material aspect requested in QUESTION.

Rules:
- Use only EVIDENCE. Do not use outside knowledge or plausible inference.
- An aspect is an independent answer obligation, not every noun or phrase in the
  question. Group entities that belong to one requested list or ordered procedure.
- SUPPORTED: every requested aspect is explicitly supported.
- PARTIAL: at least one requested aspect is supported and at least one is missing.
- UNSUPPORTED: no requested aspect is supported, the question has a false premise,
  or a requested exact value/model/command is absent.
- A passage being about the same topic is not sufficient.
- Semantic equivalence and direct paraphrases count as support; exact wording is
  not required. Do not demand an explicit closed-world statement unless the user
  asks for an exhaustive or exclusive list.
- Do not turn discourse constraints such as "starting from X" or "in order" into
  separate missing aspects when the evidence actually gives that starting point
  or order.
- Preserve procedure order, identifiers, numbers, device models, protocols, and
  field names exactly; do not broaden one term into another.
- Keep each aspect label under eight words and return at most eight aspects.
- Return classification metadata only. Never answer the question.

Return JSON with exactly these fields:
{"decision":"SUPPORTED|PARTIAL|UNSUPPORTED","requested_aspects":["..."],"supported_aspects":["..."],"missing_aspects":["..."]}"""


class OllamaSemanticAnswerabilityChecker:
    """Strict JSON classifier; any runtime or schema failure is fail-closed."""

    DECISIONS = {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}

    def __init__(self, client, *, model, num_ctx, max_tokens, keep_alive):
        self.client = client
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive

    @staticmethod
    def _string_list(payload, key):
        value = payload.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a string list")
        return [" ".join(item.split()) for item in value if item.strip()]

    def evaluate(self, *, query, context):
        started = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SEMANTIC_GATE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"QUESTION": query, "EVIDENCE": context},
                        ensure_ascii=False,
                    ),
                },
            ],
            format="json",
            think=False,
            options={
                "temperature": 0.0,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
            keep_alive=self.keep_alive,
        )
        raw = str(response["message"]["content"] or "").strip()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("semantic gate output must be an object")
        decision = str(payload.get("decision", "")).upper()
        if decision not in self.DECISIONS:
            raise ValueError("invalid semantic gate decision")
        requested = self._string_list(payload, "requested_aspects")
        supported = self._string_list(payload, "supported_aspects")
        missing = self._string_list(payload, "missing_aspects")
        if not requested:
            raise ValueError("requested_aspects cannot be empty")
        if decision == "SUPPORTED" and missing:
            raise ValueError("SUPPORTED cannot contain missing aspects")
        if decision == "PARTIAL" and (not supported or not missing):
            raise ValueError("PARTIAL requires supported and missing aspects")
        if decision == "UNSUPPORTED" and supported:
            raise ValueError("UNSUPPORTED cannot contain supported aspects")
        return {
            "decision": decision,
            "requested_aspects": requested,
            "supported_aspects": supported,
            "missing_aspects": missing,
            "latency_s": round(time.perf_counter() - started, 4),
        }
