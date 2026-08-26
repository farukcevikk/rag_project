import json
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "app"))

from rag_pipeline import (  # noqa: E402
    RAGPipeline,
    assess_context_reliability,
    declared_user_name,
    fast_social_response,
    finalize_grounded_answer,
    is_social_message,
    last_answer_basis,
    unsupported_technical_terms,
)


class ContextReliabilityTests(unittest.TestCase):
    @staticmethod
    def documents(first_score, second_score, first_content, **first_metadata):
        return [
            {
                "reranker_score": first_score,
                "content": first_content,
                **first_metadata,
            },
            {"reranker_score": second_score, "content": "unrelated secondary document"},
        ]

    def test_strict_score_accepts_without_rescue(self):
        result = assess_context_reliability(
            "unknown topic", -1.0, self.documents(-1.0, -2.0, "different content")
        )

        self.assertTrue(result["is_reliable"])
        self.assertEqual(result["reason"], "strict_score")

    def test_low_score_with_distinctive_term_and_margin_is_rescued(self):
        result = assess_context_reliability(
            "Yield % ne anlama gelir?",
            -3.78,
            self.documents(-3.78, -6.18, "The result screen displays the yield percentage."),
        )

        self.assertTrue(result["is_reliable"])
        self.assertEqual(result["reason"], "term_margin_rescue")

    def test_unknown_term_is_not_rescued_by_score_margin_alone(self):
        result = assess_context_reliability(
            "Mongata nedir?",
            -3.16,
            self.documents(-3.16, -5.87, "Architecture and data flow details."),
        )

        self.assertFalse(result["is_reliable"])

    def test_weak_margin_is_not_rescued(self):
        result = assess_context_reliability(
            "Camera retention duration",
            -3.5,
            self.documents(-3.5, -4.0, "Camera retention settings."),
        )

        self.assertFalse(result["is_reliable"])

    def test_dense_and_lexical_consensus_rescues_a_semantic_paraphrase(self):
        result = assess_context_reliability(
            "Kamera eklerken hangi kısımları doldurmalıyım?",
            -3.2,
            self.documents(
                -3.2,
                -3.5,
                "Kamera yapılandırma alanları.",
                vector_rank=1,
                bm25_rank=2,
            ),
        )

        self.assertTrue(result["is_reliable"])
        self.assertEqual(result["reason"], "retrieval_consensus_rescue")
        self.assertTrue(result["retrieval_consensus"])

    def test_single_retriever_support_cannot_trigger_consensus_rescue(self):
        result = assess_context_reliability(
            "Kamera kayıt süresi nedir?",
            -3.2,
            self.documents(
                -3.2,
                -3.5,
                "Kamera yapılandırma alanları.",
                vector_rank=1,
            ),
        )

        self.assertFalse(result["is_reliable"])
        self.assertFalse(result["retrieval_consensus"])

    def test_consensus_without_distinctive_term_cannot_rescue(self):
        result = assess_context_reliability(
            "Mongata nedir?",
            -3.2,
            self.documents(
                -3.2,
                -3.5,
                "Architecture and data flow details.",
                vector_rank=1,
                bm25_rank=1,
            ),
        )

        self.assertFalse(result["is_reliable"])

    def test_consensus_support_can_come_from_selected_diverse_document(self):
        documents = [
            {
                "reranker_score": -3.2,
                "content": "English configuration evidence.",
                "vector_rank": 2,
            },
            {
                "reranker_score": -3.23,
                "content": "Another equivalent English document.",
                "vector_rank": 1,
            },
            {
                "reranker_score": -3.26,
                "content": "Kamera yapılandırma alanları.",
                "vector_rank": 3,
                "bm25_rank": 3,
            },
        ]

        result = assess_context_reliability(
            "Kamera eklerken hangi kısımları doldurmalıyım?",
            -3.2,
            documents,
        )

        self.assertTrue(result["is_reliable"])
        self.assertEqual(result["reason"], "retrieval_consensus_rescue")
        self.assertEqual(result["top_distinctive_term_hits"], 0)
        self.assertEqual(result["distinctive_term_hits"], 1)
        self.assertEqual(result["consensus_support_rank"], 3)


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"message": {"content": json.dumps(self.payload)}}


def pipeline_with(client):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model_name = "test-model"
    pipeline.client = client
    return pipeline


class CondenseQuestionTests(unittest.TestCase):
    def test_independent_question_uses_no_history_or_model_call(self):
        client = FakeClient()
        pipeline = pipeline_with(client)
        history = [
            {"role": "user", "content": "PLC nedir?"},
            {
                "role": "assistant",
                "content": "Yanıt",
                "metadata": {"query": "recursive kirli sorgu kamera line stop"},
            },
        ]

        standalone, retrieval_query, rewrite_time = pipeline.condense_question(
            "Yield yüzdesi nasıl hesaplanır?", history
        )

        self.assertEqual(standalone, "Yield yüzdesi nasıl hesaplanır?")
        self.assertEqual(retrieval_query, standalone)
        self.assertEqual(rewrite_time, 0.0)
        self.assertEqual(client.calls, [])

    def test_discourse_connector_alone_does_not_make_question_dependent(self):
        client = FakeClient()
        pipeline = pipeline_with(client)
        history = [{"role": "user", "content": "Web arayüzü hangi porttadır?"}]

        _, retrieval_query, rewrite_time = pipeline.condense_question(
            "Peki Manager servisinin varsayılan portu nedir?", history
        )

        self.assertEqual(retrieval_query, "Peki Manager servisinin varsayılan portu nedir?")
        self.assertEqual(rewrite_time, 0.0)
        self.assertEqual(client.calls, [])

    def test_resolver_can_select_two_turns_back(self):
        client = FakeClient({
            "depends_on_history": True,
            "referenced_user_turns": [1],
        })
        pipeline = pipeline_with(client)
        history = [
            {"role": "user", "content": "PLC ve GALC nedir?"},
            {"role": "assistant", "content": "İlk yanıt"},
            {"role": "user", "content": "Kamera nasıl eklenir?"},
            {"role": "assistant", "content": "İkinci yanıt"},
        ]

        standalone, retrieval_query, _ = pipeline.condense_question(
            "İlkinin görevlerini tablo halinde göster.", history
        )

        self.assertEqual(standalone, "İlkinin görevlerini tablo halinde göster.")
        self.assertEqual(
            retrieval_query,
            "İlkinin görevlerini tablo halinde göster. PLC ve GALC nedir?",
        )
        self.assertEqual(client.calls, [])

    def test_explicit_two_messages_back_reference(self):
        client = FakeClient()
        pipeline = pipeline_with(client)
        history = [
            {"role": "user", "content": "PLC ve GALC nedir?"},
            {"role": "assistant", "content": "İlk yanıt"},
            {"role": "user", "content": "Kamera nasıl eklenir?"},
            {"role": "assistant", "content": "İkinci yanıt"},
        ]

        _, retrieval_query, _ = pipeline.condense_question(
            "İki mesaj önceki konuyu tablo halinde göster.", history
        )

        self.assertEqual(
            retrieval_query,
            "İki mesaj önceki konuyu tablo halinde göster. PLC ve GALC nedir?",
        )
        self.assertEqual(client.calls, [])

    def test_elliptical_type_followup_uses_previous_user_turn_without_llm(self):
        client = FakeClient()
        pipeline = pipeline_with(client)
        history = [
            {"role": "user", "content": "Kamera nasıl eklenir?"},
            {"role": "assistant", "content": "Kamera kurulum cevabı"},
        ]

        _, retrieval_query, _ = pipeline.condense_question(
            "Hangi türler eklenebilir?", history
        )

        self.assertEqual(
            retrieval_query,
            "Hangi türler eklenebilir? Kamera nasıl eklenir?",
        )
        self.assertEqual(client.calls, [])

    def test_explicit_camera_type_question_remains_independent(self):
        client = FakeClient()
        pipeline = pipeline_with(client)
        history = [{"role": "user", "content": "PLC nasıl yapılandırılır?"}]

        _, retrieval_query, _ = pipeline.condense_question(
            "Hangi tür kameralar eklenebilir?", history
        )

        self.assertEqual(retrieval_query, "Hangi tür kameralar eklenebilir?")
        self.assertEqual(client.calls, [])

    def test_only_last_three_raw_user_turns_are_visible(self):
        client = FakeClient({
            "depends_on_history": True,
            "referenced_user_turns": [1],
        })
        pipeline = pipeline_with(client)
        history = []
        for index in range(1, 5):
            history.extend([
                {"role": "user", "content": f"Soru {index}"},
                {
                    "role": "assistant",
                    "content": f"Yanıt {index}",
                    "metadata": {"query": f"ASLA_KULLANMA_{index}"},
                },
            ])

        _, retrieval_query, _ = pipeline.condense_question("Bunu açıkla.", history)

        resolver_prompt = client.calls[0]["messages"][1]["content"]
        self.assertNotIn("Soru 1", resolver_prompt)
        self.assertNotIn("ASLA_KULLANMA", resolver_prompt)
        self.assertIn("USER_TURN_1: Soru 2", resolver_prompt)
        self.assertEqual(retrieval_query, "Bunu açıkla. Soru 2")

    def test_resolver_failure_has_bounded_non_recursive_fallback(self):
        client = FakeClient(error=RuntimeError("resolver unavailable"))
        pipeline = pipeline_with(client)
        history = [
            {"role": "user", "content": "PLC nedir?"},
            {"role": "assistant", "content": "Yanıt", "metadata": {"query": "KİRLİ"}},
            {"role": "user", "content": "GALC nedir?"},
            {"role": "assistant", "content": "Yanıt", "metadata": {"query": "DAHA_KİRLİ"}},
        ]

        _, retrieval_query, _ = pipeline.condense_question("Bunu tablo yap.", history)

        self.assertEqual(retrieval_query, "Bunu tablo yap. GALC nedir?")
        self.assertNotIn("KİRLİ", retrieval_query)


class GroundedAnswerFinalizationTests(unittest.TestCase):
    def test_internal_status_line_is_not_exposed(self):
        answer, basis, compliant = finalize_grounded_answer(
            "MANUAL_CONTEXT_STATUS: NOT_RELIABLE\n\nKısa yanıt. [[BASIS:CONVERSATION]]",
            [],
            False,
        )

        self.assertEqual(answer, "Kısa yanıt.")
        self.assertEqual(basis, "CONVERSATION")
        self.assertTrue(compliant)

    def test_turkish_internal_evidence_line_is_not_exposed(self):
        answer, _, _ = finalize_grounded_answer(
            "İÇ KANIT — güvenilir bölüm yok.\nKapsam yanıtı. [[BASIS:NONE]]",
            [],
            False,
        )

        self.assertEqual(answer, "Kapsam yanıtı.")

    def test_marker_only_output_gets_nonempty_fallback(self):
        answer, basis, compliant = finalize_grounded_answer(
            "[[BASIS:CONVERSATION]]", [], False
        )

        self.assertTrue(answer)
        self.assertEqual(basis, "NONE")
        self.assertTrue(compliant)

    def test_source_is_added_only_for_reliable_manual_basis(self):
        answer, basis, _ = finalize_grounded_answer(
            "Ürün yanıtı. [[BASIS:MANUAL]]", ["Doğru Bölüm"], True
        )

        self.assertEqual(basis, "MANUAL")
        self.assertIn("Kaynak/Source: Doğru Bölüm", answer)

    def test_unsupported_acronym_is_reported_but_query_terms_are_allowed(self):
        violations = unsupported_technical_terms(
            "PLC, CAN üzerinden GALC ile konuşur.",
            "PLC ve GALC arasındaki belgelenmiş akış.",
            "PLC ve GALC nasıl haberleşir?",
        )

        self.assertEqual(violations, ["CAN"])


class SocialRoutingTests(unittest.TestCase):
    def test_social_speech_acts_are_recognized(self):
        for message in (
            "Merhaba, ben Faruk",
            "Nasıl gidiyor?",
            "Teşekkür ederim",
            "Görüşürüz",
            "Benim adım neydi?",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_social_message(message))

    def test_information_requests_are_not_social(self):
        for message in (
            "Babamın annesi benim neyim olur?",
            "Dolma kalem nasıl üretilir?",
            "Yarınki maç ne olur?",
            "1PAVI nedir?",
        ):
            with self.subTest(message=message):
                self.assertFalse(is_social_message(message))

    def test_explicitly_declared_name_is_read_from_user_history(self):
        history = [
            {"role": "user", "content": "Merhaba, ben Faruk."},
            {"role": "assistant", "content": "Merhaba!"},
        ]

        self.assertEqual(declared_user_name(history), "Faruk")

    def test_name_is_not_inferred_from_assistant_or_arbitrary_sentence(self):
        history = [
            {"role": "assistant", "content": "Adınız Faruk."},
            {"role": "user", "content": "Ben bugün iyiyim."},
        ]

        self.assertIsNone(declared_user_name(history))

    def test_last_architectural_answer_basis_is_read_from_metadata(self):
        history = [{
            "role": "assistant",
            "content": "Yanıt",
            "metadata": {"answer_basis": "conversation"},
        }]

        self.assertEqual(last_answer_basis(history), "CONVERSATION")

    def test_presentation_request_is_detected_as_followup(self):
        history = [{"role": "assistant", "content": "Uzun yanıt"}]

        self.assertTrue(
            RAGPipeline._needs_followup_resolution("Cevabı daha kısa yazar mısın?", history)
        )

    def test_fast_social_response_uses_declared_name(self):
        self.assertEqual(
            fast_social_response("Merhaba, ben Faruk", []),
            "Merhaba Faruk, nasıl yardımcı olabilirim?",
        )

    def test_information_request_has_no_fast_social_response(self):
        self.assertIsNone(fast_social_response("Yield yüzdesi nedir?", []))

    def test_social_fast_path_skips_all_models_and_retrieval(self):
        pipeline = RAGPipeline.__new__(RAGPipeline)

        result = pipeline.answer_question("Merhaba, ben Faruk", [])

        self.assertEqual(result["context_reliability_reason"], "social_fast_path")
        self.assertEqual(result["retrieval_time"], 0.0)
        self.assertEqual(result["rerank_time"], 0.0)
        self.assertEqual(result["generation_time"], 0.0)
        self.assertEqual(result["answer_basis"], "CONVERSATION")
        self.assertEqual(result["retrieved_document_count"], 0)
        self.assertFalse(result["diversity_supplement_used"])


if __name__ == "__main__":
    unittest.main()
