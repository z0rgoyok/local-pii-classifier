import json
import unittest

from evaluate import evaluate
from policy import Classifier, MAX_CHARS


class EmptyModel:
    def predict(self, text, **options):
        return set()


class BrokenModel:
    def predict(self, text, **options):
        raise RuntimeError("must not be exposed")


class PrivatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier(EmptyModel(), EmptyModel())

    def test_sensitive_values_survive_unicode_and_json_transformations(self):
        base = "probe@example.invalid"
        variants = [base, base.replace("@", "&#64;"), "\u200b".join(base),
                    json.dumps({"history": {"before": base}}, ensure_ascii=True)]
        for text in variants:
            with self.subTest(variant=variants.index(text)):
                self.assertEqual("PRIVATE", self.classifier.classify(text)["verdict"])

    def test_raw_json_escapes_are_decoded(self):
        request = '{"body":"\\u0070\\u0072\\u006f\\u0062\\u0065\\u0040example.invalid"}'
        self.assertEqual("PRIVATE", self.classifier.classify(request)["verdict"])

    def test_private_structured_field_at_depth(self):
        text = json.dumps({"activity": {"comments": [{"author": "fixture"}]}})
        self.assertIn("STRUCTURED_PRIVATE_FIELD", self.classifier.classify(text)["reasons"])

    def test_null_identity_is_not_evidence(self):
        text = json.dumps({"assignee": None, "reporter": ""})
        self.assertEqual("PUBLIC", self.classifier.classify(text)["verdict"])

    def test_test_record_identifier_is_private(self):
        self.assertEqual("PRIVATE", self.classifier.classify('QA fixture: orderId="sample-31"')["verdict"])

    def test_feature_requirements_are_not_private_values(self):
        safe = ["Добавить unit tests для кнопки", "Password field must use secure input",
                "Health check returns HTTP 200", "Экран зарплаты должен показывать skeleton"]
        for text in safe:
            self.assertEqual("PUBLIC", self.classifier.classify(text)["verdict"])

    def test_passed_test_counter_is_not_a_password(self):
        self.assertEqual("PUBLIC", self.classifier.classify("Tests passed: 24, skipped: 2")["verdict"])
        self.assertEqual("PRIVATE", self.classifier.classify("DB_PASS=fixture-value-42")["verdict"])

    def test_sensitive_hr_disclosure_is_private(self):
        self.assertEqual("PRIVATE", self.classifier.classify("Сотруднику изменили оклад до 170 тысяч рублей")["verdict"])

    def test_timestamp_is_not_an_ipv6_address(self):
        self.assertEqual("PUBLIC", self.classifier.classify("2026-10-15T12:34:56Z")["verdict"])
        self.assertEqual("PRIVATE", self.classifier.classify("db host: fd00::1234")["verdict"])

    def test_entities_cannot_introduce_hidden_separators(self):
        text = "pro&#8203;be&#64;example.invalid"
        self.assertEqual("PRIVATE", self.classifier.classify(text)["verdict"])

    def test_public_docs_exception_is_narrow(self):
        safe = "https://docs.python.org/3/library/json.html"
        private = [safe + "?token=fictional-value", "https://docs.python.org.evil.invalid/",
                   "https://fixture@docs.python.org/", "https://docs.python.org/" + "00000000-0000-4000-8000-000000000001"]
        self.assertEqual("PUBLIC", self.classifier.classify(safe)["verdict"])
        for url in private:
            self.assertEqual("PRIVATE", self.classifier.classify(url)["verdict"])

    def test_binary_attachment_is_abstention(self):
        answer = self.classifier.classify("opaque", "attachment_binary")
        self.assertEqual("BLOCK", answer["verdict"])
        self.assertIsNone(answer["contains_private"])

    def test_oversized_input_is_not_silently_truncated(self):
        self.assertEqual("BLOCK", self.classifier.classify("x" * (MAX_CHARS + 1))["verdict"])

    def test_json_depth_bound(self):
        nested = json.dumps([[[["safe"]]]])
        for _ in range(40):
            nested = "[" + nested + "]"
        self.assertEqual("BLOCK", self.classifier.classify(nested)["verdict"])

    def test_model_error_is_block_without_exception_text(self):
        answer = Classifier(BrokenModel(), EmptyModel()).classify("review the layout")
        self.assertEqual("BLOCK", answer["verdict"])
        self.assertNotIn("must not be exposed", json.dumps(answer))

    def test_missing_model_is_block(self):
        self.assertEqual("BLOCK", Classifier().classify("safe description")["verdict"])

    def test_binary_metrics_never_count_errors_as_detection(self):
        rows = [{"input": "safe layout", "private": True, "group": "probe"}]
        result = evaluate(Classifier(), rows)
        self.assertEqual(0, result["counts"]["true_positive"])
        self.assertEqual(1, result["counts"]["abstain_positive"])
        self.assertEqual(0, result["recall_supported"])


if __name__ == "__main__":
    unittest.main()
