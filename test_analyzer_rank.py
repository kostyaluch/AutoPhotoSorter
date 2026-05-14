import unittest
from unittest.mock import patch

import analyzer


class _FakeResponse:
    def __init__(self, response_text: str):
        self._response_text = response_text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self._response_text}


class RankImagesWithOllamaTests(unittest.TestCase):
    def test_rank_response_is_parsed_to_zero_based_indices(self):
        with patch.object(analyzer, "REQUESTS_AVAILABLE", True), \
             patch.object(analyzer, "_encode_image_base64", return_value="b64"), \
             patch.object(analyzer.requests, "post", return_value=_FakeResponse("[2, 1]")):
            order = analyzer.rank_images_with_ollama(
                ["a.jpg", "b.jpg"],
                "http://localhost:11434",
                "llava",
            )

        self.assertEqual(order, [1, 0])

    def test_rank_request_uses_deterministic_generation_options(self):
        with patch.object(analyzer, "REQUESTS_AVAILABLE", True), \
             patch.object(analyzer, "_encode_image_base64", return_value="b64"), \
             patch.object(analyzer.requests, "post", return_value=_FakeResponse("[1, 2]")) as mocked_post:
            analyzer.rank_images_with_ollama(
                ["a.jpg", "b.jpg"],
                "http://localhost:11434",
                "llava",
            )

        payload = mocked_post.call_args.kwargs["json"]
        self.assertEqual(payload["options"], analyzer._RANK_GENERATION_OPTIONS)
        self.assertIn("Do NOT simply return [1, 2, 3, ...]", payload["prompt"])
        self.assertIn("IDEAL MAIN PHOTO", payload["prompt"])
        self.assertIn("ALTERNATIVE MAIN PHOTO", payload["prompt"])
        self.assertIn("smooth visual flow", payload["prompt"])

    def test_rank_prompt_includes_preanalysis_summaries_when_passed(self):
        with patch.object(analyzer, "REQUESTS_AVAILABLE", True), \
             patch.object(analyzer, "_encode_image_base64", return_value="b64"), \
             patch.object(analyzer.requests, "post", return_value=_FakeResponse("[1, 2]")) as mocked_post:
            analyzer.rank_images_with_ollama(
                ["a.jpg", "b.jpg"],
                "http://localhost:11434",
                "llava",
                image_summaries=[
                    {
                        "clip_white_bg_score": 0.81,
                        "clip_single_product_score": 0.75,
                        "clip_collage_score": 0.11,
                        "clip_lifestyle_score": 0.05,
                        "has_text_overlay": False,
                        "is_ideal_main_eligible": True,
                        "is_alternative_main_candidate": False,
                        "rule_warnings": [],
                    },
                    {
                        "clip_white_bg_score": 0.79,
                        "clip_single_product_score": 0.70,
                        "clip_collage_score": 0.10,
                        "clip_lifestyle_score": 0.09,
                        "has_text_overlay": True,
                        "is_ideal_main_eligible": False,
                        "is_alternative_main_candidate": True,
                        "rule_warnings": ["text_overlay_detected"],
                    },
                ],
            )

        payload = mocked_post.call_args.kwargs["json"]
        self.assertIn("PRE-ANALYSIS SIGNALS", payload["prompt"])
        self.assertIn("1. white_bg:high; text_overlay:no", payload["prompt"])
        self.assertIn("2. white_bg:high; text_overlay:yes", payload["prompt"])
        self.assertIn("alternative_candidate:yes", payload["prompt"])


class RuleLayerTests(unittest.TestCase):
    def test_text_overlay_blocks_ideal_main(self):
        rules = analyzer.apply_pre_analysis_rules({
            "white_bg_score": 0.9,
            "has_text": True,
            "clip_text_overlay_score": 0.2,
            "clip_collage_score": 0.1,
            "clip_lifestyle_score": 0.1,
            "clip_white_bg_score": 0.9,
            "clip_single_product_score": 0.9,
        })
        self.assertFalse(rules["is_ideal_main_eligible"])
        self.assertTrue(rules["is_alternative_main_candidate"])
        self.assertEqual(rules["rejection_reason"], "text_overlay_blocks_ideal_main")


if __name__ == "__main__":
    unittest.main()
