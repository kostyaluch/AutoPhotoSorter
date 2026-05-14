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


if __name__ == "__main__":
    unittest.main()
