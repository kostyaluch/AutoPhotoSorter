import unittest
from unittest.mock import patch

import sorter


class ProcessFolderAlternativeMainTests(unittest.TestCase):
    @patch.object(sorter, "get_images_in_folder", return_value=["/x/a.jpg", "/x/b.jpg"])
    @patch.object(sorter, "sort_images")
    @patch.object(sorter, "analyze_image")
    def test_marks_alternative_when_text_overlay_rule_candidate(
        self,
        mocked_analyze,
        mocked_sort_images,
        _mocked_get_images,
    ):
        image_a = {
            "path": "/x/a.jpg",
            "filename": "a.jpg",
            "category": "packshot",
            "white_bg_score": 0.9,
            "has_text": True,
            "is_ideal_main_eligible": False,
            "is_alternative_main_candidate": True,
            "is_gallery_candidate": False,
            "error": None,
        }
        image_b = {
            "path": "/x/b.jpg",
            "filename": "b.jpg",
            "category": "packshot",
            "white_bg_score": 0.5,
            "has_text": False,
            "is_ideal_main_eligible": False,
            "is_alternative_main_candidate": False,
            "is_gallery_candidate": True,
            "error": None,
        }
        mocked_analyze.side_effect = [image_a, image_b]
        mocked_sort_images.return_value = (
            [image_a, image_b],
            {"main": [], "packshot": [image_a, image_b], "detail": [], "lifestyle": [], "kit": [], "infographic": []},
        )

        result = sorter.process_folder("/x", api_type="none")

        self.assertFalse(result["has_ideal_main"])
        self.assertTrue(result["has_alternative_main"])
        self.assertEqual(result["alternative_main_image"]["path"], "/x/a.jpg")
        self.assertFalse(result["fallback_used"])


if __name__ == "__main__":
    unittest.main()
