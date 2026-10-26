"""One name per target language (languages.py)."""

import unittest

from sensai.languages import canonical_language, fold


class CanonicalLanguageTest(unittest.TestCase):
    def test_known_languages_have_one_name(self):
        for written in ("japonais", "Japonais", " JAPONAIS ", "japanese", "nihongo", "日本語"):
            self.assertEqual(canonical_language(written), "japonais", written)
        for written in ("coréen", "Coreen", "korean", "한국어"):
            self.assertEqual(canonical_language(written), "coréen", written)

    def test_other_languages_are_only_lowercased(self):
        self.assertEqual(canonical_language("Swahili  Moderne"), "swahili moderne")
        self.assertEqual(canonical_language(""), "")

    def test_fold(self):
        self.assertEqual(fold("Coréen"), "coreen")
