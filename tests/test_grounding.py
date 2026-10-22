"""R2: checking the facts of a grounded answer against the lesson chunks it was given."""

import unittest

from sensai.rag.grounding import check_answer, facts, normalize

INTERRO = """Interro n°2 de japonais — vendredi 16 octobre 2026 (consignes de Tanaka-sensei)
L'interro n°2 a lieu le **vendredi 16 octobre 2026 à 14 h, salle B204**. Elle dure **30 minutes**.
| Partie | Points | Exercice |
|---|---|---|
| Vocabulaire | 8 | traduire 8 mots |
| Kana | 4 | recopier 4 mots en hiragana |
Les absents justifiés repassent l'interro le **mardi 20 octobre à 12 h 30**."""

VOYAGE = "Le voyage coûte **1 450 € par élève**. Acompte de 450 € à verser avant le 15 décembre 2026."
CORRIGE = "**Moyenne de la classe : 13,4/20.** Meilleure note : 19,5. Note la plus basse : 6."


class NormalizeTest(unittest.TestCase):
    def test_same_fact_written_differently(self):
        for a, b in (("14 h", "14h00"), ("12 h 30", "12h30"), ("1 450 €", "1450 euros"), ("13,4/20", "13.4 / 20"),
                     ("4 points", "4 pts"), ("1er décembre", "1 decembre"), ("2 heures", "2 h")):
            with self.subTest(a=a):
                self.assertEqual(facts(a), facts(b))

    def test_narrow_spaces_and_accents(self):
        self.assertEqual(normalize("1 450 € en février"), "1450 € en fevrier")

    def test_kinds(self):
        found = dict((key, kind) for kind, key in facts(
            "Le 16 octobre à 14 h en B204, 30 minutes, 1 450 €, 13,4/20, 4 points, 25 %, en 2027, moyenne 13,4."))
        self.assertEqual(found, {"16octobre": "date", "14h": "heure", "b204": "salle", "30min": "durée",
                                 "1450€": "prix", "13.4/20": "note", "4pt": "points", "25%": "pourcentage",
                                 "2027": "année", "13.4": "nombre décimal"})

    def test_lesson_words_and_numbers_are_not_facts(self):
        self.assertEqual(facts("Leçon 3, interro n°2 : 4 hiragana, ippon, 10 hon, sanbon."), [])


class CheckAnswerTest(unittest.TestCase):
    def test_faithful_answer_passes(self):
        answer = "L'interro n°2 a lieu le vendredi 16 octobre à 14h en salle B204 ; elle dure 30 minutes."
        check = check_answer(answer, [INTERRO])
        self.assertEqual(check.unsupported, [])
        self.assertEqual(len(check.checked), 4)
        self.assertIn("vérifié", check.describe())

    def test_invented_facts_are_flagged(self):
        # What the « whole corpus in the prompt » variant answered after truncation.
        answer = "Les absents justifiés repassent l'interro le vendredi 29 janvier 2027 à 9 h, salle C301."
        check = check_answer(answer, [INTERRO])
        self.assertEqual(check.unsupported, ["29janvier", "9h", "c301", "2027"])
        self.assertIn("absent", check.describe())

    def test_points_given_as_bare_numbers_in_a_grid(self):
        self.assertEqual(check_answer("La partie kana vaut 4 points.", [INTERRO]).unsupported, [])
        self.assertEqual(check_answer("La partie kana est notée sur 10 points.", [INTERRO]).unsupported, ["10pt"])

    def test_prices_dates_and_decimals(self):
        ok = check_answer("Le voyage coûte 1450 euros ; acompte de 450 € avant le 15 décembre.", [VOYAGE])
        self.assertEqual(ok.unsupported, [])
        bad = check_answer("Le voyage coûte 1 200 €, acompte avant le 1er décembre.", [VOYAGE])
        self.assertEqual(bad.unsupported, ["1decembre", "1200€"])
        self.assertEqual(check_answer("La moyenne était de 13,4/20, la meilleure 19,5.", [CORRIGE]).unsupported, [])
        self.assertEqual(check_answer("La moyenne était de 12,8/20.", [CORRIGE]).unsupported, ["12.8/20"])

    def test_the_learners_own_message_counts_as_a_source(self):
        check = check_answer("D'accord, rendez-vous le 3 novembre à 10 h.", [VOYAGE], "On révise le 3 novembre à 10 h ?")
        self.assertEqual(check.unsupported, [])

    def test_no_fact_nothing_to_check(self):
        check = check_answer("Neko veut dire chat. Essaie : « Neko ga suki desu ».", [INTERRO])
        self.assertEqual((check.checked, check.unsupported), ([], []))
        self.assertEqual(check.to_dict(), {"checked": 0, "unsupported": []})


if __name__ == "__main__":
    unittest.main()
