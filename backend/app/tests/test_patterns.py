import unittest

from backend.app import patterns


class PatternClassificationTests(unittest.TestCase):
    def test_classifies_permitting_intake_status(self) -> None:
        text = "Checking DP intake status and BP timeline for 77 Parkcrest Lane."
        self.assertEqual(patterns.classify(text), "Permitting intake & status")

    def test_classifies_site_constraints_environmental_review(self) -> None:
        text = "Do we need a riparian assessment and QEP letter for the creek and stormwater plan?"
        self.assertEqual(patterns.classify(text), "Site constraints & environmental review")

    def test_classifies_design_review_comments(self) -> None:
        text = "Which agencies have returned comments and what revisions are blocking circulation?"
        self.assertEqual(patterns.classify(text), "Design revisions & review comments")

    def test_classifies_fire_access_safety(self) -> None:
        text = "Fire requires 6.0m clear width and a swept path turning template."
        self.assertEqual(patterns.classify(text), "Fire access & safety")

    def test_classifies_inspections_closeout(self) -> None:
        text = "Confirm inspection sequence and LOA seals before final issuance."
        self.assertEqual(patterns.classify(text), "Inspections & closeout")

    def test_classifies_expedite_financing_pressure(self) -> None:
        text = "Our lender term sheet expires and carry costs are rising—can we expedite?"
        self.assertEqual(patterns.classify(text), "Expedite & financing pressure")


if __name__ == "__main__":
    unittest.main()
