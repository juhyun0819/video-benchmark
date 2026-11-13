"""CPU-only checks for strict scoring and the all-selected denominator."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from day1_compare.evaluate import evaluate, main, parse_answer


def fixtures(count=4):
    questions = [{"question_id": str(i), "question": f"Question {i}",
                  "choices": ["first", "second", "third", "fourth", "fifth"]}
                 for i in range(count)]
    labels = [{"question_id": str(i), "correct_idx": 0} for i in range(count)]
    return questions, labels


def prediction(qid, text="A", status="complete", **kwargs):
    return {"question_id": str(qid), "status": status, "prediction": text, **kwargs}


class StrictParserTests(unittest.TestCase):
    def test_accepts_only_letter_or_answer_prefix_with_punctuation(self):
        for text in ("A", " a ", "(B)", "C.", "**D**", "Answer: E", "Answer: A.", "\"B\"", "【C】"):
            with self.subTest(text=text):
                self.assertIsNotNone(parse_answer(text))

    def test_rejects_narrative_multiple_answers_and_out_of_range(self):
        for text in ("I choose A", "The answer is A", "A because it is visible", "A/B", "A or B",
                     "A\nB", "ABC", "option A", "Answer: A then B", "Answer: A. Explanation",
                     "F", "", None, 1, "Answer: 1"):
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text))
        self.assertIsNone(parse_answer("E", num_choices=4))


class ScoringTests(unittest.TestCase):
    def test_all_selected_denominator_and_paired_counts(self):
        questions, labels = fixtures()
        on = [prediction(0), prediction(1), prediction(2, "B"), prediction(3, "B")]
        off = [prediction(0), prediction(1, "B"), prediction(2), prediction(3, "B")]
        summary, rows = evaluate(questions, labels, on, off)
        self.assertEqual(summary["arms"]["on"]["accuracy_all_selected"], 0.5)
        for key in ("both_correct", "fixed_by_on", "broken_by_on", "neither_correct"):
            self.assertEqual(summary["paired"][key], 1)
        self.assertEqual(summary["paired"]["accuracy_delta_on_minus_off"], 0)
        self.assertEqual([row["question_id"] for row in rows], ["0", "1", "2", "3"])
        self.assertEqual(rows[0]["on"]["raw_prediction_record"], on[0])

    def test_failures_missing_and_truncated_stay_in_denominator(self):
        questions, labels = fixtures(16)
        on = [prediction(0), prediction(1, "A", "truncated"),
              prediction(2, "A", "execution_error"), prediction(3, "I choose A", parsed_answer="A")]
        summary, rows = evaluate(questions, labels, on, [])
        arm = summary["arms"]["on"]
        self.assertEqual(arm["selected"], 16)
        self.assertEqual(arm["complete"], 2)
        self.assertEqual(arm["parsed_complete"], 1)
        self.assertEqual(arm["parsed_any_status"], 3)
        self.assertEqual(arm["correct"], 1)
        self.assertEqual(arm["accuracy_all_selected"], 1 / 16)
        self.assertEqual(arm["status_counts"]["missing"], 12)
        self.assertEqual(rows[4]["on"]["status"], "missing")
        self.assertIsNone(rows[3]["on"]["parsed_answer"])
        self.assertEqual(summary["arms"]["off"]["accuracy_all_selected"], 0)
        self.assertIsNone(summary["arms"]["off"]["accuracy_completed"])

    def test_choices_nested_and_coverage_flags_are_preserved(self):
        questions, labels = fixtures(1)
        questions[0]["choices"] = [["timestamp one", "timestamp two"], ["another timestamp"]]
        questions[0]["input_aliases"] = {"video 1": {"sampled_frame_count": 17}}
        protocol = {"selected_records": [{"question_id": "0", "actual_render_alignment_verified": False,
                                            "coverage": {"sampled_frames": 17}}]}
        summary, rows = evaluate(questions, labels, [prediction(0)], [prediction(0, "E")], protocol)
        self.assertIsNone(rows[0]["off"]["parsed_answer"])
        self.assertEqual(rows[0]["choices"], questions[0]["choices"])
        self.assertTrue(summary["coverage_flags_available"])
        self.assertIs(rows[0]["coverage_and_provisional_flags"]["protocol_record"]["actual_render_alignment_verified"], False)

    def test_invalid_or_stale_ids_and_labels_fail_closed(self):
        questions, labels = fixtures(1)
        cases = [
            (questions * 2, labels, [], []),
            (questions, [], [], []),
            (questions, [{"question_id": "0", "correct_idx": True}], [], []),
            (questions, [{"question_id": "0", "correct_idx": 5}], [], []),
            (questions, labels, [prediction(0), prediction(0)], []),
            (questions, labels, [prediction("unselected")], []),
            (questions, labels, [prediction(0, status="unknown")], []),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                evaluate(*args)

    def test_cli_writes_auditable_outputs_and_refuses_overwrite(self):
        questions, labels = fixtures(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = []
            for name, value in (("questions", questions), ("labels", labels),
                                ("on", [prediction(0)]), ("off", {"predictions": [prediction(0, "B")]})):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value))
                args.extend([f"--{name}", str(path)])
            args.extend(["--output", str(root / "summary")])
            with redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0)
            summary = json.loads((root / "summary" / "summary.json").read_text())
            self.assertEqual(summary["paired"]["fixed_by_on"], 1)
            self.assertEqual(len(summary["input_files"]["labels"]["sha256"]), 64)
            self.assertTrue((root / "summary" / "comparison.json").is_file())
            with self.assertRaises(FileExistsError):
                main(args)


if __name__ == "__main__":
    unittest.main()
