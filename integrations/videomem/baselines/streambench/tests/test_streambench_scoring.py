import unittest
try:
 from baselines.streambench.scoring import score_answer
except ImportError:
 score_answer=None

class ScoringTests(unittest.TestCase):
 def test_mc_letters_and_text_use_explicit_correct_index(self):
  self.assertIsNotNone(score_answer)
  q={'question':'what','choices':['one','two','three'],'correct_idx':2,'correct_answer':'three'}
  self.assertEqual(score_answer('C',q),(True,'multiple_choice'))
  self.assertEqual(score_answer('three',q),(True,'multiple_choice'))
  self.assertEqual(score_answer('B',q),(False,'multiple_choice'))
  self.assertEqual(score_answer('Maybe B or C',q),(False,'multiple_choice'))
 def test_no_label_is_unknown_and_invalid_index_fails(self):
  self.assertIsNotNone(score_answer)
  self.assertEqual(score_answer('yes',{'question':'what'}),(None,None))
  with self.assertRaises(ValueError):score_answer('A',{'choices':['one'],'correct_idx':4})
