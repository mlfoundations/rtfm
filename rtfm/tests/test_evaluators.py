"""
Tests for the evaluators module and related text-parsing functions.

To run tests: python -m unittest rtfm/tests/test_evaluators.py -v

"""
import unittest

from rtfm.generation_utils import parse_generated_text
from rtfm.special_tokens import QA_SEP_TOKEN, EOC_TOKEN


class TestParseGeneratedText(unittest.TestCase):
    def test_parse_valid_generated_text(self):
        generated_text = f"This is some text. What is the completion?{QA_SEP_TOKEN}Some completion.{EOC_TOKEN}"
        parsed, is_valid = parse_generated_text(generated_text)
        self.assertEqual(parsed, "Some completion.")
        self.assertTrue(is_valid)

    def test_raises_no_sep_token(self):
        generated_text = f"This is some text. What is the completion?Some completion."
        with self.assertRaises(ValueError):
            parse_generated_text(generated_text)

    def test_parse_fewshot_generated_text(self):
        """Check a few-shot completion (both without EOC token and with it) to ensure correct parsing of few-shot."""
        COMPLETION_TEXT = "A COMPLETION!"
        fewshot_text = f"Predict whether this person's annual income is greater than $50,000.00 per year: Choices: No Yes The Max education level achieved is master's degree. The Marital Status is married with civilian spouse. The Age is 44.0. The Worker class is Private. The Occupation is Prof-specialty. The Relationship is Husband. The Race is White. The Sex is Male. The Capital Gain is 0.0. The Capital Loss is 0.0. The Hours worked per week is 40.0. The Native country is United-States. Does this person have an annual income of greater than $50,000.00 per year? No or Yes{QA_SEP_TOKEN}No{EOC_TOKEN} Predict whether this person's annual income is greater than $50,000.00 per year: Choices: No Yes The Max education level achieved is some college. The Marital Status is married with civilian spouse. The Age is 29.0. The Worker class is Private. The Occupation is Adm-clerical. The Relationship is Wife. The Race is White. The Sex is Female. The Capital Gain is 0.0. The Capital Loss is 0.0. The Hours worked per week is 40.0. The Native country is United-States. Does this person have an annual income of greater than $50,000.00 per year? No or Yes{QA_SEP_TOKEN}Yes{EOC_TOKEN} Predict whether this person's annual income is greater than $50,000.00 per year: Choices: No Yes The Max education level achieved is associates degree - vocational. The Marital Status is married with civilian spouse. The Age is 27.0. The Worker class is Private. The Occupation is Adm-clerical. The Relationship is Wife. The Race is White. The Sex is Female. The Capital Gain is 0.0. The Capital Loss is 0.0. The Hours worked per week is 35.0. The Native country is United-States. Does this person have an annual income of greater than $50,000.00 per year? No or Yes{QA_SEP_TOKEN}No{EOC_TOKEN} Predict whether this person's annual income is greater than $50,000.00 per year: Choices: No Yes The Max education level achieved is associates degree - vocational. The Marital Status is married with civilian spouse. The Age is 23.0. The Worker class is Private. The Occupation is Craft-repair. The Relationship is Husband. The Race is White. The Sex is Male. The Capital Gain is 0.0. The Capital Loss is 0.0. The Hours worked per week is 40.0. The Native country is United-States. Does this person have an annual income of greater than $50,000.00 per year? No or Yes{QA_SEP_TOKEN}No{EOC_TOKEN} Predict whether this person's annual income is greater than $50,000.00 per year: Choices: No Yes The Max education level achieved is doctorate. The Marital Status is married with civilian spouse. The Age is 50.0. The Worker class is Self-emp-not-inc. The Occupation is Prof-specialty. The Relationship is Husband. The Race is White. The Sex is Male. The Capital Gain is 0.0. The Capital Loss is 0.0. The Hours worked per week is 60.0. The Native country is United-States. Does this person have an annual income of greater than $50,000.00 per year? No or Yes{QA_SEP_TOKEN}{COMPLETION_TEXT}"

        # Check when the completion lacks an EOC token.
        parsed, is_valid = parse_generated_text(fewshot_text)
        self.assertEqual(parsed, COMPLETION_TEXT)
        self.assertFalse(is_valid)

        # Check when the completion contains an EOC token.
        parsed, is_valid = parse_generated_text(fewshot_text + EOC_TOKEN)
        self.assertEqual(parsed, COMPLETION_TEXT)
        self.assertTrue(is_valid)
        return
