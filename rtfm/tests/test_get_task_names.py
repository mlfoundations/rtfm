"""
Tests for task names parsing

To run tests: python -m unittest rtfm/tests/test_get_task_names.py -v

"""
import unittest

from rtfm.utils import get_task_names_list


class TestGetTaskNames(unittest.TestCase):
    def test_get_task_names_no_wildcard(self):
        for task in ("adult", "electricity", "california"):
            task_names = get_task_names_list(task)
            self.assertListEqual(task_names, [task])

    def test_get_task_names_with_wildcard(self):
        task_names_no_wildcard = get_task_names_list("acsincome")
        task_names_with_wildcard = get_task_names_list("acs*")
        self.assertTrue(len(task_names_no_wildcard) < len(task_names_with_wildcard))
        self.assertTrue(all(x.startswith("acs") for x in task_names_with_wildcard))
