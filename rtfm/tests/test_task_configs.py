"""
Tests for task configs.

To run tests: python -m unittest rtfm/tests/test_task_configs.py -v
"""

import unittest

from rtfm.task_config import get_tlm_config
from rtfm.task_config.configs import TLMConfig


class TestGetTaskConfigs(unittest.TestCase):
    def _test_get_task(self, task):
        res = get_tlm_config(task)
        self.assertIsInstance(res, TLMConfig)

    def test_get_adult(self):
        self._test_get_task("adult")

    def test_get_california(self):
        res = get_tlm_config("california")
        self.assertIsInstance(res, TLMConfig)

    def test_get_electricity(self):
        res = get_tlm_config("electricity")
        self.assertIsInstance(res, TLMConfig)
