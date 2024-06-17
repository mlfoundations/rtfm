import random
import unittest

from rtfm.data import merge_samples


class TestMergeSamples(unittest.TestCase):
    def test_merge_samples(self):
        random.seed(42)
        batch_size = 32
        ids_and_lens = [[i, random.randint(1, 16)] for i in range(batch_size)]
        merged = merge_samples(ids_and_lens, 0.5)

        # check that output list is same length
        self.assertEqual(len(merged), len(ids_and_lens))

        # check that some samples are merged
        merged_ids = [x[0] for x in merged]
        self.assertTrue(len(set(merged_ids)) < batch_size)

        # check that output lengths are identical (this also checks that order is preserved)
        self.assertListEqual([x[1] for x in merged], [x[1] for x in ids_and_lens])

    def test_merge_samples_nomerge(self):
        random.seed(42)
        batch_size = 32
        ids_and_lens = [[i, random.randint(1, 16)] for i in range(batch_size)]
        merged = merge_samples(ids_and_lens, 0.0)
        self.assertListEqual(ids_and_lens, merged)
