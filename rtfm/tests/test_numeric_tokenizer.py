import unittest

import torch

import rtfm.tokenization.numeric


class TestPeriodicTokenizer(unittest.TestCase):
    def test_periodic_tokenizer(self, d_tokenized=100):
        batch_size = 64
        num_features = 16
        options = rtfm.tokenization.numeric.PeriodicOptions(
            d_tokenized // 2, 1.0, False, "log-linear"
        )
        tokenizer = rtfm.tokenization.numeric.PeriodicTokenizerModule(
            num_features, options
        )
        dummy_batch = torch.randn((batch_size, num_features))
        tokenized = tokenizer(dummy_batch)
        self.assertEqual(list(tokenized.shape), [batch_size, num_features, d_tokenized])


if __name__ == "__main__":
    unittest.main()
