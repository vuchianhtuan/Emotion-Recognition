import unittest

import numpy as np

from src.mrmr_selection import preprocess_subject_fft
from src.preprocess import PreprocessConfig, preprocess_subject_for_mrmr


def _make_subject(seed: int = 123, n_trials: int = 1, n_channels: int = 40, n_samples: int = 8064):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_trials, n_channels, n_samples)).astype(np.float32)
    labels = np.full((n_trials, 4), 6.0, dtype=np.float32)
    return {"data": data, "labels": labels}


class TestPreprocessPipeline(unittest.TestCase):
    def test_shape_sanity_one_subject(self):
        subject = _make_subject()
        out = preprocess_subject_fft(subject)
        self.assertGreater(out.shape[0], 0)
        self.assertEqual(out[0][0].shape, (32, 5))
        self.assertEqual(out[0][1].shape, (2,))

    def test_deterministic_with_same_input(self):
        subject = _make_subject(seed=999)
        config = PreprocessConfig()
        out1 = preprocess_subject_for_mrmr(subject, config)
        out2 = preprocess_subject_for_mrmr(subject, config)
        np.testing.assert_allclose(out1[0][0], out2[0][0], rtol=0, atol=0)
        np.testing.assert_array_equal(out1[0][1], out2[0][1])


if __name__ == "__main__":
    unittest.main()
