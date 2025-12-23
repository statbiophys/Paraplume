"""Run tests on the package."""

import unittest
from pathlib import Path

import pandas as pd
from Paraplume.paraplume.infer import predict_paratope, predict_paratope_seq


class Test(unittest.TestCase):
    def setUp(self):
        # Determine path to tests/data relative to this test file
        test_dir = Path(__file__).parent
        data_dir = test_dir / "data"

        self.df_paired = pd.read_csv(data_dir / "test.csv")
        self.df_heavy = pd.read_csv(data_dir / "test_heavy.csv")
        self.df_light = pd.read_csv(data_dir / "test_light.csv")

    def test_sequence(self):
        heavy_seq = self.df_paired["sequence_heavy"].values[0]
        light_seq = self.df_paired["sequence_light"].values[0]
        for large in [True, False]:
            paratope_heavy, paratope_light = predict_paratope_seq(heavy_seq, light_seq, large=large)
            self.assertIsInstance(paratope_heavy[0], float)
            self.assertIsInstance(paratope_light[0], float)

            paratope_heavy, paratope_light = predict_paratope_seq(
                sequence_heavy=heavy_seq, large=large, single_chain=True
            )
            self.assertIsInstance(paratope_heavy[0], float)

            paratope_heavy, paratope_light = predict_paratope_seq(
                sequence_light=light_seq, large=large, single_chain=True
            )
            self.assertIsInstance(paratope_light[0], float)

    def test_configurations(self):
        for large in [True, False]:
            paratope_heavy = predict_paratope(self.df_heavy, large=large, single_chain=True)
            self.assertIn("model_prediction_heavy", paratope_heavy.columns)
            self.assertIsInstance(paratope_heavy["model_prediction_heavy"].values[0][0], float)

            paratope_light = predict_paratope(self.df_light, large=large, single_chain=True)
            self.assertIn("model_prediction_light", paratope_light.columns)
            self.assertIsInstance(paratope_light["model_prediction_light"].values[0][0], float)

            paratope_paired = predict_paratope(self.df_paired, large=large)
            self.assertTrue(
                "model_prediction_light" in paratope_paired.columns
                and "model_prediction_heavy" in paratope_paired.columns
            )
            self.assertIsInstance(paratope_paired["model_prediction_heavy"].values[0][0], float)
            self.assertIsInstance(paratope_paired["model_prediction_light"].values[0][0], float)


if __name__ == "__main__":
    unittest.main()
