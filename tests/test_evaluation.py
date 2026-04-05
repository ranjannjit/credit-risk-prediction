import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from utils.preprocessing import load_and_preprocess
from utils.evaluation import run_baseline_logistic, train_rnn_model, evaluate_rnn_model


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "loan_status": "Fully Paid",
                "loan_amnt": 10000,
                "term": " 36 months",
                "home_ownership": "RENT",
            },
            {
                "loan_status": "Charged Off",
                "loan_amnt": 15000,
                "term": " 60 months",
                "home_ownership": "MORTGAGE",
            },
            {
                "loan_status": "Fully Paid",
                "loan_amnt": 12000,
                "term": " 36 months",
                "home_ownership": "RENT",
            },
            {
                "loan_status": "Charged Off",
                "loan_amnt": 18000,
                "term": " 60 months",
                "home_ownership": "MORTGAGE",
            },
            {
                "loan_status": "Fully Paid",
                "loan_amnt": 9000,
                "term": " 36 months",
                "home_ownership": "RENT",
            },
        ]
        self.df = pd.DataFrame(self.rows)

    def test_run_baseline_logistic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "sample.csv")
            self.df.to_csv(csv_path, index=False)

            (
                X_train,
                X_test,
                y_train,
                y_test,
                scaler,
                features,
                all_columns,
                numeric_cols,
                categorical_cols,
                pre_dummies_df,
                train_idx,
                test_idx,
            ) = load_and_preprocess(csv_path)
            logistic_res = run_baseline_logistic(X_train, y_train, X_test, y_test)

            self.assertIn("accuracy", logistic_res)
            self.assertIn("roc_auc", logistic_res)
            self.assertIn("model", logistic_res)
            self.assertIsInstance(logistic_res["accuracy"], float)
            self.assertGreaterEqual(logistic_res["accuracy"], 0.0)
            self.assertLessEqual(logistic_res["accuracy"], 1.0)

    def test_rnn_training_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "sample.csv")
            self.df.to_csv(csv_path, index=False)

            (
                X_train,
                X_test,
                y_train,
                y_test,
                scaler,
                features,
                all_columns,
                numeric_cols,
                categorical_cols,
                pre_dummies_df,
                train_idx,
                test_idx,
            ) = load_and_preprocess(csv_path)

            # simple train with few epochs to keep test fast
            rnn_model = train_rnn_model(
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=2,
                batch_size=2,
                lr=1e-2,
            )
            rnn_results = evaluate_rnn_model(rnn_model, X_test, y_test, batch_size=2)

            self.assertIn("accuracy", rnn_results)
            self.assertIn("roc_auc", rnn_results)
            self.assertIn("y_pred", rnn_results)
            self.assertIn("y_prob", rnn_results)
            self.assertIsInstance(rnn_results["accuracy"], float)
            self.assertGreaterEqual(rnn_results["accuracy"], 0.0)
            self.assertLessEqual(rnn_results["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
