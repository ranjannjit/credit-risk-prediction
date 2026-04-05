import os
import tempfile
import unittest
import pandas as pd

from utils.preprocessing import load_and_preprocess


class TestPreprocessing(unittest.TestCase):
    def test_load_and_preprocess_returns_expected_shapes_and_columns(self):
        rows = [
            {
                "loan_status": "Fully Paid",
                "loan_amnt": 10000,
                "funded_amnt": 10000,
                "funded_amnt_inv": 10000,
                "int_rate": 10.99,
                "installment": 210.0,
                "annual_inc": 50000,
                "dti": 10.0,
                "delinq_2yrs": 0,
                "fico_range_low": 660,
                "fico_range_high": 680,
                "inq_last_6mths": 0,
                "open_acc": 5,
                "pub_rec": 0,
                "revol_bal": 2000,
                "revol_util": 30.0,
                "total_acc": 12,
                "out_prncp": 10000,
                "term": " 36 months",
                "home_ownership": "RENT",
                "purpose": "debt_consolidation",
                "verification_status": "Verified",
                "pymnt_plan": "n",
                "issue_d": "Jan-2015",
                "initial_list_status": "f",
            },
            {
                "loan_status": "Charged Off",
                "loan_amnt": 15000,
                "funded_amnt": 15000,
                "funded_amnt_inv": 15000,
                "int_rate": 18.99,
                "installment": 320.0,
                "annual_inc": 75000,
                "dti": 15.0,
                "delinq_2yrs": 1,
                "fico_range_low": 700,
                "fico_range_high": 720,
                "inq_last_6mths": 1,
                "open_acc": 6,
                "pub_rec": 0,
                "revol_bal": 3000,
                "revol_util": 50.0,
                "total_acc": 14,
                "out_prncp": 15000,
                "term": " 60 months",
                "home_ownership": "MORTGAGE",
                "purpose": "credit_card",
                "verification_status": "Source Verified",
                "pymnt_plan": "n",
                "issue_d": "Feb-2016",
                "initial_list_status": "f",
            },
            {
                "loan_status": "Current",
                "loan_amnt": 20000,
                "funded_amnt": 20000,
                "funded_amnt_inv": 20000,
                "int_rate": 15.99,
                "installment": 450.0,
                "annual_inc": 90000,
                "dti": 20.0,
                "delinq_2yrs": 0,
                "fico_range_low": 720,
                "fico_range_high": 740,
                "inq_last_6mths": 2,
                "open_acc": 8,
                "pub_rec": 1,
                "revol_bal": 2500,
                "revol_util": 40.0,
                "total_acc": 18,
                "out_prncp": 20000,
                "term": " 36 months",
                "home_ownership": "RENT",
                "purpose": "home_improvement",
                "verification_status": "Not Verified",
                "pymnt_plan": "n",
                "issue_d": "Mar-2017",
                "initial_list_status": "f",
            },
            {
                "loan_status": "Fully Paid",
                "loan_amnt": 5000,
                "funded_amnt": 5000,
                "funded_amnt_inv": 5000,
                "int_rate": 8.99,
                "installment": 160.0,
                "annual_inc": 40000,
                "dti": 12.0,
                "delinq_2yrs": 0,
                "fico_range_low": 640,
                "fico_range_high": 660,
                "inq_last_6mths": 0,
                "open_acc": 4,
                "pub_rec": 0,
                "revol_bal": 1000,
                "revol_util": 20.0,
                "total_acc": 10,
                "out_prncp": 5000,
                "term": " 36 months",
                "home_ownership": "RENT",
                "purpose": "other",
                "verification_status": "Verified",
                "pymnt_plan": "n",
                "issue_d": "Apr-2014",
                "initial_list_status": "f",
            },
        ]

        df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "sample_loans.csv")
            df.to_csv(csv_path, index=False)

            X_train, X_test, y_train, y_test, scaler, feature_names = (
                load_and_preprocess(csv_path)
            )

            self.assertEqual(X_train.shape[0], 2)
            self.assertEqual(X_test.shape[0], 1)
            self.assertEqual(X_train.shape[1], X_test.shape[1])
            self.assertEqual(X_train.shape[1], len(feature_names))
            self.assertIn("loan_amnt", feature_names)
            self.assertTrue(any(col.startswith("term_") for col in feature_names))
            self.assertTrue(
                any(col.startswith("home_ownership_") for col in feature_names)
            )
            self.assertTrue(set(y_train.unique()).issubset({0, 1}))
            self.assertTrue(set(y_test.unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
