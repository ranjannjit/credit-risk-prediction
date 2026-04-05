
python -m venv venv_credit
venv_credit\Script\activate

python.exe -m pip install --upgrade pip



#Run training from the project root, not from inside models:
cd c:\Users\ranja\Ranjan_Doc\Ranjan\NJIT\ms\Deep Learning\Project\vscode\Credit_Risk_Loan_Default_Prediction

1. python utils/preprocessing.py

python -m models.train

python -m unittest tests.test_preprocessing

uvicorn api.main:app --reload

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"income":50000,"loan_amount":10000,"credit_score":650}'

streamlit run dashboard/app.py