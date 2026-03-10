from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Bank Credit Risk Prediction API is running"

@app.route("/predict")
def predict():

    age = float(request.args.get("age"))
    income = float(request.args.get("income"))
    loan = float(request.args.get("loan"))
    credit = float(request.args.get("credit"))
    employment = float(request.args.get("employment"))
    dti = float(request.args.get("dti"))
    late = float(request.args.get("late"))
    utilization = float(request.args.get("utilization"))
    term = float(request.args.get("term"))

    data = np.array([[age, income, loan, credit, employment, dti, late, utilization, term]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        return "⚠️ High Risk: Loan likely to default"
    else:
        return "✅ Low Risk: Loan likely safe"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)