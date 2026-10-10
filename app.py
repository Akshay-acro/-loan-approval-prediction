from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("loan_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = {
                "ApplicantIncome": request.form.get("ApplicantIncome", ""),
                "CoapplicantIncome": request.form.get("CoapplicantIncome", ""),
                "LoanAmount": request.form.get("LoanAmount", ""),
                "Loan_Amount_Term": request.form.get("Loan_Amount_Term", ""),
                "Credit_History": request.form.get("Credit_History", ""),
                "Employment_Status": request.form.get("Employment_Status", ""),
                "Property_Area": request.form.get("Property_Area", "")
            }

            applicant_income = float(form_data["ApplicantIncome"])
            coapplicant_income = float(form_data["CoapplicantIncome"])
            loan_amount = float(form_data["LoanAmount"])
            loan_term = float(form_data["Loan_Amount_Term"])
            credit_history = int(form_data["Credit_History"])

            if not all([
                form_data["Employment_Status"],
                form_data["Property_Area"]
            ]):
                raise ValueError("Please complete all fields.")

            if min(
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term
            ) < 0:
                raise ValueError("Numeric values cannot be negative.")

            if credit_history not in [0, 1]:
                raise ValueError("Credit history must be 0 or 1.")

            if loan_term <= 0:
                raise ValueError("Loan term must be greater than zero.")

            if loan_amount <= 0:
                raise ValueError("Loan amount must be greater than zero.")

            input_data = pd.DataFrame([{
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Employment_Status": form_data["Employment_Status"],
                "Property_Area": form_data["Property_Area"]
            }])

            result = model.predict(input_data)[0]

            if result == "Y":
                prediction = "Likely Approved"
            else:
                prediction = "Likely Rejected"

        except (ValueError, TypeError):
            error = "Please enter valid values in all fields."
        except Exception:
            error = "Unable to generate a prediction. Please check your inputs."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)