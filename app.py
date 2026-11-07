import pickle
import numpy as np
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model and supporting files
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

with open("preprocessing.pkl", "rb") as file:
    config = pickle.load(file)

with open("customer_records.pkl", "rb") as file:
    customer_data = pickle.load(file)


NUMERIC_FIELDS = config["numerical_features"]
CATEGORICAL_FIELDS = config["categorical_features"]
NOMINAL_FIELDS = config["nominal_columns"]

TRAIN_MEDIANS = config["train_medians"]
EDUCATION_MAPPING = config["education_mapping"]


def prepare_customer_input(customer):

    # Select only the raw features used by the notebook
    selected_features = (
        NUMERIC_FIELDS + CATEGORICAL_FIELDS
    )

    row = customer[selected_features].copy()

    row = pd.DataFrame([row])

    # Apply notebook's education mapping
    if "EDUCATION" in CATEGORICAL_FIELDS:

        row["EDUCATION"] = (
            row["EDUCATION"].map(
                EDUCATION_MAPPING
            )
        )

    # Apply notebook's one-hot encoding
    row = pd.get_dummies(
        row,
        columns=NOMINAL_FIELDS,
        dtype=int
    )

    # Match the exact training feature order
    row = row.reindex(
        columns=feature_columns,
        fill_value=0
    )

    row = row.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Use training medians for missing values
    row = row.fillna(
        pd.Series(TRAIN_MEDIANS)
    ).fillna(0)

    return row


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    customer_id = ""

    if request.method == "POST":

        customer_id = request.form.get(
            "customer_id", ""
        ).strip()

        try:

            if not customer_id:
                raise ValueError(
                    "Please enter a Customer ID."
                )

            # Find customer in the cleaned dataset
            matching_customer = customer_data[
                customer_data["PROSPECTID"].astype(str)
                == customer_id
            ]

            if matching_customer.empty:
                raise ValueError(
                    "Customer ID not found in the available dataset."
                )

            customer = matching_customer.iloc[0]

            # Prepare model input
            model_input = prepare_customer_input(
                customer
            )

            # Generate prediction
            prediction_encoded = model.predict(
                model_input
            )

            prediction = label_encoder.inverse_transform(
                np.asarray(
                    prediction_encoded,
                    dtype=int
                )
            )[0]

            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    model_input
                )[0]

                confidence = round(
                    float(np.max(probabilities)) * 100,
                    2
                )

            # Display selected customer details
            customer_details = {
                "Age": customer.get("AGE", "N/A"),
                "Monthly Income": customer.get(
                    "NETMONTHLYINCOME", "N/A"
                ),
                "Education": customer.get(
                    "EDUCATION", "N/A"
                ),
                "Gender": customer.get(
                    "GENDER", "N/A"
                )
            }

            result = {
                "customer_id": customer_id,
                "priority": str(prediction),
                "confidence": confidence,
                "details": customer_details
            }

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        customer_id=customer_id
    )


if __name__ == "__main__":
    app.run(debug=True)