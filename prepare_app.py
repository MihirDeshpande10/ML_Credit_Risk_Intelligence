import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

FILE_1 = "case_study1.xlsx"
FILE_2 = "case_study2.xlsx"

TARGET = "Approved_Flag"
ID_COL = "PROSPECTID"

EDUCATION_MAPPING = {
    "SSC": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "POST-GRADUATE": 4,
    "OTHERS": 1,
    "PROFESSIONAL": 3
}

CATEGORICAL_CANDIDATES = [
    "MARITALSTATUS",
    "EDUCATION",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2"
]


def main():

    print("Loading customer datasets...")

    df1 = pd.read_excel(FILE_1)
    df2 = pd.read_excel(FILE_2)

    # Same cleaning rules as the notebook
    df1 = df1.copy()
    df2 = df2.copy()

    df1 = df1.loc[
        df1["Age_Oldest_TL"] != -99999
    ].copy()

    columns_to_be_removed = [
        col for col in df2.columns
        if (df2[col] == -99999).sum() > 10000
    ]

    df2 = df2.drop(
        columns=columns_to_be_removed
    ).copy()

    for col in df2.columns:
        df2 = df2.loc[
            df2[col] != -99999
        ].copy()

    # Merge customer datasets
    df = pd.merge(
        df1,
        df2,
        how="inner",
        on=ID_COL,
        validate="one_to_one"
    )

    df = df[
        df[TARGET].isin(["P1", "P2", "P3", "P4"])
    ].copy()

    # Save cleaned records for customer ID lookup
    with open("customer_records.pkl", "wb") as file:
        pickle.dump(df, file)

    # Recreate the same training split
    X_all = df.drop(
        columns=[TARGET, ID_COL]
    )

    y_all = df[TARGET].astype(str)

    X_train_raw, _, y_train, _ = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all
    )

    # Read the exact feature order used by the trained model
    with open("feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)

    # Recover numerical model inputs
    numerical_features = [
        col for col in feature_columns
        if col in X_train_raw.columns
        and pd.api.types.is_numeric_dtype(
            X_train_raw[col]
        )
    ]

    # Recover categorical model inputs
    categorical_features = []
    nominal_columns = []

    for col in CATEGORICAL_CANDIDATES:

        if col not in X_train_raw.columns:
            continue

        if col == "EDUCATION" and col in feature_columns:
            categorical_features.append(col)
            continue

        dummy_columns = pd.get_dummies(
            X_train_raw[[col]],
            columns=[col],
            dtype=int
        ).columns.tolist()

        if any(
            name in feature_columns
            for name in dummy_columns
        ):
            categorical_features.append(col)
            nominal_columns.append(col)

    selected_features = (
        numerical_features + categorical_features
    )

    if not selected_features:
        raise ValueError(
            "Could not recover the model input features."
        )

    # Rebuild training preprocessing to obtain training medians
    X_train = X_train_raw[
        selected_features
    ].copy()

    if "EDUCATION" in selected_features:
        X_train["EDUCATION"] = (
            X_train["EDUCATION"].map(
                EDUCATION_MAPPING
            )
        )

    X_train = pd.get_dummies(
        X_train,
        columns=nominal_columns,
        dtype=int
    )

    X_train = X_train.reindex(
        columns=feature_columns,
        fill_value=0
    )

    train_medians = X_train.median(
        numeric_only=True
    )

    X_train = X_train.fillna(
        train_medians
    ).fillna(0)

    category_options = {}

    for col in categorical_features:
        category_options[col] = sorted(
            X_train_raw[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    config = {
        "feature_columns": feature_columns,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "nominal_columns": nominal_columns,
        "category_options": category_options,
        "train_medians": train_medians.to_dict(),
        "education_mapping": EDUCATION_MAPPING
    }

    with open("preprocessing.pkl", "wb") as file:
        pickle.dump(config, file)

    print("\nSetup completed successfully.")
    print("Customer records:", len(df))
    print("Numerical features:", len(numerical_features))
    print("Categorical features:", categorical_features)
    print("Model input columns:", len(feature_columns))
    print("Created customer_records.pkl")
    print("Created preprocessing.pkl")


if __name__ == "__main__":
    main()