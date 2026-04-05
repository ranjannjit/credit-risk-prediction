import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def load_and_preprocess(path, nrows=100000):
    df = pd.read_csv(path, nrows=100000, low_memory=False)

    # Keep only the two target classes we care about
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

    # Target encoding
    df["loan_status"] = df["loan_status"].map({"Fully Paid": 0, "Charged Off": 1})

    # Drop noisy or high-cardinality features that are not useful for this baseline model
    drop_cols = [
        "id",
        "member_id",
        "emp_title",
        "emp_length",
        "url",
        "desc",
        "title",
        "zip_code",
    ]
    df = df.drop(drop_cols, axis=1, errors="ignore")
    # Drop entirely empty columns before imputation
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != "loan_status"]

    # Drop columns with no observed values before imputation
    numeric_cols = [col for col in numeric_cols if df[col].notna().any()]
    categorical_cols = [col for col in categorical_cols if df[col].notna().any()]

    # Impute numeric and categorical columns separately
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Capture final "pre-dummy" feature lists and a copy of the dataset before one-hot encoding
    all_columns = df.columns.tolist()
    pre_dummies_df = df.copy()

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        df = pd.get_dummies(df, drop_first=True)

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"].astype(int)

    # Keep indices for reproducible cross-module comparisons
    row_indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(row_indices, test_size=0.2, random_state=42)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        X.columns.tolist(),
        all_columns,
        numeric_cols,
        categorical_cols,
        pre_dummies_df,
        train_idx,
        test_idx,
    )


def plot_feature_distribution(df, feature, target=None, bins=30):
    """Draw a plot for a numeric or categorical feature.

    If `target` is provided, draws the feature distribution by target class.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns")

    is_numeric = pd.api.types.is_numeric_dtype(df[feature])

    plt.figure(figsize=(10, 5))
    if is_numeric:
        if target is not None and target in df.columns:
            sns.histplot(
                data=df, x=feature, hue=target, bins=bins, kde=True, stat="density"
            )
        else:
            sns.histplot(df[feature].dropna(), bins=bins, kde=True, stat="density")
        plt.title(f"Numeric distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")

    else:
        order = df[feature].value_counts().index
        if target is not None and target in df.columns:
            sns.countplot(data=df, x=feature, hue=target, order=order)
        else:
            sns.countplot(data=df, x=feature, order=order)
        plt.title(f"Categorical counts of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="Run preprocessing and show column info"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "lending_club.csv"
        ),
        help="Path to CSV file (default: data/lending_club.csv relative to repo root)",
    )
    parser.add_argument(
        "--nrows", type=int, default=100000, help="Number of rows to read from CSV"
    )
    args = parser.parse_args()

    print("Loading and preprocessing data from:", args.csv_path)
    try:
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
        ) = load_and_preprocess(args.csv_path, nrows=args.nrows)
        print("Done preprocessing")
        print("---")
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)
        print("Target classes:", sorted(set(y_train.tolist())))
        print("---")
        print(f"Final feature count after dummies: {len(features)}")
        print("Sample final features:", features[:10])
        print("---")
        print(f"All columns before dummies: {len(all_columns)}")
        print("Numeric cols:", numeric_cols)
        print("Categorical cols:", categorical_cols)

        # Plot one example feature from each type, when available.
        if numeric_cols:
            num_feature = numeric_cols[0]
            print(f"Plotting numeric feature distribution for: {num_feature}")
            plot_feature_distribution(
                df=pd.read_csv(args.csv_path, nrows=100000, low_memory=False),
                feature=num_feature,
                target="loan_status",
            )

        if categorical_cols:
            cat_feature = categorical_cols[0]
            print(f"Plotting categorical feature distribution for: {cat_feature}")
            plot_feature_distribution(
                df=pd.read_csv(args.csv_path, nrows=100000, low_memory=False),
                feature=cat_feature,
                target="loan_status",
            )

    except FileNotFoundError as e:
        print(f"ERROR: file not found: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
