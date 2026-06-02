import joblib
import pandas as pd

from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


def drop_columns(df, columns):
    if not columns:
        return df

    return df.drop(columns=columns)


def encode_data(
    df,
    categorical_columns,
    method
):
    processed_df = df.copy()

    if method == "Ordinal Encoding":

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        processed_df[categorical_columns] = (
            encoder.fit_transform(
                processed_df[categorical_columns]
            )
        )

        joblib.dump(
            encoder,
            "models/encoder.pkl"
        )

        return processed_df

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        categories="auto"
    )

    encoded = encoder.fit_transform(
        processed_df[categorical_columns]
    )

    joblib.dump(
        encoder,
        "models/encoder.pkl"
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(
            categorical_columns
        ),
        index=processed_df.index
    )

    processed_df = processed_df.drop(
        columns=categorical_columns
    )

    processed_df = pd.concat(
        [
            processed_df,
            encoded_df
        ],
        axis=1
    )

    return processed_df


def scale_data(
    df,
    numerical_columns,
    method
):
    if method == "None":
        return df

    processed_df = df.copy()

    if method == "StandardScaler":

        scaler = StandardScaler()

    elif method == "MinMaxScaler":

        scaler = MinMaxScaler()

    else:

        scaler = RobustScaler()

    processed_df[numerical_columns] = (
        scaler.fit_transform(
            processed_df[numerical_columns]
        )
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    return processed_df