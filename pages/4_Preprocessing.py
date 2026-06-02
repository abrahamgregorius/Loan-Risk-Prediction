import json
import joblib
import pandas as pd
import streamlit as st

from utils.data_loader import load_data
from utils.preprocessing import (
    drop_columns,
    encode_data,
    scale_data
)

st.title("Data Preprocessing")

df = load_data()

st.subheader("Pipeline Configuration")

categorical_columns = [
    "ed"
]

numerical_columns = [
    "income",
    "age",
    "employ",
    "address",
    "debtinc",
    "creddebt",
    "othdebt"
]

columns_to_drop = st.multiselect(
    "Columns to Remove",
    categorical_columns + numerical_columns,
    default=[]
)

encoding_method = st.selectbox(
    "Encoding Method",
    [
        "Ordinal Encoding",
        "One Hot Encoding"
    ]
)

scaling_method = st.selectbox(
    "Scaling Method",
    [
        "None",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler"
    ]
)

test_size = st.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

st.subheader("Original Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)

if st.button(
    "Apply Preprocessing",
    use_container_width=True
):

    processed_df = df.copy()

    processed_df = drop_columns(
        processed_df,
        columns_to_drop
    )

    remaining_categorical = [
        col
        for col in categorical_columns
        if col in processed_df.columns
    ]

    remaining_numerical = [
        col
        for col in numerical_columns
        if col in processed_df.columns
    ]

    if remaining_categorical:

        processed_df = encode_data(
            processed_df,
            remaining_categorical,
            encoding_method
        )

    if remaining_numerical:

        processed_df = scale_data(
            processed_df,
            remaining_numerical,
            scaling_method
        )

    processed_df.to_parquet(
        "dataset/processed.parquet",
        index=False
    )

    features = [
        col
        for col in processed_df.columns
        if col != "default"
    ]

    joblib.dump(
        features,
        "models/features.pkl"
    )

    preprocessing_meta = {
        "encoding_method":
            encoding_method,

        "scaling_method":
            scaling_method,

        "categorical_columns":
            remaining_categorical,

        "numerical_columns":
            remaining_numerical,

        "dropped_columns":
            columns_to_drop
    }

    joblib.dump(
        preprocessing_meta,
        "models/preprocessing_meta.pkl"
    )

    config = {
        "encoding_method":
            encoding_method,

        "scaling_method":
            scaling_method,

        "test_size":
            test_size,

        "categorical_columns":
            remaining_categorical,

        "numerical_columns":
            remaining_numerical,

        "dropped_columns":
            columns_to_drop,

        "rows":
            len(processed_df),

        "columns":
            len(processed_df.columns)
    }

    with open(
        "dataset/preprocessing_config.json",
        "w"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    st.success(
        "Preprocessed dataset saved successfully."
    )

    st.subheader(
        "Processed Dataset"
    )

    st.dataframe(
        processed_df.head(10),
        use_container_width=True
    )

    st.subheader(
        "Preprocessing Summary"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        f"{len(processed_df):,}"
    )

    col2.metric(
        "Columns",
        len(processed_df.columns)
    )

st.subheader("Saved Dataset")

try:

    processed_df = pd.read_parquet(
        "dataset/processed.parquet"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        f"{len(processed_df):,}"
    )

    col2.metric(
        "Columns",
        len(processed_df.columns)
    )

    st.dataframe(
        processed_df.head(10),
        use_container_width=True
    )

    try:

        with open(
            "dataset/preprocessing_config.json"
        ) as file:

            config = json.load(file)

        st.subheader(
            "Saved Configuration"
        )

        st.dataframe(
            pd.DataFrame({
                "Setting":
                    config.keys(),
                "Value":
                    [str(v) for v in config.values()]
            }),
            use_container_width=True,
            hide_index=True
        )

    except Exception:
        pass

except Exception:

    st.info(
        "No processed dataset found."
    )