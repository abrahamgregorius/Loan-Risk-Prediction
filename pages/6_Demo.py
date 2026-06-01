import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_data

st.title("Interactive Loan Risk Prediction")

try:

    model = joblib.load(
        "models/best_model.pkl"
    )

    features = joblib.load(
        "models/features.pkl"
    )

    encoder = joblib.load(
        "models/encoder.pkl"
    )

    preprocessing_meta = joblib.load(
        "models/preprocessing_meta.pkl"
    )

    threshold = joblib.load(
        "models/threshold.pkl"
    )

    try:

        scaler = joblib.load(
            "models/scaler.pkl"
        )

    except Exception:

        scaler = None

except Exception as e:

    st.warning(
        f"Missing model files: {e}"
    )

    st.stop()

df = load_data()

categorical_columns = preprocessing_meta[
    "categorical_columns"
]

numerical_columns = preprocessing_meta[
    "numerical_columns"
]

encoding_method = preprocessing_meta[
    "encoding_method"
]

st.write(
    """
    Input applicant information and predict
    whether the applicant is likely to default
    on a loan.
    """
)

col1, col2 = st.columns(2)

with col1:

    income = st.number_input(
        "Income",
        min_value=0,
        value=int(df["Income"].median())
    )

    age = st.number_input(
        "Age",
        min_value=18,
        value=int(df["Age"].median())
    )

    experience = st.number_input(
        "Experience",
        min_value=0,
        value=int(df["Experience"].median())
    )

    married = st.selectbox(
        "Married / Single",
        sorted(
            df["Married/Single"]
            .astype(str)
            .unique()
            .tolist()
        )
    )

    house = st.selectbox(
        "House Ownership",
        sorted(
            df["House_Ownership"]
            .astype(str)
            .unique()
            .tolist()
        )
    )

with col2:

    car = st.selectbox(
        "Car Ownership",
        sorted(
            df["Car_Ownership"]
            .astype(str)
            .unique()
            .tolist()
        )
    )

    profession = st.selectbox(
        "Profession",
        sorted(
            df["Profession"]
            .astype(str)
            .unique()
            .tolist()
        )
    )

    current_job_years = st.number_input(
        "Current Job Years",
        min_value=0,
        value=int(
            df["CURRENT_JOB_YRS"].median()
        )
    )

    current_house_years = st.number_input(
        "Current House Years",
        min_value=0,
        value=int(
            df["CURRENT_HOUSE_YRS"].median()
        )
    )

if st.button(
    "Analyze Risk",
    use_container_width=True
):

    raw_input_df = pd.DataFrame({
        "Income": [income],
        "Age": [age],
        "Experience": [experience],

        "Married/Single": [married],
        "House_Ownership": [house],
        "Car_Ownership": [car],
        "Profession": [profession],

        "CURRENT_JOB_YRS": [
            current_job_years
        ],

        "CURRENT_HOUSE_YRS": [
            current_house_years
        ]
    })

    input_df = raw_input_df.copy()

    try:

        if encoding_method == "Ordinal Encoding":

            input_df[categorical_columns] = (
                encoder.transform(
                    input_df[categorical_columns]
                )
            )

        else:

            encoded = encoder.transform(
                input_df[categorical_columns]
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(
                    categorical_columns
                )
            )

            input_df = input_df.drop(
                columns=categorical_columns
            )

            input_df = pd.concat(
                [
                    input_df.reset_index(drop=True),
                    encoded_df.reset_index(drop=True)
                ],
                axis=1
            )

    except Exception as e:

        st.error(
            f"Encoding error: {e}"
        )

        st.stop()

    try:

        if scaler is not None:

            existing_numerical = [
                col
                for col in numerical_columns
                if col in input_df.columns
            ]

            input_df[existing_numerical] = (
                scaler.transform(
                    input_df[
                        existing_numerical
                    ]
                )
            )

    except Exception as e:

        st.error(
            f"Scaling error: {e}"
        )

        st.stop()

    for feature in features:

        if feature not in input_df.columns:
            input_df[feature] = 0

    input_df = input_df[
        features
    ]

    probability = model.predict_proba(
        input_df
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    st.subheader(
        "Prediction Result"
    )

    if prediction == 0:

        st.success(
            "LOW RISK"
        )

    else:

        st.error(
            "HIGH RISK"
        )

    col1, col2 = st.columns(2)

    col1.metric(
        "Default Probability",
        f"{probability:.2%}"
    )

    col2.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={
                "text":
                "Risk Probability"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "thickness": 0.3
                },
                "steps": [
                    {
                        "range": [0, 40],
                        "color": "lightgreen"
                    },
                    {
                        "range": [40, 70],
                        "color": "gold"
                    },
                    {
                        "range": [70, 100],
                        "color": "salmon"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.subheader(
        "Interpretation"
    )

    if prediction == 0:

        st.success(
            "Applicant classified as LOW RISK."
        )

    else:

        st.warning(
            "Applicant classified as HIGH RISK."
        )

    st.write(
        f"""
        Predicted probability of default: **{probability:.2%}**

        Decision threshold: **{threshold:.2f}**
        """
    )

    st.subheader(
        "Applicant Summary"
    )

    st.dataframe(
        raw_input_df,
        use_container_width=True
    )

    st.subheader(
        "Model Input Features"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )