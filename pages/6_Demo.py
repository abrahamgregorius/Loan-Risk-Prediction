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

except Exception:

    st.warning(
        "Please train a model first."
    )

    st.stop()

df = load_data()

married_map = {v: i for i, v in enumerate(sorted(df["Married/Single"].astype(str).unique()))}
house_map = {v: i for i, v in enumerate(sorted(df["House_Ownership"].astype(str).unique()))}
car_map = {v: i for i, v in enumerate(sorted(df["Car_Ownership"].astype(str).unique()))}
profession_map = {v: i for i, v in enumerate(sorted(df["Profession"].astype(str).unique()))}

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
            .unique()
            .tolist()
        )
    )

    house = st.selectbox(
        "House Ownership",
        sorted(
            df["House_Ownership"]
            .unique()
            .tolist()
        )
    )

with col2:

    car = st.selectbox(
        "Car Ownership",
        sorted(
            df["Car_Ownership"]
            .unique()
            .tolist()
        )
    )

    profession = st.selectbox(
        "Profession",
        sorted(
            df["Profession"]
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

    input_df = pd.DataFrame(
        {
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
        }
    )

    input_df = pd.DataFrame({
				"Income": [income],
				"Age": [age],
				"Experience": [experience],

				"Married/Single": [married_map.get(married, 0)],
				"House_Ownership": [house_map.get(house, 0)],
				"Car_Ownership": [car_map.get(car, 0)],
				"Profession": [profession_map.get(profession, 0)],

				"CURRENT_JOB_YRS": [current_job_years],
				"CURRENT_HOUSE_YRS": [current_house_years]
		})
    for feature in features:

        if feature not in input_df.columns:
            input_df[feature] = 0

    input_df = input_df[
        features
    ]

    prediction = model.predict(
        input_df
    )[0]

    probability = model.predict_proba(
        input_df
    )[0][1]

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

    st.metric(
        "Default Probability",
        f"{probability:.2%}"
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

        st.write(
            """
            This applicant shows a relatively
            low probability of default based on
            income stability, work experience,
            and ownership characteristics.
            """
        )

        st.success(
            "Suitable for loan approval consideration."
        )

    else:

        st.write(
            """
            This applicant exhibits characteristics
            associated with elevated default risk.
            Additional financial assessment may
            be necessary before approval.
            """
        )

        st.warning(
            "Additional financial assessment is recommended."
        )

    st.subheader(
        "Applicant Summary"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )