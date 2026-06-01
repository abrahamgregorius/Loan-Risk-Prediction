import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data

df = load_data()

st.title("Exploratory Data Analysis")

st.write("Dataset Source: dataset/loan.csv")

st.subheader("Missing Value Analysis")

missing_df = pd.DataFrame(
    {
        "Feature": df.columns,
        "Missing Values": df.isnull().sum().values
    }
)

st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)

total_missing = int(df.isnull().sum().sum())

if total_missing == 0:
    st.success("No missing values detected in the dataset.")
else:
    st.warning(f"{total_missing:,} missing values detected.")

st.subheader("Descriptive Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

st.subheader("Visualization")

visualization = st.selectbox(
    "Select Visualization",
    [
        "Risk Flag Distribution",
        "Income Distribution",
        "Age Distribution",
        "Experience Distribution",
        "House Ownership Distribution",
        "Car Ownership Distribution",
        "Correlation Heatmap",
        "Income vs Risk Flag",
        "Age vs Risk Flag",
        "Experience vs Risk Flag"
    ]
)

insight = ""

if visualization == "Risk Flag Distribution":

    counts = df["Risk_Flag"].value_counts().sort_index()

    chart_df = pd.DataFrame(
        {
            "Risk": ["Low Risk", "High Risk"],
            "Count": [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        }
    )

    fig = px.bar(
        chart_df,
        x="Risk",
        y="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    low_pct = (
        counts.get(0, 0) / len(df)
    ) * 100

    insight = (
        f"The dataset is imbalanced. "
        f"Approximately {low_pct:.2f}% of applicants "
        f"are classified as low risk."
    )

elif visualization == "Income Distribution":

    fig = px.histogram(
        df,
        x="Income",
        nbins=50
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"Average income is "
        f"{df['Income'].mean():,.0f}."
    )

elif visualization == "Age Distribution":

    fig = px.histogram(
        df,
        x="Age",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average applicant age is "
        f"{df['Age'].mean():.1f} years."
    )

elif visualization == "Experience Distribution":

    fig = px.histogram(
        df,
        x="Experience",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average work experience is "
        f"{df['Experience'].mean():.1f} years."
    )

elif visualization == "House Ownership Distribution":

    ownership = (
        df["House_Ownership"]
        .value_counts()
        .reset_index()
    )

    ownership.columns = [
        "Ownership",
        "Count"
    ]

    fig = px.bar(
        ownership,
        x="Ownership",
        y="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    dominant = ownership.iloc[0]["Ownership"]

    insight = (
        f"The most common house ownership "
        f"status is '{dominant}'."
    )

elif visualization == "Car Ownership Distribution":

    ownership = (
        df["Car_Ownership"]
        .value_counts()
        .reset_index()
    )

    ownership.columns = [
        "Ownership",
        "Count"
    ]

    fig = px.pie(
        ownership,
        names="Ownership",
        values="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    dominant = ownership.iloc[0]["Ownership"]

    insight = (
        f"Most applicants have car ownership "
        f"status '{dominant}'."
    )

elif visualization == "Correlation Heatmap":

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns
        )
    )

    fig.update_layout(height=700)

    st.plotly_chart(fig, use_container_width=True)

    risk_corr = (
        corr["Risk_Flag"]
        .drop("Risk_Flag")
        .abs()
        .sort_values(ascending=False)
    )

    insight = (
        f"The strongest numerical relationship "
        f"with Risk_Flag is "
        f"'{risk_corr.index[0]}'."
    )

elif visualization == "Income vs Risk Flag":

    fig = px.box(
        df,
        x="Risk_Flag",
        y="Income"
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        "Income distribution differs between "
        "risk groups and may contribute to "
        "loan default prediction."
    )

elif visualization == "Age vs Risk Flag":

    fig = px.box(
        df,
        x="Risk_Flag",
        y="Age"
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        "Age patterns vary across risk groups "
        "and may provide predictive value."
    )

elif visualization == "Experience vs Risk Flag":

    fig = px.box(
        df,
        x="Risk_Flag",
        y="Experience"
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        "Work experience may influence the "
        "probability of loan default."
    )

st.subheader("Key Insight")

st.info(insight)

st.subheader("EDA Summary")

total_records = len(df)
high_risk = int(df["Risk_Flag"].sum())
low_risk = total_records - high_risk

st.write(
    f"""
    The dataset contains {total_records:,} loan applicants.

    Low-risk applicants: {low_risk:,}

    High-risk applicants: {high_risk:,}

    Exploratory analysis indicates that demographic,
    financial, and ownership-related features may
    contribute to loan risk classification and should
    be further evaluated during model training.
    """
)