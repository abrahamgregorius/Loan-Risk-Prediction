import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_data

df = load_data()

st.title("Exploratory Data Analysis")

st.write("Dataset Source: dataset/loans.csv")

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
        "Default Distribution",
        "Income Distribution",
        "Age Distribution",
        "Employment Experience Distribution",
        "Debt to Income Ratio Distribution",
        "Credit to Debt Ratio Distribution",
        "Other Debts Distribution",
        "Correlation Heatmap",
        "Income vs Default",
        "Age vs Default",
        "Employment Experience vs Default"
    ]
)

insight = ""

if visualization == "Default Distribution":

    counts = df["default"].value_counts().sort_index()

    chart_df = pd.DataFrame(
        {
            "Status": ["No Default", "Default"],
            "Count": [
                counts.get(0, 0),
                counts.get(1, 0)
            ]
        }
    )

    fig = px.bar(
        chart_df,
        x="Status",
        y="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    no_default_pct = (
        counts.get(0, 0) / len(df)
    ) * 100

    insight = (
        f"The dataset is imbalanced. "
        f"Approximately {no_default_pct:.2f}% of applicants "
        f"have not defaulted."
    )

elif visualization == "Income Distribution":

    fig = px.histogram(
        df,
        x="income",
        nbins=50
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"Average income is "
        f"{df['income'].mean():,.0f}."
    )

elif visualization == "Age Distribution":

    fig = px.histogram(
        df,
        x="age",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average applicant age is "
        f"{df['age'].mean():.1f} years."
    )

elif visualization == "Employment Experience Distribution":

    fig = px.histogram(
        df,
        x="employ",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average work experience is "
        f"{df['employ'].mean():.1f} years."
    )

elif visualization == "Debt to Income Ratio Distribution":

    fig = px.histogram(
        df,
        x="debtinc",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average debt-to-income ratio is "
        f"{df['debtinc'].mean():.2f}."
    )

elif visualization == "Credit to Debt Ratio Distribution":

    fig = px.histogram(
        df,
        x="creddebt",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average credit-to-debt ratio is "
        f"{df['creddebt'].mean():.2f}."
    )

elif visualization == "Other Debts Distribution":

    fig = px.histogram(
        df,
        x="othdebt",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        f"The average other debts is "
        f"{df['othdebt'].mean():.2f}."
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

    default_corr = (
        corr["default"]
        .drop("default")
        .abs()
        .sort_values(ascending=False)
    )

    insight = (
        f"The strongest numerical relationship "
        f"with default is "
        f"'{default_corr.index[0]}'."
    )

elif visualization == "Income vs Default":

    fig = px.box(
        df,
        x="default",
        y="income"
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        "Income distribution differs between "
        "default groups and may contribute to "
        "loan default prediction."
    )

elif visualization == "Age vs Default":

    fig = px.box(
        df,
        x="default",
        y="age"
    )

    st.plotly_chart(fig, use_container_width=True)

    insight = (
        "Age patterns vary across default groups "
        "and may provide predictive value."
    )

elif visualization == "Employment Experience vs Default":

    fig = px.box(
        df,
        x="default",
        y="employ"
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
defaults = int(df["default"].sum())
no_defaults = total_records - defaults

st.write(
    f"""
    The dataset contains {total_records:,} loan applicants.

    No defaults: {no_defaults:,}

    Defaults: {defaults:,}

    Exploratory analysis indicates that demographic,
    financial, and debt-related features may
    contribute to loan default classification and should
    be further evaluated during model training.
    """
)