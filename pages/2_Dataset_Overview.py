import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

df = load_data()

st.title("Dataset Overview")

rows, cols = df.shape
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()
target_counts = df["Risk_Flag"].value_counts()

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", f"{rows:,}")
col2.metric("Total Columns", cols)
col3.metric("Missing Values", missing_values)
col4.metric("Duplicate Rows", duplicate_rows)

st.subheader("Data Types")

dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index()
dtype_df.columns = ["Feature", "Data Type"]

st.dataframe(dtype_df, use_container_width=True, hide_index=True)

st.subheader("Target Distribution (Risk_Flag)")

low_risk = int(target_counts.get(0, 0))
high_risk = int(target_counts.get(1, 0))

col1, col2 = st.columns(2)

col1.metric("Low Risk (0)", f"{low_risk:,}")
col1.metric("High Risk (1)", f"{high_risk:,}")

pie_df = pd.DataFrame({
    "Risk": ["Low Risk", "High Risk"],
    "Count": [low_risk, high_risk]
})

fig = px.pie(
    pie_df,
    names="Risk",
    values="Count",
    hole=0.4,
    title="Risk Distribution"
)

st.plotly_chart(fig, use_container_width=True)

imbalance_ratio = high_risk / max(low_risk, 1)

st.info(f"Class Imbalance Ratio (1/0): {imbalance_ratio:.4f}")

st.subheader("Feature Overview")

feature_cols = [col for col in df.columns if col != "Risk_Flag"]

feature_summary = pd.DataFrame({
    "Feature": feature_cols,
    "Data Type": [df[col].dtype for col in feature_cols],
    "Unique Values": [df[col].nunique() for col in feature_cols],
    "Missing Values": [df[col].isnull().sum() for col in feature_cols]
})

st.dataframe(feature_summary, use_container_width=True)

st.subheader("Numeric Feature Statistics")

numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Risk_Flag"])

st.dataframe(numeric_df.describe().T, use_container_width=True)
st.subheader("Top Categorical Distributions")

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols[:4]:  # limit biar UI tidak penuh
    top_vals = df[col].value_counts().head(10)

    fig = px.bar(
        x=top_vals.index,
        y=top_vals.values,
        title=f"Top values: {col}"
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Data Quality Insights")

st.write("**Missing Data Percentage per Feature**")

missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = missing_pct.reset_index()
missing_df.columns = ["Feature", "Missing %"]

fig = px.bar(
    missing_df.sort_values("Missing %", ascending=False),
    x="Feature",
    y="Missing %",
    title="Missing Data Distribution"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset Interpretation")

st.warning("""
This dataset is synthetic and designed for machine learning practice.

Important implications:
- Income is NOT tied to real-world currency
- Feature relationships may be artificially generated
- High model performance does not guarantee real-world accuracy
""")

st.info("""
Target Variable:
- 0 = Low Risk (No Default)
- 1 = High Risk (Default Risk)
""")

st.subheader("Quick Insights")

total = len(df)
low_pct = (low_risk / total) * 100
high_pct = (high_risk / total) * 100

st.write(f"""
- Dataset size: **{total:,} rows**
- Features: **{cols} columns**
- Low risk: **{low_pct:.2f}%**
- High risk: **{high_pct:.2f}%**
- Missing values: **{missing_values}**
- Duplicate rows: **{duplicate_rows}**
""")