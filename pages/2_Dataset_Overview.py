import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

df = load_data()

st.title("Dataset Overview")

rows, cols = df.shape
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()
target_counts = df["default"].value_counts()

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

st.subheader("Target Distribution (Default)")

no_default = int(target_counts.get(0, 0))
default = int(target_counts.get(1, 0))

col1, col2 = st.columns(2)

col1.metric("No Default (0)", f"{no_default:,}")
col2.metric("Default (1)", f"{default:,}")

pie_df = pd.DataFrame({
    "Status": ["No Default", "Default"],
    "Count": [no_default, default]
})

fig = px.pie(
    pie_df,
    names="Status",
    values="Count",
    hole=0.4,
    title="Default Distribution"
)

st.plotly_chart(fig, use_container_width=True)

imbalance_ratio = default / max(no_default, 1)

st.info(f"Class Imbalance Ratio (1/0): {imbalance_ratio:.4f}")

st.subheader("Feature Overview")

feature_cols = [col for col in df.columns if col != "default"]

feature_summary = pd.DataFrame({
    "Feature": feature_cols,
    "Data Type": [df[col].dtype for col in feature_cols],
    "Unique Values": [df[col].nunique() for col in feature_cols],
    "Missing Values": [df[col].isnull().sum() for col in feature_cols]
})

st.dataframe(feature_summary, use_container_width=True)

st.subheader("Numeric Feature Statistics")

numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["default"], errors="ignore")

st.dataframe(numeric_df.describe().T, use_container_width=True)

st.subheader("📋 Detailed Column Descriptions")

# Create column descriptions
column_descriptions = {
    "age": {
        "description": "Age of the Customer",
        "type": "Numeric (Integer)",
        "range": f"{df['age'].min():.0f} - {df['age'].max():.0f} years",
        "mean": f"{df['age'].mean():.1f} years",
        "std": f"{df['age'].std():.1f} years",
        "insight": "Represents the age of the loan applicant. Younger applicants may have higher default risk due to less credit history and income stability."
    },
    "ed": {
        "description": "Education Level",
        "type": "Categorical (Numeric)",
        "range": f"{int(df['ed'].min()):.0f} - {int(df['ed'].max()):.0f}",
        "values": f"Levels: {sorted(df['ed'].unique())}",
        "insight": "Education level coded as 1-5 (typically 1=High School to 5=Postgraduate). Higher education is often associated with stable employment and lower default risk."
    },
    "employ": {
        "description": "Work Experience (Years of Employment)",
        "type": "Numeric (Integer)",
        "range": f"{df['employ'].min():.0f} - {df['employ'].max():.0f} years",
        "mean": f"{df['employ'].mean():.1f} years",
        "std": f"{df['employ'].std():.1f} years",
        "insight": "Total years of work experience. More experienced workers typically have more stable income and lower default probability."
    },
    "address": {
        "description": "Years at Current Address",
        "type": "Numeric (Integer)",
        "range": f"{df['address'].min():.0f} - {df['address'].max():.0f} years",
        "mean": f"{df['address'].mean():.1f} years",
        "std": f"{df['address'].std():.1f} years",
        "insight": "Duration at current residence. Longer residence indicates residential stability, which correlates with better credit behavior."
    },
    "income": {
        "description": "Yearly Income of the Customer",
        "type": "Numeric (Continuous)",
        "range": f"{df['income'].min():.0f} - {df['income'].max():.0f} units",
        "mean": f"{df['income'].mean():.1f} units",
        "std": f"{df['income'].std():.1f} units",
        "insight": "Annual income in arbitrary units. Higher income generally indicates better repayment capacity and lower default risk."
    },
    "debtinc": {
        "description": "Debt-to-Income Ratio",
        "type": "Numeric (Continuous)",
        "range": f"{df['debtinc'].min():.2f} - {df['debtinc'].max():.2f}",
        "mean": f"{df['debtinc'].mean():.2f}",
        "std": f"{df['debtinc'].std():.2f}",
        "insight": "Ratio of total debt payments to income. Higher ratios indicate less disposable income for loan repayment, increasing default risk."
    },
    "creddebt": {
        "description": "Credit-to-Debt Ratio",
        "type": "Numeric (Continuous)",
        "range": f"{df['creddebt'].min():.2f} - {df['creddebt'].max():.2f}",
        "mean": f"{df['creddebt'].mean():.2f}",
        "std": f"{df['creddebt'].std():.2f}",
        "insight": "Ratio of available credit to existing debt. Higher ratios indicate more available credit cushion and potentially lower default risk."
    },
    "othdebt": {
        "description": "Other Debts",
        "type": "Numeric (Continuous)",
        "range": f"{df['othdebt'].min():.2f} - {df['othdebt'].max():.2f} units",
        "mean": f"{df['othdebt'].mean():.2f} units",
        "std": f"{df['othdebt'].std():.2f} units",
        "insight": "Amount of miscellaneous or other debts. Higher other debts reduce overall financial capacity and may increase default risk."
    },
    "default": {
        "description": "Customer Default Status (Target Variable)",
        "type": "Binary (0 or 1)",
        "values": "0 = No Default, 1 = Defaulted",
        "distribution": f"0 (No Default): {no_default:,} ({no_default/len(df)*100:.1f}%) | 1 (Default): {default:,} ({default/len(df)*100:.1f}%)",
        "insight": "Whether the customer defaulted on the loan. This is the target variable we are trying to predict."
    }
}

# Display each column description
for col in df.columns:
    if col in column_descriptions:
        info = column_descriptions[col]
        
        with st.expander(f"ℹ️ **{info['description']}** ({col})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Data Type:** {info['type']}")
                if "range" in info:
                    st.write(f"**Range:** {info['range']}")
                elif "values" in info:
                    st.write(f"**{info['values']}")
                elif "distribution" in info:
                    st.write(f"**Distribution:** {info['distribution']}")
            
            with col2:
                if "mean" in info:
                    st.write(f"**Mean:** {info['mean']}")
                if "std" in info:
                    st.write(f"**Std Dev:** {info['std']}")
            
            st.write(f"\n**Insight:** {info['insight']}")

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
no_default_pct = (no_default / total) * 100
default_pct = (default / total) * 100

st.write(f"""
- Dataset size: **{total:,} rows**
- Features: **{cols} columns**
- No Default: **{no_default_pct:.2f}%** ({no_default:,} customers)
- Default: **{default_pct:.2f}%** ({default:,} customers)
- Missing values: **{missing_values}**
- Duplicate rows: **{duplicate_rows}**
""")