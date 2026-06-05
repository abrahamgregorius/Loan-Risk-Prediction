import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("Loan Risk Prediction AI")
st.caption("Machine Learning Pipeline Project")

st.write(
		"""
		Welcome to Loan Risk Prediction AI. This application demonstrates a complete
		Machine Learning pipeline for predicting loan default risk based on
		demographic and financial information.
		"""
)

st.subheader("Application Features")

col1, col2, col3 = st.columns(3)

with col1:
		st.container(border=True)
		st.markdown(
				"""
				### Dataset Analysis

				- Dataset Overview
				- Feature Description
				- Target Analysis
				- Sample Records
				"""
		)

with col2:
		st.container(border=True)
		st.markdown(
				"""
				### Exploratory Data Analysis

				- Distribution Analysis
				- Class Balance
				- Feature Relationships
				- Correlation Analysis
				"""
		)

with col3:
		st.container(border=True)
		st.markdown(
				"""
				### Machine Learning

				- Data Preprocessing
				- Model Training
				- Model Comparison
				- Risk Prediction
				"""
		)

st.subheader("Machine Learning Workflow")

col1, col2, col3, col4 = st.columns(4)

with col1:
		st.info(
				"""
				Dataset & EDA

				Dataset Overview

				Feature Understanding

				Data Exploration
				"""
		)

with col2:
		st.info(
				"""
				Preprocessing

				Categorical Encoding

				Train-Test Split

				Feature Scaling
				"""
		)

with col3:
		st.info(
				"""
				Model Training

				Logistic Regression

				Random Forest

				Gradient Boosting
				"""
		)

with col4:
		st.info(
				"""
				Prediction

				Interactive Demo

				Risk Analysis

				Recommendation
				"""
		)

st.subheader("Project Objective")

st.write(
		"""
		The objective of this project is to build and evaluate machine learning
		classification models capable of predicting loan default risk.

		Multiple algorithms are trained and compared using standard evaluation
		metrics to identify the most effective model for risk prediction.
		"""
)

st.subheader("Modules")

modules = [
		"Dataset Overview",
		"Exploratory Data Analysis",
		"Data Preprocessing",
		"Model Training",
		"Model Evaluation",
		"Interactive Prediction"
]

for module in modules:
		st.markdown(f"- {module}")

st.subheader("Team Members")

col1, col2, col3 = st.columns(3)

with col1:
		st.container(border=True)
		st.markdown(
				"""
				#### Abraham Gregorius Anderson Thio
				"""
		)

with col2:
		st.container(border=True)
		st.markdown(
				"""
				#### Alwan Athallah Mumtaz
				"""
		)

with col3:
		st.container(border=True)
		st.markdown(
				"""
				#### Stanislaus Alva Jufinto
				"""
		)

st.divider()

st.caption("Loan Risk Prediction AI")