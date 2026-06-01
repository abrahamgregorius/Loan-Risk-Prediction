import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv(
        "dataset/loan.csv"
    )

    df = df.drop(
        columns=[
            "CITY",
            "STATE"
        ]
    )

    return df