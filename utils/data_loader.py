import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv(
        "dataset/loans.csv"
    )
    
    # Remove rows with NaN in target variable
    df = df.dropna(subset=['default'])
    
    # Reset index
    df = df.reset_index(drop=True)

    return df