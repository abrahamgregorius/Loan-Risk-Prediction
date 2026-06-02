import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

st.title("Model Training & Evaluation")

try:
    df = pd.read_parquet(
        "dataset/processed.parquet"
    )

except Exception:

    st.warning(
        "Please run preprocessing first."
    )

    st.stop()

try:

    with open(
        "dataset/preprocessing_config.json"
    ) as file:

        config = json.load(file)

except Exception:

    st.warning(
        "Preprocessing configuration not found."
    )

    st.stop()

st.subheader(
    "Pipeline Configuration"
)

st.dataframe(
    pd.DataFrame({
        "Setting": config.keys(),
        "Value": config.values()
    }),
    use_container_width=True,
    hide_index=True
)

threshold = st.slider(
    "Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05
)

if st.button(
    "Train Models",
    use_container_width=True
):

    X = df.drop(
        columns=["default"]
    )

    y = df["default"]

    st.subheader(
        "Class Distribution"
    )

    st.write(
        y.value_counts()
    )

    st.write(
        y.value_counts(
            normalize=True
        )
    )

    positive_rate = y.mean()

    st.info(
        f"Positive Class Rate: {positive_rate:.2%}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=42,
        stratify=y
    )

    scale_pos_weight = (
        y_train.value_counts()[0]
        /
        y_train.value_counts()[1]
    )

    models = {

        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=42
        ),

        "Balanced Random Forest": BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        ),

        "XGBoost (Best for Imbalance)": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(
                y_test,
                y_pred
            ),
            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_test,
                y_prob
            ),
            "PR-AUC": average_precision_score(
                y_test,
                y_prob
            )
        }

        results.append(
            metrics
        )

        trained_models[name] = {
            "model": model,
            "prob": y_prob
        }

    results_df = pd.DataFrame(
        results
    )

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    metric_df = results_df.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        metric_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    best_row = results_df.loc[
        results_df["PR-AUC"].idxmax()
    ]

    best_name = best_row["Model"]

    best_model = trained_models[
        best_name
    ]["model"]

    best_prob = trained_models[
        best_name
    ]["prob"]

    st.success(
        f"Best Model: {best_name} | PR-AUC: {best_row['PR-AUC']:.4f}"
    )

    st.subheader(
        "Model Interpretation"
    )

    st.info(f"""
Model terbaik dipilih berdasarkan PR-AUC.

Accuracy : {best_row['Accuracy']:.2%}
Precision : {best_row['Precision']:.2%}
Recall : {best_row['Recall']:.2%}
F1 Score : {best_row['F1 Score']:.2%}
ROC-AUC : {best_row['ROC-AUC']:.4f}
PR-AUC : {best_row['PR-AUC']:.4f}
""")

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    joblib.dump(
        threshold,
        "models/threshold.pkl"
    )

    joblib.dump(
        {
            "model_name": best_name,
            "threshold": float(
                threshold
            ),
            "roc_auc": float(
                best_row["ROC-AUC"]
            ),
            "pr_auc": float(
                best_row["PR-AUC"]
            )
        },
        "models/model_metadata.pkl"
    )

    y_pred_best = (
        best_prob >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred_best
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual 0",
            "Actual 1"
        ],
        columns=[
            "Predicted 0",
            "Predicted 1"
        ]
    )

    st.subheader(
        "Confusion Matrix"
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test,
            best_prob
        )
    )

    pr_df = pd.DataFrame({
        "Recall": recall_curve,
        "Precision": precision_curve
    })

    st.subheader(
        "Precision Recall Curve"
    )

    fig = px.line(
        pr_df,
        x="Recall",
        y="Precision",
        title=(
            f"PR Curve (AUC = {best_row['PR-AUC']:.4f})"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    importance_df = None

    if hasattr(
        best_model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance":
                best_model.feature_importances_
        })

    elif hasattr(
        best_model,
        "coef_"
    ):

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance":
                np.abs(
                    best_model.coef_[0]
                )
        })

    if importance_df is not None:

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        st.subheader(
            "Top 15 Most Important Features"
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )