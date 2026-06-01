import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

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
		df = pd.read_parquet("dataset/processed.parquet")
except Exception:
		st.warning("Please run preprocessing first.")
		st.stop()

try:
		with open("dataset/preprocessing_config.json") as file:
				config = json.load(file)
except Exception:
		st.warning("Preprocessing configuration not found.")
		st.stop()

st.subheader("Pipeline Configuration")

st.dataframe(
		pd.DataFrame({
				"Setting": config.keys(),
				"Value": config.values()
		}),
		use_container_width=True,
		hide_index=True
)

threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.3)

if st.button("Train Models", use_container_width=True):

		X = df.drop(columns=["Risk_Flag"])
		y = df["Risk_Flag"]

		st.subheader("Class Distribution")
		st.write(y.value_counts())
		st.write(y.value_counts(normalize=True))

		X_train, X_test, y_train, y_test = train_test_split(
				X,
				y,
				test_size=config["test_size"],
				random_state=42,
				stratify=y
		)

		scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

		models = {
				"Logistic Regression (Balanced)": LogisticRegression(
						class_weight="balanced",
						max_iter=2000
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
						random_state=42
				)
		}

		results = []
		trained_models = {}

		for name, model in models.items():

				model.fit(X_train, y_train)

				y_prob = model.predict_proba(X_test)[:, 1]
				y_pred = (y_prob >= threshold).astype(int)

				metrics = {
						"Model": name,
						"Accuracy": accuracy_score(y_test, y_pred),
						"Precision": precision_score(y_test, y_pred, zero_division=0),
						"Recall": recall_score(y_test, y_pred, zero_division=0),
						"F1 Score": f1_score(y_test, y_pred, zero_division=0),
						"ROC-AUC": roc_auc_score(y_test, y_prob),
						"PR-AUC": average_precision_score(y_test, y_prob)
				}

				results.append(metrics)
				trained_models[name] = (model, y_prob)

		results_df = pd.DataFrame(results)

		st.subheader("Model Comparison")
		st.dataframe(results_df, use_container_width=True)

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
		
		st.subheader("How to Read Evaluation Metrics")

		st.markdown("""
			### Accuracy
			Persentase prediksi yang benar dari seluruh data.

			**Formula:**
			(TP + TN) / Total Data

			**Interpretation:**
			- Cocok jika dataset seimbang.
			- Bisa menyesatkan pada dataset imbalanced.

			Contoh:
			Jika 95% data adalah non-risk dan model selalu menebak non-risk,
			accuracy bisa tetap tinggi meskipun gagal menemukan kasus risk.

			---

			### Precision
			Dari semua yang diprediksi **Risk**, berapa yang benar-benar Risk.

			**Formula:**
			TP / (TP + FP)

			**Interpretation:**
			- Precision tinggi → sedikit False Positive.
			- Penting jika salah menuduh seseorang sebagai Risk itu mahal.

			Contoh:
			Precision = 0.80

			Artinya:
			Dari 100 orang yang diprediksi Risk,
			80 benar-benar Risk.

			---

			### Recall
			Dari semua Risk yang sebenarnya ada,
			berapa yang berhasil ditemukan model.

			**Formula:**
			TP / (TP + FN)

			**Interpretation:**
			- Recall tinggi → sedikit kasus Risk yang terlewat.
			- Biasanya metrik paling penting untuk fraud, disease, default prediction.

			Contoh:
			Recall = 0.70

			Artinya:
			Dari 100 kasus Risk,
			70 berhasil ditemukan model.

			---

			### F1 Score
			Keseimbangan antara Precision dan Recall.

			**Formula:**
			2 × (Precision × Recall) / (Precision + Recall)

			**Interpretation:**
			- Cocok untuk dataset imbalanced.
			- Semakin tinggi semakin baik.

			---

			### ROC-AUC
			Kemampuan model membedakan kelas Risk dan Non-Risk.

			Interpretasi umum:

			- 0.50 → Tebak acak
			- 0.60–0.70 → Cukup
			- 0.70–0.80 → Baik
			- 0.80–0.90 → Sangat Baik
			- >0.90 → Excellent

			---

			### PR-AUC (Precision-Recall AUC)
			Area di bawah kurva Precision-Recall.

			**Interpretation:**
			- Sangat penting untuk dataset imbalanced.
			- Lebih representatif dibanding Accuracy.
			- Semakin tinggi semakin baik.

			Karena kasus Risk biasanya jauh lebih sedikit daripada Non-Risk,
			PR-AUC sering dijadikan metrik utama pemilihan model.
		""")

		
		st.plotly_chart(fig, use_container_width=True)

		best_row = results_df.loc[
				results_df["PR-AUC"].idxmax()
		]

		best_name = best_row["Model"]
		best_model, best_prob = trained_models[best_name]

		st.success(
				f"Best Model: {best_name} | PR-AUC: {best_row['PR-AUC']:.4f}"
		)
  
		st.subheader("Model Interpretation")

		st.info(f"""
			Model terbaik dipilih berdasarkan **PR-AUC = {best_row['PR-AUC']:.4f}**.

			Interpretasi:

			- Accuracy : {best_row['Accuracy']:.2%}
			- Precision : {best_row['Precision']:.2%}
			- Recall : {best_row['Recall']:.2%}
			- F1 Score : {best_row['F1 Score']:.2%}

			Artinya:

			- Dari seluruh prediksi, model benar sekitar {best_row['Accuracy']:.2%}.
			- Dari semua yang diprediksi Risk, sekitar {best_row['Precision']:.2%} benar-benar Risk.
			- Model berhasil menemukan sekitar {best_row['Recall']:.2%} dari seluruh kasus Risk yang ada.
			- Keseimbangan Precision dan Recall sebesar {best_row['F1 Score']:.2%}.
		""")


		joblib.dump(best_model, "models/best_model.pkl")
		joblib.dump(X.columns.tolist(), "models/features.pkl")

		y_pred_best = (best_prob >= threshold).astype(int)

		cm = confusion_matrix(y_test, y_pred_best)

		cm_df = pd.DataFrame(
				cm,
				index=["Actual 0", "Actual 1"],
				columns=["Predicted 0", "Predicted 1"]
		)

		st.subheader("Confusion Matrix")

		fig = px.imshow(cm_df, text_auto=True, aspect="auto")
		st.plotly_chart(fig, use_container_width=True)

		if hasattr(best_model, "feature_importances_"):

				importance_df = pd.DataFrame({
						"Feature": X.columns,
						"Importance": best_model.feature_importances_
				}).sort_values("Importance", ascending=False).head(15)

				st.subheader("Feature Importance")

				fig = px.bar(
						importance_df,
						x="Importance",
						y="Feature",
						orientation="h"
				)

				st.plotly_chart(fig, use_container_width=True)