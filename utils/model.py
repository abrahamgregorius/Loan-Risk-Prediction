from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def split_data(X, y, test_size):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


def apply_smote(X_train, y_train):
    smote = SMOTE(
        random_state=42,
        k_neighbors=5
    )

    return smote.fit_resample(X_train, y_train)


def get_models(lr_c, rf_estimators, rf_depth, gb_estimators, gb_learning_rate):

    return {
        "Logistic Regression": LogisticRegression(
            C=lr_c,
            max_iter=2000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_depth,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb_estimators,
            learning_rate=gb_learning_rate,
            random_state=42
        )
    }


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }