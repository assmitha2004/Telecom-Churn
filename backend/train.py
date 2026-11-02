"""
Offline training script for the Telecom Churn project.

Run this ONCE on your own computer (it needs the 79 MB dataset and the heavy
libraries: pandas, scikit-learn, imbalanced-learn). It writes one small file:

    backend/artifacts/model.json

The web API only reads that JSON file. So the API does not need pandas,
scikit-learn or the dataset, and it starts in milliseconds. This is what makes
it possible to deploy the API on Vercel.

Usage (from the project root):

    pip install -r backend/requirements-train.txt
    python backend/train.py

Pipeline (same idea as the original notebook):
    1. Drop columns with more than 30% missing values, date columns, circle_id
    2. Keep only HIGH-VALUE customers (average recharge of June+July is in the top 30%)
    3. Tag churners: no calls and no mobile data in month 9 (September)
    4. Remove every month-9 column (and sep_vbc_3g)  -> no data leakage
    5. Add "usage decreased in August" trend features
    6. Split 70/30, SMOTE on the TRAIN part only, StandardScaler, PCA(60), Logistic Regression
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "telecom_churn_data.csv")
OUT_PATH = os.path.join(HERE, "artifacts", "model.json")

RANDOM_STATE = 42
N_KEY_FEATURES = 11  # number of user-facing inputs shown on the Predict page

# ----------------------------------------------------------------------------
# Human readable names for the columns
# ----------------------------------------------------------------------------
MONTHS = {"6": "Jun", "7": "Jul", "8": "Aug"}

BASE_LABELS = {
    "arpu": "Revenue per user",
    "onnet_mou": "On-net call minutes",
    "offnet_mou": "Off-net call minutes",
    "roam_ic_mou": "Roaming incoming minutes",
    "roam_og_mou": "Roaming outgoing minutes",
    "loc_og_t2t_mou": "Local outgoing minutes, same operator",
    "loc_og_t2m_mou": "Local outgoing minutes, other operator",
    "loc_og_t2f_mou": "Local outgoing minutes, fixed line",
    "loc_og_t2c_mou": "Local outgoing minutes, call centre",
    "loc_og_mou": "Local outgoing minutes",
    "std_og_t2t_mou": "STD outgoing minutes, same operator",
    "std_og_t2m_mou": "STD outgoing minutes, other operator",
    "std_og_t2f_mou": "STD outgoing minutes, fixed line",
    "std_og_t2c_mou": "STD outgoing minutes, call centre",
    "std_og_mou": "STD outgoing minutes",
    "isd_og_mou": "ISD outgoing minutes",
    "spl_og_mou": "Special outgoing minutes",
    "og_others": "Other outgoing minutes",
    "total_og_mou": "Total outgoing minutes",
    "loc_ic_t2t_mou": "Local incoming minutes, same operator",
    "loc_ic_t2m_mou": "Local incoming minutes, other operator",
    "loc_ic_t2f_mou": "Local incoming minutes, fixed line",
    "loc_ic_mou": "Local incoming minutes",
    "std_ic_t2t_mou": "STD incoming minutes, same operator",
    "std_ic_t2m_mou": "STD incoming minutes, other operator",
    "std_ic_t2f_mou": "STD incoming minutes, fixed line",
    "std_ic_t2o_mou": "STD incoming minutes, other",
    "std_ic_mou": "STD incoming minutes",
    "total_ic_mou": "Total incoming minutes",
    "spl_ic_mou": "Special incoming minutes",
    "isd_ic_mou": "ISD incoming minutes",
    "ic_others": "Other incoming minutes",
    "total_rech_num": "Number of recharges",
    "total_rech_amt": "Total recharge amount",
    "max_rech_amt": "Maximum recharge amount",
    "last_day_rch_amt": "Last day recharge amount",
    "total_rech_data": "Number of data recharges",
    "max_rech_data": "Maximum data recharge",
    "count_rech_2g": "Number of 2G recharges",
    "count_rech_3g": "Number of 3G recharges",
    "av_rech_amt_data": "Average data recharge amount",
    "vol_2g_mb": "2G data used (MB)",
    "vol_3g_mb": "3G data used (MB)",
    "arpu_3g": "3G revenue per user",
    "arpu_2g": "2G revenue per user",
    "night_pck_user": "Night pack user (1 = yes)",
    "monthly_2g": "Monthly 2G packs",
    "sachet_2g": "Sachet 2G packs",
    "monthly_3g": "Monthly 3G packs",
    "sachet_3g": "Sachet 3G packs",
    "fb_user": "Facebook pack user (1 = yes)",
    "aon": "Age on network (days)",
    "loc_og_t2o_mou": "Local outgoing minutes, other",
    "std_og_t2o_mou": "STD outgoing minutes, other",
    "loc_ic_t2o_mou": "Local incoming minutes, other",
}

FULL_LABELS = {
    "jun_vbc_3g": "3G volume-based cost (Jun)",
    "jul_vbc_3g": "3G volume-based cost (Jul)",
    "aug_vbc_3g": "3G volume-based cost (Aug)",
    "avg_rech_amt_6_7": "Average recharge amount (Jun-Jul)",
    "decrease_mou_action": "Outgoing minutes fell in Aug (1 = yes)",
    "decrease_rech_num_action": "Number of recharges fell in Aug (1 = yes)",
    "decrease_rech_amt_action": "Recharge amount fell in Aug (1 = yes)",
    "decrease_arpu_action": "Revenue per user fell in Aug (1 = yes)",
    "decrease_vbc_action": "3G volume-based cost fell in Aug (1 = yes)",
}


def make_label(col: str) -> str:
    """Turn a column name such as total_ic_mou_8 into 'Total incoming minutes (Aug)'."""
    if col in FULL_LABELS:
        return FULL_LABELS[col]
    base, _, tail = col.rpartition("_")
    if tail in MONTHS and base in BASE_LABELS:
        return f"{BASE_LABELS[base]} ({MONTHS[tail]})"
    if col in BASE_LABELS:
        return BASE_LABELS[col]
    return col.replace("_", " ").capitalize()


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """Clean the data and build features. Returns X (DataFrame), y (Series)."""
    # 1. Drop columns with more than 30% missing values
    missing_ratio = df.isnull().mean()
    df = df.drop(columns=missing_ratio[missing_ratio > 0.3].index.tolist())

    # 2. Drop date columns and circle_id (no predictive value)
    date_cols = [c for c in df.columns if "date" in c]
    df = df.drop(columns=date_cols + ["circle_id"], errors="ignore")

    # 3. Keep only high-value customers
    df["avg_rech_amt_6_7"] = (df["total_rech_amt_6"] + df["total_rech_amt_7"]) / 2
    cutoff = df["avg_rech_amt_6_7"].quantile(0.7)
    df = df[df["avg_rech_amt_6_7"] >= cutoff].copy()

    # 4. Tag churners: no calls and no mobile data in month 9
    tag_cols = [c for c in ["total_ic_mou_9", "total_og_mou_9", "vol_2g_mb_9", "vol_3g_mb_9"] if c in df.columns]
    df["churn"] = (df[tag_cols].fillna(0).sum(axis=1) == 0).astype(int)

    # 5. Remove everything from the churn phase (month 9).
    #    IMPORTANT: sep_vbc_3g is also September data but its name does not end with _9.
    #    The original project kept it. That is data leakage, so we remove it here.
    leak_cols = [c for c in df.columns if c.endswith("_9") or c.startswith("sep_")]
    df = df.drop(columns=leak_cols)

    # 6. Trend features: did usage in August fall below the June-July average?
    for prefix, label in [
        ("total_og_mou", "mou"),
        ("total_rech_num", "rech_num"),
        ("total_rech_amt", "rech_amt"),
        ("arpu", "arpu"),
    ]:
        c6, c7, c8 = f"{prefix}_6", f"{prefix}_7", f"{prefix}_8"
        if all(c in df.columns for c in (c6, c7, c8)):
            good_phase = (df[c6].fillna(0) + df[c7].fillna(0)) / 2
            df[f"decrease_{label}_action"] = (df[c8].fillna(0) < good_phase).astype(int)
    if all(c in df.columns for c in ("jun_vbc_3g", "jul_vbc_3g", "aug_vbc_3g")):
        good_phase = (df["jun_vbc_3g"].fillna(0) + df["jul_vbc_3g"].fillna(0)) / 2
        df["decrease_vbc_action"] = (df["aug_vbc_3g"].fillna(0) < good_phase).astype(int)

    df = df.drop(columns=["mobile_number"], errors="ignore").fillna(0)

    y = df["churn"]
    X = df.drop(columns=["churn"]).select_dtypes(include=[np.number])
    return X, y


def metrics_dict(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "sensitivity": round(float(recall_score(y_true, y_pred)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    started = time.time()
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found: {DATA_PATH}\nPut telecom_churn_data.csv in backend/data/ and run again.")

    print("1/6 Reading dataset ...")
    raw = pd.read_csv(DATA_PATH)
    total_records, total_columns = len(raw), raw.shape[1]

    print("2/6 Cleaning data and building features ...")
    X, y = preprocess(raw)
    del raw
    feature_cols = X.columns.tolist()
    print(f"    high-value customers: {len(X):,} | churners: {int(y.sum()):,} ({y.mean() * 100:.2f}%) | features: {len(feature_cols)}")

    print("3/6 Splitting and balancing (SMOTE on train only) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    print("4/6 Scaling, PCA and Logistic Regression ...")
    scaler = StandardScaler().fit(X_train_res)
    pca = PCA(n_components=60, random_state=RANDOM_STATE).fit(scaler.transform(X_train_res))
    Xtr = pca.transform(scaler.transform(X_train_res))
    Xte = pca.transform(scaler.transform(X_test))
    clf = LogisticRegression(C=1, max_iter=1000, random_state=RANDOM_STATE).fit(Xtr, y_train_res)

    proba_test = clf.predict_proba(Xte)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    pred_train = clf.predict(Xtr)

    metrics = {
        "train": metrics_dict(y_train_res, pred_train),
        "test": metrics_dict(y_test, pred_test),
        "auc": round(float(roc_auc_score(y_test, proba_test)), 4),
        "average_precision": round(float(average_precision_score(y_test, proba_test)), 4),
        "n_components": int(pca.n_components_),
        "variance_explained": round(float(pca.explained_variance_ratio_.sum()) * 100, 2),
    }
    print(f"    test accuracy {metrics['test']['accuracy']:.4f} | sensitivity {metrics['test']['sensitivity']:.4f} | AUC {metrics['auc']:.4f}")

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    step = max(1, len(fpr) // 120)
    roc = [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in zip(fpr[::step], tpr[::step])]
    roc.append({"fpr": 1.0, "tpr": 1.0})

    print("5/6 Collapsing the pipeline into one weight per feature ...")
    # scaler -> PCA -> logistic regression is a chain of linear steps, so it is exactly:
    #   logit = intercept + sum_j weight_j * ((x_j - mean_j) / scale_j - pca_mean_j)
    weights = clf.coef_[0] @ pca.components_
    intercept = float(clf.intercept_[0])
    mean, scale, pca_mean = scaler.mean_, scaler.scale_, pca.mean_

    z_test = (X_test.to_numpy() - mean) / scale
    logit_test = intercept + (z_test - pca_mean) @ weights
    proba_check = 1.0 / (1.0 + np.exp(-logit_test))
    max_diff = float(np.max(np.abs(proba_check - proba_test)))
    print(f"    max difference vs scikit-learn predict_proba: {max_diff:.2e}")
    assert max_diff < 1e-9, "Collapsed model does not match scikit-learn model!"

    # Feature importance = size of the weight (features are standardised, so weights are comparable)
    abs_w = np.abs(weights)
    share = abs_w / abs_w.sum() * 100
    order = np.argsort(abs_w)[::-1]
    importances = [
        {
            "feature": feature_cols[i],
            "label": make_label(feature_cols[i]),
            "importance": round(float(share[i]), 3),
            "influence": "raises risk" if weights[i] > 0 else "lowers risk",
        }
        for i in order[:15]
    ]

    print("6/6 Building Predict page schema and examples ...")
    defaults = X.median()

    # Choose the inputs for the Predict page: the strongest features, but skip a feature
    # when it is almost the same as one already chosen (correlation above 0.8).
    # Example: "total incoming minutes" and "local incoming minutes" move together,
    # so a user could type values that cannot exist in real life.
    top_candidates = [feature_cols[i] for i in order[:60]]
    corr = X[top_candidates].corr().abs()
    chosen = []
    for col in top_candidates:
        # Keep the form focused on fields a telecom operator can reasonably ask a user to provide.
        # Special incoming minutes is a tiny, niche field and is left at its dataset median.
        if col == "spl_ic_mou_8":
            continue
        if all(corr.loc[col, other] < 0.8 for other in chosen):
            chosen.append(col)
        if len(chosen) == N_KEY_FEATURES:
            break

    key_features = []
    for col in chosen:
        i = feature_cols.index(col)
        values = X[col]
        is_binary = set(np.unique(values)).issubset({0, 1})
        key_features.append(
            {
                "name": col,
                "label": make_label(col),
                "default": round(float(defaults[col]), 4),
                "min": 0 if is_binary else round(float(values.quantile(0.01)), 4),
                "max": 1 if is_binary else round(float(values.quantile(0.99)), 4),
                "binary": bool(is_binary),
                "weight": round(float(weights[i]), 4),
            }
        )

    key_names = [k["name"] for k in key_features]
    presets = [
        {
            "id": "typical",
            "label": "Typical customer",
            "description": "Median value of every feature.",
            "values": {c: round(float(defaults[c]), 4) for c in key_names},
        },
        {
            "id": "loyal",
            "label": "Typical loyal customer",
            "description": "Median values of customers who did NOT churn.",
            "values": {c: round(float(X.loc[y == 0, c].median()), 4) for c in key_names},
        },
        {
            "id": "churner",
            "label": "Typical churner",
            "description": "Median values of customers who churned.",
            "values": {c: round(float(X.loc[y == 1, c].median()), 4) for c in key_names},
        },
    ]

    artifact = {
        "meta": {
            "model": "StandardScaler -> PCA(60) -> LogisticRegression, collapsed to linear weights",
            "random_state": RANDOM_STATE,
            "threshold": 0.5,
            # The model is trained on SMOTE-balanced data, so its score is higher than the real
            # churn rate (8.6%). We call it a "risk score", not a real-world probability.
            "risk_bands": {"low_below": 0.40, "high_from": 0.60},
        },
        "feature_cols": feature_cols,
        "labels": {c: make_label(c) for c in feature_cols},
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "pca_mean": pca_mean.tolist(),
        "weights": weights.tolist(),
        "intercept": intercept,
        "defaults": {c: float(defaults[c]) for c in feature_cols},
        "key_features": key_features,
        "presets": presets,
        "metrics": metrics,
        "roc": roc,
        "importances": importances,
        "stats": {
            "total_records": int(total_records),
            "total_columns": int(total_columns),
            "high_value_customers": int(len(X)),
            "churn_count": int(y.sum()),
            "churn_rate": round(float(y.mean()) * 100, 2),
            "feature_count": len(feature_cols),
            "train_size": int(len(X_train)),
            "train_size_after_smote": int(len(X_train_res)),
            "test_size": int(len(X_test)),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f)
    print(f"Saved {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1024:.0f} KB) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
