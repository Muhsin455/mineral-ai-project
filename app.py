import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="MineralAI", page_icon="⛰️", layout="wide")

st.title("⛰️ MineralAI")
st.subheader("AI for Mineral Exploration Using Satellite & Seismic Data")
st.write(
    "A research prototype that trains a machine-learning model from a labelled "
    "dataset and displays its evaluation. Predictions are decision-support only, "
    "not proof that a mineral deposit exists."
)

with st.expander("Important: data and scientific limitations", expanded=True):
    st.warning(
        "Do not treat predictions as confirmed mineral discoveries. This app does not "
        "download or process raw satellite imagery automatically. Upload a real, "
        "documented dataset containing extracted features and verified labels. "
        "Never report demo data or untested values as real results."
    )

st.header("1. Upload a labelled dataset")
st.markdown(
    "Upload a CSV where each row represents a location/sample. Include numeric feature "
    "columns (for example, spectral indices, elevation, magnetic/geophysical features "
    "or valid seismic-survey attributes) and one target label column."
)

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    df = None
    st.info("Upload your real dataset to enable training. No fabricated data is used automatically.")

if df is not None:
    st.header("2. Configure the model")
    cols = list(df.columns)
    target = st.selectbox("Select the verified target/label column", cols)
    possible_features = [c for c in cols if c != target and pd.api.types.is_numeric_dtype(df[c])]
    features = st.multiselect(
        "Select numeric feature columns",
        possible_features,
        default=possible_features
    )
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    n_estimators = st.slider("Random Forest trees", 50, 500, 200, 50)

    if st.button("Train and evaluate model", type="primary"):
        if not features:
            st.error("Select at least one numeric feature column.")
            st.stop()

        work = df[features + [target]].copy()
        work = work.dropna(subset=[target])
        if work[target].nunique() < 2:
            st.error("The target column must contain at least two classes.")
            st.stop()
        if len(work) < 10:
            st.error("Please provide at least 10 labelled rows; more data is strongly recommended.")
            st.stop()

        X = work[features].replace([np.inf, -np.inf], np.nan)
        y_raw = work[target].astype(str)
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)

        counts = pd.Series(y).value_counts()
        stratify = y if counts.min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
        except ValueError as exc:
            st.error(f"Train/test split failed: {exc}")
            st.stop()

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators, random_state=42,
                class_weight="balanced"
            ))
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        average = "weighted"
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average=average, zero_division=0)
        rec = recall_score(y_test, pred, average=average, zero_division=0)
        f1 = f1_score(y_test, pred, average=average, zero_division=0)

        st.session_state["trained_model"] = model
        st.session_state["feature_cols"] = features
        st.session_state["target_col"] = target
        st.session_state["class_names"] = list(encoder.classes_)
        st.session_state["label_encoder"] = encoder

        st.header("3. Evaluation on held-out test data")
        st.caption(
            f"These metrics were calculated from this uploaded dataset's test split "
            f"({len(y_test)} rows). They are not general proof of field accuracy."
        )
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{acc:.3f}")
        b.metric("Precision (weighted)", f"{prec:.3f}")
        c.metric("Recall (weighted)", f"{rec:.3f}")
        d.metric("F1 score (weighted)", f"{f1:.3f}")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(encoder.classes_)))
        ax.set_yticks(range(len(encoder.classes_)))
        ax.set_xticklabels(encoder.classes_, rotation=45, ha="right")
        ax.set_yticklabels(encoder.classes_)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig)

        report = classification_report(
            y_test, pred, labels=np.arange(len(encoder.classes_)),
            target_names=[str(x) for x in encoder.classes_],
            zero_division=0, output_dict=True
        )
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        clf = model.named_steps["classifier"]
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": clf.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.subheader("Feature importance (model-specific)")
        st.dataframe(importance, use_container_width=True)
        st.bar_chart(importance.set_index("Feature"))

        results = pd.DataFrame({
            "actual_label": encoder.inverse_transform(y_test),
            "predicted_label": encoder.inverse_transform(pred)
        })
        st.download_button(
            "Download test predictions CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="mineralai_test_predictions.csv",
            mime="text/csv"
        )

st.header("4. Predict for a new sample")
if "trained_model" not in st.session_state:
    st.info("Train a model above before entering a new sample.")
else:
    model = st.session_state["trained_model"]
    features = st.session_state["feature_cols"]
    encoder = st.session_state["label_encoder"]
    with st.form("new_sample_form"):
        vals = {}
        cols = st.columns(min(3, len(features)))
        for i, feature in enumerate(features):
            with cols[i % len(cols)]:
                vals[feature] = st.number_input(
                    feature, value=0.0, format="%.6f", key=f"sample_{feature}"
                )
        submit = st.form_submit_button("Predict sample class")
    if submit:
        sample = pd.DataFrame([vals], columns=features)
        result = int(model.predict(sample)[0])
        label = encoder.inverse_transform([result])[0]
        st.success(f"Model output class: {label}")
        st.caption(
            "This is a model classification based on the uploaded training data, "
            "not a confirmed mineral deposit. Validate with geology experts and field data."
        )

st.header("5. Recommended workflow")
st.markdown(
    """
1. Obtain openly accessible satellite imagery and document the sensor, date, area and licence.
2. Obtain verified mineral occurrence labels from an official geological source.
3. Extract features for the same locations and spatial resolution.
4. Keep labels separate from predictors; remove duplicate/leaking columns.
5. Split data by geographic area where possible to reduce spatial leakage.
6. Train and evaluate using a held-out test set; document class imbalance.
7. Validate predictions independently before interpreting them geologically.
"""
)
st.caption("MineralAI educational prototype. Results depend entirely on the uploaded data and validation method.")
