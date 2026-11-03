import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Term Deposit Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.title("Bank Term Deposit Subscription Prediction")

st.write(
    "Evaluate and compare trained machine learning classification "
    "models using Bank Marketing test data."
)


# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}


expected_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model Evaluation")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(model_files.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

st.sidebar.info(
    "Upload test data containing the original 16 features "
    "and the target column 'y'."
)


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------

if uploaded_file is None:

    st.info(
        "Upload a test CSV file from the sidebar to begin "
        "model evaluation."
    )

    st.subheader("Available Models")

    model_list = pd.DataFrame({
        "Classification Model": [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
            "Support Vector Machine"
        ]
    })

    st.dataframe(
        model_list,
        hide_index=True,
        use_container_width=True
    )

else:

    try:

        # -------------------------------------------------
        # Read Uploaded Data
        # -------------------------------------------------

        test_data = pd.read_csv(uploaded_file)

        required_columns = expected_features + ["y"]

        missing_columns = [
            column
            for column in required_columns
            if column not in test_data.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded file is missing the following "
                f"required columns: {missing_columns}"
            )

            st.stop()


        # -------------------------------------------------
        # Prepare Features and Target
        # -------------------------------------------------

        X_test = test_data[expected_features].copy()

        y_test = test_data["y"].map({
            "no": 0,
            "yes": 1
        })

        if y_test.isnull().any():

            st.error(
                "The target column 'y' must contain only "
                "'yes' and 'no' values."
            )

            st.stop()


        # -------------------------------------------------
        # Dataset Summary
        # -------------------------------------------------

        st.subheader("Test Data Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Test Records",
            f"{test_data.shape[0]:,}"
        )

        col2.metric(
            "Input Features",
            len(expected_features)
        )

        col3.metric(
            "Target Variable",
            "y"
        )


        with st.expander("Preview Uploaded Test Data"):

            st.dataframe(
                test_data.head(10),
                use_container_width=True
            )


        # -------------------------------------------------
        # Load Selected Model
        # -------------------------------------------------

        selected_model = joblib.load(
            model_files[selected_model_name]
        )


        # -------------------------------------------------
        # Generate Predictions
        # Important: No model retraining occurs here
        # -------------------------------------------------

        y_pred = selected_model.predict(X_test)


        if hasattr(selected_model, "predict_proba"):

            y_score = selected_model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = selected_model.decision_function(
                X_test
            )


        # -------------------------------------------------
        # Evaluation Metrics
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_score
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
             y_test,
            y_pred
        )


        # -------------------------------------------------
        # Selected Model
        # -------------------------------------------------

        st.divider()

        st.subheader("Selected Model")

        st.success(selected_model_name)


        # -------------------------------------------------
        # Metric Cards
        # -------------------------------------------------

        st.subheader("Model Performance")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        metric_col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        metric_col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        metric_col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        metric_col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        st.divider()

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_title(
            f"{selected_model_name} - Confusion Matrix"
        )

        st.pyplot(fig)


        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------

        st.subheader("Classification Report")
    report = classification_report(
            y_test,
            y_pred,
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )


        # -------------------------------------------------
        # Deployment Note
        # -------------------------------------------------

        st.info(
            "The selected saved model pipeline is loaded directly. "
            "The uploaded test data is used only for prediction and "
            "evaluation. The model is not retrained."
        )


    except Exception as error:

        st.error(
            f"Unable to evaluate the uploaded test data: {error}"
        )
