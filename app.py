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
# Application Theme
# ---------------------------------------------------------

st.markdown("""
<style>

/* -------------------------------------------------------
   Main Application
------------------------------------------------------- */

.stApp {
    background-color: #F7F8F7;
    color: #0A2F24;
}


/* -------------------------------------------------------
   Typography
------------------------------------------------------- */

h1 {
    color: #0A2F24;
    font-weight: 700;
}

h2 {
    color: #0A2F24;
    font-weight: 650;
}

h3 {
    color: #0A2F24;
}


/* -------------------------------------------------------
   Sidebar
------------------------------------------------------- */

[data-testid="stSidebar"] {
    background-color: #E7FFD9;
    border-right: 3px solid #3DCD58;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #0A2F24;
}


/* -------------------------------------------------------
   Metric Cards
------------------------------------------------------- */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DDE5DF;
    border-left: 6px solid #3DCD58;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0px 2px 7px rgba(10, 47, 36, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #0A2F24;
}

[data-testid="stMetricValue"] {
    color: #0A2F24;
    font-weight: 700;
}


/* -------------------------------------------------------
   Download and Action Buttons
------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {

    background-color: #3DCD58;
    color: #0A2F24;

    border: 1px solid #3DCD58;
    border-radius: 7px;

    font-weight: 650;

    transition: 0.2s;
}

.stButton > button:hover,
.stDownloadButton > button:hover {

    background-color: #0A2F24;
    color: #E7FFD9;

    border-color: #0A2F24;
}


/* -------------------------------------------------------
   File Uploader
------------------------------------------------------- */

[data-testid="stFileUploader"] {

    background-color: #FFFFFF;

    border-radius: 8px;

    padding: 5px;
}


/* -------------------------------------------------------
   Expander
------------------------------------------------------- */

[data-testid="stExpander"] {

    background-color: #FFFFFF;

    border: 1px solid #DDE5DF;

    border-radius: 8px;
}


/* -------------------------------------------------------
   Information Messages
------------------------------------------------------- */

[data-testid="stAlert"] {

    border-radius: 8px;
}


/* -------------------------------------------------------
   Dataframes
------------------------------------------------------- */

[data-testid="stDataFrame"] {

    border: 1px solid #DDE5DF;

    border-radius: 8px;
}


/* -------------------------------------------------------
   Divider
------------------------------------------------------- */

hr {

    border: none;

    border-top: 1px solid #DDE5DF;

}


/* -------------------------------------------------------
   Caption
------------------------------------------------------- */

.stCaption {

    color: #49645B;
}

</style>
""", unsafe_allow_html=True)    

# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------

st.title("Bank Term Deposit Subscription Prediction")

st.caption(
    "Machine Learning Model Evaluation Dashboard"
)

st.write(
    "Compare five trained classification models using "
    "the UCI Bank Marketing test dataset."
)

overview_col1, overview_col2, overview_col3 = st.columns(3)

overview_col1.metric(
    "Classification Models",
    "5"
)

overview_col2.metric(
    "Input Features",
    "16"
)

overview_col3.metric(
    "Problem Type",
    "Binary"
)

st.divider()


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
model_descriptions = {
    "Logistic Regression":
        "A linear probabilistic classification model used as a strong baseline.",

    "Decision Tree":
        "A tree-based classifier capable of learning non-linear relationships in the data.",

    "K-Nearest Neighbors":
        "An instance-based classifier that predicts classes using nearby observations.",

    "Gaussian Naive Bayes":
        "A probabilistic classifier based on Bayes' theorem and feature independence assumptions.",

    "Support Vector Machine":
        "A margin-based classifier that identifies an effective decision boundary between classes."
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

st.sidebar.markdown("---")

# ---------------------------------------------------------
# Sample Test Data Download
# ---------------------------------------------------------

st.sidebar.subheader("Sample Test Data")

try:
    with open("test_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Test Data",
            data=file,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True
        )

except FileNotFoundError:
    st.sidebar.warning(
        "Sample test data is currently unavailable."
    )

st.sidebar.info(
    "Upload a CSV containing the 16 input features "
    "and the target column 'y' for model evaluation."
)

# ---------------------------------------------------------
# Expected Data Format
# ---------------------------------------------------------

with st.sidebar.expander("Expected CSV Format & Columns"):

    st.markdown("""
The uploaded CSV must contain **16 input features**
and the target column **y**.

**Numerical features**
- age
- balance
- day
- duration
- campaign
- pdays
- previous

**Categorical features**
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- poutcome

**Target values**
- `yes`
- `no`

You can download the sample test dataset above and use it as a reference for the expected CSV structure.
""")
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

        st.caption(
            model_descriptions[selected_model_name]
        )

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
        # Detailed Model Evaluation
        # -------------------------------------------------

        st.divider()

        st.subheader("Detailed Model Evaluation")

        visual_col1, visual_col2 = st.columns([1, 1.4])


        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        with visual_col1:

        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        # Calculate row-wise percentages
        cm_percent = (
        cm.astype("float")
            / cm.sum(axis=1)[:, None]
            * 100
        )

        # Create labels containing count + percentage
        cm_labels = []

        for i in range(cm.shape[0]):
            row_labels = []
    
            for j in range(cm.shape[1]):
                row_labels.append(
                    f"{cm[i, j]:,}\n{cm_percent[i, j]:.1f}%"
                )
    
            cm_labels.append(row_labels)
    
        fig, ax = plt.subplots(figsize=(5, 4))
    
        sns.heatmap(
            cm,
            annot=cm_labels,
            fmt="",
            cmap="Greens",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            cbar=True,
            ax=ax
        )
    
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected_model_name)
    
        st.pyplot(fig)
    
        st.caption(
            "Each cell shows the number of records and the "
            "percentage within the corresponding actual class."
        )


        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------

        with visual_col2:

            st.markdown("#### Classification Report")

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
