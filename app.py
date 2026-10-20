
import json
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION AND WEBSITE DESIGN
# =========================================================

st.set_page_config(
    page_title="Genetic Disorder Risk Assessment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    h1 {
        color: #17365d;
        font-weight: 700;
    }

    h2, h3 {
        color: #24527a;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dce5f0;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        background-color: white;
        padding: 22px;
        border: 1px solid #dce5f0;
        border-radius: 12px;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #24527a;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background-color: #eaf0f8;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. LOAD THE TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("genetic_disorder_model.joblib")


try:
    model = load_model()

    with open("model_info.json", "r", encoding="utf-8") as file:
        model_info = json.load(file)

except Exception as error:
    st.error("Unable to load the trained model or model information.")
    st.code(str(error))
    st.stop()


# =========================================================
# 3. PAGE HEADING AND EDUCATIONAL WARNING
# =========================================================

st.title("🧬 Explainable Genetic Disorder Risk Assessment System")

st.write(
    "This educational project demonstrates how a machine-learning "
    "model classifies sample dataset records and explains its decision."
)

st.warning(
    "Educational use only. This model is not medically validated "
    "and must not be used for diagnosis or treatment decisions."
)


# =========================================================
# 4. MODEL INFORMATION
# =========================================================

st.subheader("Model Information")

col1, col2 = st.columns(2)

col1.metric(
    "Algorithm",
    model_info.get("algorithm", "Decision Tree Classifier")
)

col2.metric(
    "Test Accuracy",
    f'{model_info.get("test_accuracy", 0.6000527):.2%}'
)

st.caption(
    "Test accuracy describes performance on the held-out dataset. "
    "It is not clinical accuracy."
)


# =========================================================
# 5. CATEGORICAL INPUT OPTIONS
# =========================================================

choices = {
    "Genes in mother's side": ["No", "Yes"],
    "Inherited from father": ["No", "Yes"],
    "Maternal gene": ["No", "Yes"],
    "Paternal gene": ["No", "Yes"],

    "Respiratory Rate (breaths/min)": [
        "Normal (30-60)",
        "Tachypnea"
    ],

    "Heart Rate (rates/min": [
        "Normal",
        "Tachycardia"
    ],

    "Parental consent": ["Yes"],
    "Follow-up": ["High", "Low"],
    "Gender": ["Ambiguous", "Female", "Male"],

    "Birth asphyxia": [
        "No",
        "No record",
        "Not available",
        "Yes"
    ],

    "Autopsy shows birth defect (if applicable)": [
        "No",
        "Not applicable",
        "Yes"
    ],

    "Place of birth": ["Home", "Institute"],

    "Folic acid details (peri-conceptional)": [
        "No",
        "Yes"
    ],

    "H/O serious maternal illness": ["No", "Yes"],

    "H/O radiation exposure (x-ray)": [
        "-",
        "No",
        "Not applicable",
        "Yes"
    ],

    "H/O substance abuse": [
        "-",
        "No",
        "Not applicable",
        "Yes"
    ],

    "Assisted conception IVF/ART": ["No", "Yes"],

    "History of anomalies in previous pregnancies": [
        "No",
        "Yes"
    ],

    "Birth defects": ["Multiple", "Singular"],

    "Blood test result": [
        "abnormal",
        "inconclusive",
        "normal",
        "slightly abnormal"
    ]
}


# =========================================================
# 6. NUMERIC INPUT SETTINGS
# =========================================================

numeric_settings = {
    "Patient Age": (0.0, 14.0, 7.0),

    "Blood cell count (mcL)": (
        4.146229815,
        5.60982897,
        4.9000352685
    ),

    "Mother's age": (18.0, 51.0, 35.0),
    "Father's age": (20.0, 64.0, 42.0),

    "No. of previous abortion": (
        0.0,
        4.0,
        2.0
    ),

    "White Blood cell count (thousand per microliter)": (
        3.0,
        12.0,
        7.465056402
    )
}


# These features were constant in the inspected training data.
# They are fixed here for this demonstration.

fixed_tests = {
    "Test 1": 0.0,
    "Test 2": 0.0,
    "Test 3": 0.0,
    "Test 4": 1.0,
    "Test 5": 0.0
}


# Default symptom values

symptom_defaults = {
    "Symptom 1": 1.0,
    "Symptom 2": 1.0,
    "Symptom 3": 1.0,
    "Symptom 4": 0.0,
    "Symptom 5": 0.0
}


# =========================================================
# 7. INPUT FORM
# =========================================================

st.subheader("Enter Sample Dataset Features")

st.caption(
    "Use sample values for this educational demonstration. "
    "Do not enter real patient-identifying information."
)

feature_names = list(model.feature_names_in_)
input_values = {}

with st.form("prediction_form"):

    st.write("### Sample Features")

    for feature in feature_names:

        # Categorical features
        if feature in choices:

            input_values[feature] = st.selectbox(
                feature,
                options=choices[feature],
                key=f"choice_{feature}"
            )

        # Numeric features
        elif feature in numeric_settings:

            minimum, maximum, default = numeric_settings[feature]

            # Whole-number inputs
            if feature in [
                "Patient Age",
                "Mother's age",
                "Father's age",
                "No. of previous abortion"
            ]:

                input_values[feature] = st.number_input(
                    feature,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1,
                    key=f"number_{feature}"
                )

            # Decimal-number inputs
            else:

                input_values[feature] = st.number_input(
                    feature,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=0.1,
                    format="%.4f",
                    key=f"number_{feature}"
                )

        # Fixed test features
        elif feature in fixed_tests:

            input_values[feature] = fixed_tests[feature]

            st.write(
                f"**{feature}:** {int(fixed_tests[feature])} "
                "(fixed to the value observed in the dataset)"
            )

        # Binary symptoms
        elif feature in symptom_defaults:

            input_values[feature] = st.selectbox(
                feature,
                options=[0.0, 1.0],
                index=int(symptom_defaults[feature]),
                format_func=lambda value: str(int(value)),
                key=f"symptom_{feature}"
            )

        # Handle an unexpected feature
        else:

            st.error(
                f"Input configuration missing for: {feature}"
            )

            input_values[feature] = None

    submitted = st.form_submit_button("Predict category")


# =========================================================
# 8. PREDICTION, MODEL SCORES AND EXPLANATION
# =========================================================

if submitted:

    if any(value is None for value in input_values.values()):

        st.error("Please check the input configuration.")

    else:

        input_df = pd.DataFrame(
            [[input_values[name] for name in feature_names]],
            columns=feature_names
        )

        try:

            # Make the prediction
            prediction = model.predict(input_df)[0]

            st.subheader("Model Prediction")

            st.success(
                f"Predicted dataset category: {prediction}"
            )

            st.info(
                "This is a machine-learning classification, not a "
                "medical diagnosis or a validated genetic-risk estimate."
            )

            # -------------------------------------------------
            # MODEL SCORES FOR ALL THREE CATEGORIES
            # -------------------------------------------------

            st.subheader("Model Prediction Scores")

            probabilities = model.predict_proba(input_df)[0]

            classifier = model.named_steps["classifier"]
            class_names = classifier.classes_

            probability_df = pd.DataFrame({
                "Category": class_names,
                "Model Score (%)": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                by="Model Score (%)",
                ascending=False
            )

            probability_df["Model Score (%)"] = (
                probability_df["Model Score (%)"].round(2)
            )

            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "These are scores from the trained classifier, not "
                "the probability that a person has a genetic disorder."
            )

            # -------------------------------------------------
            # EXPLAIN THE DECISION-TREE PREDICTION
            # -------------------------------------------------

            preprocessor = model.named_steps["preprocessor"]

            transformed_input = preprocessor.transform(input_df)

            if hasattr(transformed_input, "toarray"):
                transformed_row = transformed_input.toarray()[0]
            else:
                transformed_row = transformed_input[0]

            decision_path = classifier.decision_path(
                transformed_input
            )

            leaf_id = classifier.apply(transformed_input)[0]
            tree = classifier.tree_

            transformed_features = (
                preprocessor.get_feature_names_out()
            )

            node_indices = decision_path.indices[
                decision_path.indptr[0]:
                decision_path.indptr[1]
            ]

            st.subheader(
                "Why did the model make this prediction?"
            )

            st.write(
                "These rules show the path followed by the "
                "trained decision tree for this sample."
            )

            rule_count = 0

            for node_id in node_indices:

                if node_id == leaf_id:
                    continue

                feature_index = tree.feature[node_id]
                threshold = tree.threshold[node_id]

                feature_name = transformed_features[feature_index]
                value = float(transformed_row[feature_index])

                operator = "<=" if value <= threshold else ">"

                st.write(
                    f"- **{feature_name}**: {value:.3f} "
                    f"{operator} {threshold:.3f}"
                )

                rule_count += 1

            if rule_count == 0:

                st.write(
                    "The model reached a leaf node without split rules."
                )

            st.caption(
                "These rules explain the model's classification only. "
                "They do not establish medical causes or genetic risk."
            )

        except Exception as error:

            st.error(
                "The model could not process these inputs."
            )

            st.code(str(error))


# =========================================================
# 9. MODEL EVALUATION
# =========================================================

st.divider()

st.header("📊 Model Evaluation")

st.write(
    "The following metrics were recorded during the earlier "
    "evaluation of the trained model on its held-out test dataset."
)

col1, col2, col3 = st.columns(3)

col1.metric("Test Accuracy", "60.01%")
col2.metric("Macro F1-Score", "52%")
col3.metric("Train-Test Accuracy Gap", "3.00%")

st.caption(
    "These are saved evaluation results. They are not recalculated "
    "from the current form submission."
)


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

st.subheader("Confusion Matrix")

st.write("Rows represent actual categories.")
st.write("Columns represent categories predicted by the model.")

class_names_for_report = [
    "Mitochondrial genetic inheritance disorders",
    "Multifactorial genetic inheritance disorders",
    "Single-gene inheritance diseases"
]

confusion_values = [
    [1447, 61, 430],
    [107, 118, 172],
    [676, 92, 690]
]

confusion_df = pd.DataFrame(
    confusion_values,
    index=class_names_for_report,
    columns=class_names_for_report
)

st.dataframe(
    confusion_df,
    use_container_width=True
)


# =========================================================
# 11. CLASSIFICATION REPORT
# =========================================================

st.subheader("Classification Report")

report_df = pd.DataFrame({
    "Category": [
        "Mitochondrial genetic inheritance disorders",
        "Multifactorial genetic inheritance disorders",
        "Single-gene inheritance diseases"
    ],

    "Precision": [0.65, 0.44, 0.54],
    "Recall": [0.76, 0.31, 0.47],
    "F1-Score": [0.70, 0.36, 0.50],
    "Test Samples": [1938, 397, 1458]
})

st.dataframe(
    report_df,
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Class metrics are rounded. Macro F1 is the unweighted "
    "average of the F1-scores for the three categories."
)


# =========================================================
# 12. MODEL LIMITATIONS
# =========================================================

st.subheader("⚠️ Model Limitations")

st.markdown("""
- The model achieved approximately 60% test accuracy on the
  held-out dataset.
- Performance differs between the three categories.
- The multifactorial category had a lower F1-score of 0.36.
- The dataset contains missing values and an uneven number of
  examples across categories.
- Test 1–Test 5 had constant values in the inspected training
  data and are fixed in this demonstration.
- The model learns patterns from its training dataset. It does
  not establish medical causes or determine a person's actual
  genetic risk.
- Prediction scores are model outputs and are not validated
  clinical probabilities.
- This project is for educational demonstration only and must
  not be used for diagnosis or treatment decisions.
""")

st.divider()

st.caption(
    "Explainable Genetic Disorder Risk Assessment System | "
    "Educational machine-learning project"
)
