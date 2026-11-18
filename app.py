
import re
import io
import html
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from pypdf import PdfReader
from docx import Document

from src.company_verification import assess_company


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🔍",
    layout="wide"
)


# =====================================================
# 2. CUSTOM COLOUR THEME
# =====================================================


st.markdown("""
<style>

/* Main dark background */
.stApp {
    background-color: #101827;
    color: #FFFFFF;
}

/* Headings */
h1, h2, h3, h4 {
    color: #75B9FF !important;
}

/* Normal text on dark background */
.stMarkdown p,
.stMarkdown li,
.stCaption {
    color: #E8EEF7 !important;
}

/* WHITE TEXT INPUT BACKGROUND: USE DARK FONT */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #172B42 !important;
    -webkit-text-fill-color: #172B42 !important;
}

/* Input placeholder */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}

/* WHITE DROPDOWN: USE DARK FONT */
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
}

[data-baseweb="select"] span,
[data-baseweb="select"] input {
    color: #172B42 !important;
    -webkit-text-fill-color: #172B42 !important;
}

/* Dropdown options */
[role="option"] {
    background-color: #FFFFFF !important;
    color: #172B42 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    color: #172B42 !important;
    border: 1px solid #75B9FF !important;
    border-radius: 10px;
    padding: 15px;
}

/* All text inside file uploader */
[data-testid="stFileUploader"] * {
    color: #172B42 !important;
}

/* Buttons */
.stButton button {
    background-color: #2878D0 !important;
    color: #FFFFFF !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #20344B !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #20344B !important;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
}

/* Radio and checkbox labels */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #FFFFFF !important;
}

/* Divider */
hr {
    border-color: #405873 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. FILE PATHS
# =====================================================

MODEL_PATH = "models/best_model_provisional.joblib"

TEXT_VECTORIZER_PATH = (
    "data/processed/features/text_vectorizer.joblib"
)

METADATA_ENCODER_PATH = (
    "data/processed/features/metadata_encoder.joblib"
)


# =====================================================
# 4. FEATURE COLUMNS
# =====================================================

CATEGORICAL_COLUMNS = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]


# =====================================================
# 5. LOAD MODEL AND PREPROCESSORS
# =====================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(
        TEXT_VECTORIZER_PATH
    )

    encoder = joblib.load(
        METADATA_ENCODER_PATH
    )

    return model, vectorizer, encoder


try:
    model, vectorizer, encoder = load_model_files()

except Exception as e:

    st.error(
        "Unable to load the model or preprocessing files."
    )

    st.code(str(e))

    st.stop()


# =====================================================
# 6. TEXT CLEANING
# =====================================================

def clean_text(text):

    text = str(text)

    text = html.unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =====================================================
# 7. DOCUMENT EXTRACTION
# =====================================================

def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    file_bytes = uploaded_file.getvalue()

    extracted_text = ""

    try:

        # PDF extraction
        if file_name.endswith(".pdf"):

            reader = PdfReader(
                io.BytesIO(file_bytes)
            )

            pages = []

            for page in reader.pages:

                pages.append(
                    page.extract_text() or ""
                )

            extracted_text = "\n".join(pages)

        # DOCX extraction
        elif file_name.endswith(".docx"):

            document = Document(
                io.BytesIO(file_bytes)
            )

            content = []

            for paragraph in document.paragraphs:

                if paragraph.text.strip():

                    content.append(
                        paragraph.text
                    )

            for table in document.tables:

                for row in table.rows:

                    row_text = " | ".join(
                        cell.text.strip()
                        for cell in row.cells
                    )

                    if row_text.strip():

                        content.append(row_text)

            extracted_text = "\n".join(content)

        # TXT extraction
        elif file_name.endswith(".txt"):

            for encoding in [
                "utf-8",
                "utf-8-sig",
                "cp1252",
                "latin-1"
            ]:

                try:

                    extracted_text = (
                        file_bytes.decode(encoding)
                    )

                    break

                except UnicodeDecodeError:

                    continue

        else:

            raise ValueError(
                "Only PDF, DOCX and TXT files are supported."
            )

    except Exception as e:

        raise ValueError(
            f"Unable to extract document text: {e}"
        )

    if not extracted_text.strip():

        raise ValueError(
            "No readable text was found in this file. "
            "Scanned PDFs may require OCR."
        )

    return extracted_text


# =====================================================
# 8. BUILD MODEL FEATURES
# =====================================================

def build_features(text, metadata):

    cleaned_text = clean_text(text)

    # TF-IDF features
    text_features = vectorizer.transform(
        [cleaned_text]
    )

    # Categorical features
    categorical_data = pd.DataFrame(
        [[metadata.get(col, "Unknown")
          for col in CATEGORICAL_COLUMNS]],
        columns=CATEGORICAL_COLUMNS
    )

    metadata_features = encoder.transform(
        categorical_data
    )

    # Numeric features
    numeric_values = np.array([[
        metadata.get("telecommuting", 0),
        metadata.get("has_company_logo", 0),
        metadata.get("has_questions", 0)
    ]])

    numeric_features = sp.csr_matrix(
        numeric_values
    )

    # Combine all features
    combined_features = sp.hstack(
        [
            text_features,
            metadata_features,
            numeric_features
        ],
        format="csr"
    )

    
    # Validate expected feature count
    # Validate expected feature count
    expected_features = getattr(model, "n_features_in_", None)

    if expected_features is None:
        raise ValueError(
        "Unable to determine the number of features expected by the model."
    )

    if combined_features is None:
        raise ValueError("Feature generation failed. No features were created.")

    feature_shape = combined_features.shape
    if feature_shape is None:
        raise ValueError("Unable to determine the generated feature dimensions.")

    actual_features = feature_shape[1]

    if actual_features != expected_features:
        raise ValueError(
        f"Feature mismatch! Model expects {expected_features} features, "
        f"but the application generated {actual_features} features."
    )

    return combined_features

# =====================================================
# 9. GET FEATURE NAMES
# =====================================================
# =====================================================
# 9. GET FEATURE NAMES
# =====================================================

def get_feature_names():

    try:

        text_names = list(
            vectorizer.get_feature_names_out()
        )

    except Exception:

        text_names = []

    try:

        metadata_names = list(
            encoder.get_feature_names_out(
                CATEGORICAL_COLUMNS
            )
        )

    except Exception:

        try:

            metadata_names = list(
                encoder.get_feature_names(
                    CATEGORICAL_COLUMNS
                )
            )

        except Exception:

            metadata_names = []

    all_names = (
        text_names
        + metadata_names
        + NUMERIC_COLUMNS
    )

    expected = model.n_features_in_

    if len(all_names) != expected:

        all_names = [
            f"Feature_{i}"
            for i in range(expected)
        ]

    return all_names


FEATURE_NAMES = get_feature_names()


# =====================================================
# 10. SHAP EXPLANATION
# =====================================================

def get_shap_explanation(features):

    try:

        import shap

        # Zero-reference background for explanation
        background = sp.csr_matrix(
            (1, features.shape[1])
        )

        explainer = shap.LinearExplainer(
            model,
            background
        )

        explanation = explainer.shap_values(
            features
        )

        if isinstance(explanation, list):

            if len(explanation) > 1:

                explanation = explanation[1]

            else:

                explanation = explanation[0]

        explanation = np.asarray(
            explanation
        )

        explanation = np.squeeze(
            explanation
        )

        if explanation.ndim > 1:

            explanation = explanation[0]

        explanation = explanation.astype(float)

        if len(explanation) != len(FEATURE_NAMES):

            raise ValueError(
                "SHAP output feature count does not match."
            )

        return explanation, "SHAP"

    except Exception as shap_error:

        # Fallback for binary linear models.
        # This is a linear feature-contribution explanation,
        # not a SHAP calculation.

        if hasattr(model, "coef_"):

            coefficients = np.asarray(
                model.coef_
            )

            if coefficients.shape[0] == 1:

                coefficients = coefficients[0]

            else:

                coefficients = coefficients[1]

            contributions = np.asarray(
                features.multiply(
                    coefficients
                ).toarray()
            ).flatten()

            return (
                contributions,
                "Linear feature contributions"
            )

        raise RuntimeError(
            f"Explanation unavailable: {shap_error}"
        )


# =====================================================
# 11. PAGE TITLE
# =====================================================

st.title("🔍 Fake Job Detection System")

st.write(
    "Enter a job description or upload a job file."
)

st.write(
    "The system will analyze the information and "
    "predict whether the job is FAKE or GENUINE."
)

st.divider()


# =====================================================
# 12. INPUT METHOD
# =====================================================

st.header("1. Enter or Upload Job Details")

input_method = st.radio(
    "Choose input method",
    [
        "Enter Job Description",
        "Upload Job File"
    ],
    horizontal=True
)

job_text = ""


if input_method == "Enter Job Description":

    job_text = st.text_area(
        "Enter Job Description",
        height=250,
        placeholder="Paste the complete job description here..."
    )


else:

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or TXT file",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded File: {uploaded_file.name}"
        )

        try:

            job_text = extract_text(
                uploaded_file
            )

            st.success(
                "Job information extracted successfully!"
            )

            with st.expander(
                "View Extracted Job Information",
                expanded=True
            ):

                st.text_area(
                    "Extracted Text",
                    value=job_text,
                    height=300,
                    disabled=True
                )

                st.caption(
                    f"Total extracted characters: "
                    f"{len(job_text)}"
                )

        except Exception as e:

            st.error(str(e))


# =====================================================
# 13. OPTIONAL METADATA
# =====================================================

st.subheader("Optional: Add Additional Job Details")

with st.expander(
    "Enter additional job information",
    expanded=False
):

    location = st.text_input(
        "Job Location",
        value="Unknown"
    )

    department = st.text_input(
        "Department",
        value="Unknown"
    )

    salary_range = st.text_input(
        "Salary Range",
        value="Unknown"
    )

    employment_type = st.text_input(
        "Employment Type",
        value="Unknown"
    )

    required_experience = st.text_input(
        "Required Experience",
        value="Unknown"
    )

    required_education = st.text_input(
        "Required Education",
        value="Unknown"
    )

    industry = st.text_input(
        "Industry",
        value="Unknown"
    )

    function_name = st.text_input(
        "Job Function",
        value="Unknown"
    )

    telecommuting = st.checkbox(
        "Remote job",
        value=False
    )

    has_company_logo = st.checkbox(
        "Company logo available",
        value=False
    )

    has_questions = st.checkbox(
        "Screening questions available",
        value=False
    )


metadata = {
    "location": location or "Unknown",
    "department": department or "Unknown",
    "salary_range": salary_range or "Unknown",
    "employment_type": employment_type or "Unknown",
    "required_experience": required_experience or "Unknown",
    "required_education": required_education or "Unknown",
    "industry": industry or "Unknown",
    "function": function_name or "Unknown",

    "telecommuting": int(telecommuting),
    "has_company_logo": int(has_company_logo),
    "has_questions": int(has_questions)
}

# =====================================================
# 14. COMPANY VERIFICATION DETAILS
# =====================================================

st.subheader("Company Verification Details")

with st.expander("Company Verification Details", expanded=False):

    company_name = st.text_input(
        "Company Name",
        value=""
    )

    company_website_url = st.text_input(
        "Company Website URL",
        value=""
    )

    official_company_email = st.text_input(
        "Official Company Email",
        value=""
    )

    company_registration_number = st.text_input(
        "Company Registration Number (optional)",
        value=""
    )

    company_address = st.text_area(
        "Company Address",
        value="",
        height=100
    )

    company_contact_number = st.text_input(
        "Company Contact Number",
        value=""
    )

    recruiter_name = st.text_input(
        "Recruiter's Name",
        value=""
    )

    recruiter_email = st.text_input(
        "Recruiter's Email",
        value=""
    )

    job_application_url = st.text_input(
        "Job Application URL",
        value=""
    )

    st.caption(
        "Optional details are not treated as proof of legitimacy by themselves. "
        "Missing information is handled conservatively and does not automatically "
        "classify a company as fraudulent."
    )


# =====================================================
# 15. PREDICTION BUTTON
# =====================================================

if st.button(
    "🔍 Check Job",
    use_container_width=True
):

    if not job_text.strip():

        st.warning(
            "Please enter a job description "
            "or upload a job file."
        )

    else:

        try:

            features = build_features(
                job_text,
                metadata
            )

            prediction = int(
                model.predict(features)[0]
            )

            # Decision score
            decision_score = None

            if hasattr(model, "decision_function"):

                decision_score = float(
                    np.asarray(
                        model.decision_function(
                            features
                        )
                    ).flatten()[0]
                )

            # Explanation
            explanation, explanation_type = (
                get_shap_explanation(features)
            )

            company_assessment = assess_company(
                company_name=company_name,
                website_url=company_website_url,
                official_email=official_company_email,
                company_registration_number=company_registration_number,
                company_address=company_address,
                company_contact_number=company_contact_number,
                recruiter_name=recruiter_name,
                recruiter_email=recruiter_email,
                job_application_url=job_application_url,
                job_description=job_text,
            )

            st.session_state["job_text"] = job_text

            st.session_state["metadata"] = metadata

            st.session_state["prediction"] = prediction

            st.session_state["decision_score"] = (
                decision_score
            )

            st.session_state["explanation"] = explanation

            st.session_state["explanation_type"] = (
                explanation_type
            )

            st.session_state["company_assessment"] = (
                company_assessment.to_dict()
            )

            st.session_state["prediction_done"] = True

            st.session_state["feature_names"] = (
                FEATURE_NAMES
            )

        except Exception as e:

            st.error(
                "Prediction failed. Please check your "
                "model and preprocessing files."
            )

            st.exception(e)


# =====================================================
# 15. DISPLAY PREDICTION
# =====================================================

if st.session_state.get(
    "prediction_done",
    False
):

    st.divider()

    st.header("2. Prediction Result")

    prediction = st.session_state["prediction"]

    if prediction == 1:

        st.error(
            "🚨 FAKE JOB PREDICTION"
        )

        st.write(
            "The machine learning model classified "
            "this job as potentially fraudulent."
        )

    else:

        st.success(
            "✅ GENUINE JOB PREDICTION"
        )

        st.write(
            "The machine learning model classified "
            "this job as genuine."
        )

    st.warning(
        "This is a model prediction, not a guarantee "
        "that a job is genuine or fraudulent. "
        "Always verify the employer independently."
    )

    score = st.session_state["decision_score"]

    if score is not None:

        st.subheader("Model Decision Score")

        st.metric(
            "Decision Score",
            f"{score:.4f}"
        )

        st.caption(
            "For a binary Linear SVM with classes [0, 1], "
            "positive values generally indicate class 1 "
            "(Fake), while negative values indicate "
            "class 0 (Genuine). This score is not a probability."
        )


    # =================================================
    # 16. COMPANY VERIFICATION DASHBOARD
    # =================================================

    st.divider()

    st.header("3. Company Verification & Fraud Risk Assessment")

    assessment = st.session_state.get("company_assessment", {})
    risk_score = assessment.get("risk_score", 0)
    verification_confidence = assessment.get("verification_confidence", 0.0)
    final_outcome = assessment.get("final_outcome", "UNABLE TO VERIFY")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric("ML Prediction", "Fake" if prediction == 1 else "Genuine")

    with metrics_col2:
        st.metric("ML Decision Score", f"{score:.4f}" if score is not None else "N/A")

    with metrics_col3:
        st.metric("Company Risk Score", f"{risk_score}/100")

    with metrics_col4:
        st.metric("Verification Confidence", f"{verification_confidence:.2f}")

    st.subheader("Company Verification Summary")
    st.write(f"Assessment: **{final_outcome}**")

    if final_outcome == "GENUINE-LOOKING / LOW RISK":
        st.success(final_outcome)
    elif final_outcome == "HIGH RISK / POTENTIAL FRAUD":
        st.error(final_outcome)
    elif final_outcome == "SUSPICIOUS / NEEDS VERIFICATION":
        st.warning(final_outcome)
    else:
        st.info(final_outcome)

    check_results = assessment.get("check_results", {})
    for check_name, result in check_results.items():
        status = str(result.get("status", "unavailable")).upper()
        if status == "VERIFIED":
            status_label = "✅ Verified"
        elif status == "FAILED":
            status_label = "❌ Unsuccessful"
        elif status == "INCONCLUSIVE":
            status_label = "⚠️ Inconclusive"
        else:
            status_label = "ℹ️ Unavailable"

        st.markdown(f"- **{check_name.replace('_', ' ').title()}**: {status_label}")

    with st.expander("Website verification result", expanded=False):
        st.write(check_results.get("website_authenticity", {}))

    with st.expander("Email verification result", expanded=False):
        st.write(check_results.get("email_authenticity", {}))

    with st.expander("Registration verification result", expanded=False):
        st.write(check_results.get("registration_verification", {}))

    with st.expander("Contact verification result", expanded=False):
        st.write(check_results.get("contact_consistency", {}))

    with st.expander("Suspicious indicators", expanded=False):
        indicators = assessment.get("suspicious_indicators", [])
        if indicators:
            for item in indicators:
                st.markdown(f"- {item}")
        else:
            st.write("No suspicious indicators were identified from the supplied evidence.")

    with st.expander("Positive verification evidence", expanded=False):
        evidence = assessment.get("positive_evidence", [])
        if evidence:
            for item in evidence:
                st.markdown(f"- {item}")
        else:
            st.write("No strong independent verification evidence was available.")

    with st.expander("Missing verification information", expanded=False):
        missing = assessment.get("missing_information", [])
        if missing:
            for item in missing:
                st.markdown(f"- {item}")
        else:
            st.write("No essential information was missing for this review.")

    with st.expander("Why Was This Company Flagged?", expanded=True):
        reasons = assessment.get("reasons", [])
        if reasons:
            for item in reasons:
                st.markdown(f"- {item}")
        else:
            st.write("No obvious risk factors emerged from the available information.")

    with st.expander("Final combined assessment", expanded=False):
        st.write(f"**Outcome:** {final_outcome}")
        st.write(
            "The final verdict combines the Linear SVM result with independent company "
            "verification evidence, unresolved uncertainty, and missing facts. "
            "A low-risk company cannot be declared genuine unless independent evidence supports it."
        )

    # =================================================
    # 17. SHAP EXPLANATION - TEXT ONLY
    # =================================================

    st.divider()

    st.header("4. SHAP Explanation")

    st.write(
        "This section explains which features influenced "
        "the machine learning prediction."
    )

    explanation = st.session_state["explanation"]

    explanation_type = st.session_state[
        "explanation_type"
    ]

    feature_names = st.session_state[
        "feature_names"
    ]

    st.caption(
        f"Explanation method: {explanation_type}"
    )

    top_indices = np.argsort(
        np.abs(explanation)
    )[::-1][:15]

    for index in top_indices:

        value = float(
            explanation[index]
        )

        feature = feature_names[index]

        if value > 0:

            direction = "Towards Fake"
            symbol = "🔴"

        elif value < 0:

            direction = "Towards Genuine"
            symbol = "🟢"

        else:

            direction = "Low Influence"
            symbol = "⚪"

        st.markdown(
            f"{symbol} **{feature}** — "
            f"{direction} "
            f"(Contribution: {value:.4f})"
        )

    if explanation_type == "SHAP":

        st.info(
            "Positive SHAP values push the Linear SVM "
            "decision score towards Fake, while negative "
            "values push it towards Genuine, assuming "
            "classes [0, 1] and the zero reference background."
        )

    else:

        st.info(
            "SHAP could not be calculated. The displayed "
            "values are linear feature contributions from "
            "the model, not SHAP values."
        )


    # =================================================
    # 18. EXPLANATION STABILITY
    # =================================================

    st.divider()

    st.header("5. Explanation Stability")

    st.write(
        "Explanation Stability checks whether important "
        "features remain similar when the job description "
        "is slightly changed."
    )

    st.write(
        "Enter a slightly modified job description below."
    )

    modified_text = st.text_area(
        "Modified Job Description",
        height=220,
        placeholder="Paste a slightly modified job description here..."
    )

    if st.button(
        "Check Explanation Stability",
        use_container_width=True
    ):

        if not modified_text.strip():

            st.warning(
                "Please enter a modified job description."
            )

        else:

            try:

                modified_features = build_features(
                    modified_text,
                    st.session_state["metadata"]
                )

                modified_explanation, modified_type = (
                    get_shap_explanation(
                        modified_features
                    )
                )

                original_explanation = explanation

                # Top 10 important features
                original_top = set(
                    np.argsort(
                        np.abs(original_explanation)
                    )[-10:]
                )

                modified_top = set(
                    np.argsort(
                        np.abs(modified_explanation)
                    )[-10:]
                )

                intersection = (
                    original_top & modified_top
                )

                union = (
                    original_top | modified_top
                )

                # Jaccard similarity
                if len(union) > 0:

                    jaccard = (
                        len(intersection)
                        / len(union)
                    )

                else:

                    jaccard = 0.0

                # Cosine similarity
                original_flat = np.asarray(
                    original_explanation
                ).flatten()

                modified_flat = np.asarray(
                    modified_explanation
                ).flatten()

                denominator = (
                    np.linalg.norm(original_flat)
                    * np.linalg.norm(modified_flat)
                )

                if denominator > 0:

                    cosine = float(
                        np.dot(
                            original_flat,
                            modified_flat
                        ) / denominator
                    )

                else:

                    cosine = 0.0

                # Display results
                st.subheader("Stability Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Jaccard Similarity",
                        f"{jaccard * 100:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Cosine Similarity",
                        f"{cosine:.4f}"
                    )

                st.subheader(
                    "Common Important Features"
                )

                if intersection:

                    for index in sorted(
                        intersection,
                        key=lambda i: abs(
                            original_explanation[i]
                        ),
                        reverse=True
                    ):

                        st.markdown(
                            f"• {feature_names[index]}"
                        )

                else:

                    st.write(
                        "No common important features found."
                    )

                st.subheader("Result Interpretation")

                st.write(
                    "Jaccard Similarity measures the overlap "
                    "between the top important features "
                    "in both explanations."
                )

                st.write(
                    "Cosine Similarity measures how similar "
                    "the two explanation vectors are in direction."
                )

                if jaccard >= 0.70:

                    st.success(
                        "The explanations have strong overlap "
                        "in their top features."
                    )

                elif jaccard >= 0.40:

                    st.warning(
                        "The explanations have moderate overlap."
                    )

                else:

                    st.warning(
                        "The explanations have low overlap."
                    )

                st.caption(
                    "These are comparisons between two inputs. "
                    "They do not guarantee overall model reliability. "
                    "The interpretation thresholds are simple "
                    "display guidelines, not established standards."
                )

            except Exception as e:

                st.error(
                    "Unable to calculate explanation stability."
                )

                st.exception(e)


    # =================================================
    # 19. SAFETY RECOMMENDATIONS
    # =================================================

    st.divider()

    st.header("6. Safety Recommendations")

    recommendations = [
        "Verify the company using its official website.",
        "Check the recruiter's email address.",
        "Avoid paying registration or interview fees.",
        "Be careful with unrealistic salary promises.",
        "Do not share OTPs or bank details.",
        "Contact the company directly to verify the offer."
    ]

    for recommendation in recommendations:

        st.markdown(
            f"🔹 {recommendation}"
        )

    st.info(
        "Even if the model predicts GENUINE, "
        "independently verify the job before accepting an offer."
    )


# =====================================================
# 19. FOOTER
# =====================================================

st.divider()

st.caption(
    "Fake Job Detection System | "
    "Machine Learning Based Recruitment Fraud Detection"
)
