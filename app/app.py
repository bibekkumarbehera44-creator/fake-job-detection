
import streamlit as st
import joblib
import re
from pypdf import PdfReader
from docx import Document


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Job Scam Detector",
    page_icon="🔍",
    layout="wide"
)


# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    svm_model = joblib.load(
        "models/final_linear_svm.pkl"
    )

    lr_model = joblib.load(
        "models/final_logistic_regression.pkl"
    )

    tfidf_vectorizer = joblib.load(
        "models/final_tfidf_vectorizer.pkl"
    )

    return svm_model, lr_model, tfidf_vectorizer


svm_model, lr_model, tfidf_vectorizer = load_models()


# ==========================================
# RISK INDICATORS
# ==========================================

risk_patterns = {

    "Suspicious Application Links": [
        "using link",
        "apply using",
        "click link",
        "application link"
    ],

    "Earning-Related Language": [
        "earn",
        "money",
        "income",
        "cash",
        "payment"
    ],

    "Data Entry Related Patterns": [
        "data entry",
        "clerk",
        "administrative assistant",
        "receptionist"
    ],

    "Offshore or Unusual Employment Terms": [
        "offshore",
        "work home",
        "timejob"
    ]

}


# ==========================================
# CUSTOM CSS - IMPROVED VISIBILITY
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

/* Main title */
h1 {
    color: #60a5fa !important;
    font-weight: bold !important;
}

/* Section headings */
h2, h3 {
    color: #f8fafc !important;
}

/* General text */
p, label, .stMarkdown {
    color: #e2e8f0 !important;
}

/* Radio buttons */
div[role="radiogroup"] label {
    color: #ffffff !important;
}

/* Text area */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #64748b !important;
    border-radius: 10px !important;
    font-size: 16px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #172033;
    border: 1px solid #64748b;
    border-radius: 10px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 2px solid #60a5fa !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-weight: bold !important;
    font-size: 16px !important;
    min-height: 48px;
}

/* Button hover */
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    border: 2px solid #93c5fd !important;
    color: #ffffff !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
}

/* Expander */
.streamlit-expanderHeader {
    color: #93c5fd !important;
    font-weight: bold !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #2563eb !important;
}

/* Info, success, warning boxes */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    "<h1 style='text-align:center;'>🔍 AI Job Scam Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center; font-size:18px;">
    Explainable and Robust Machine Learning for
    Fraudulent Online Job Posting Detection
    </p>
    """,
    unsafe_allow_html=True
)

st.info(
    "Analyze online job postings using machine learning "
    "and interpretable risk indicators."
)


# ==========================================
# HOW TO USE
# ==========================================

with st.expander("📘 How to Use This Application"):

    st.markdown("""
    ### Follow these simple steps:

    **Step 1:** Select an input method:
    - Paste the job description.
    - Upload a `.txt`, `.pdf`, or `.docx` file.

    **Step 2:** Enter the job posting details.
    Include the job title, company profile, description,
    requirements, and benefits when available.

    **Step 3:** Click **Analyze Job Posting**.

    **Step 4:** Review the results:
    - Fraudulent or Legitimate prediction
    - Estimated fraud probability
    - Risk indicators found in the text
    - Comparison of predictions from two models

    **Important:** This application provides a machine
    learning prediction. Always independently verify
    the company and job offer before applying.
    """)


st.divider()


# ==========================================
# INPUT SECTION
# ==========================================

st.header("📄 Submit a Job Posting")

st.write("Choose how you want to submit the job posting:")


input_method = st.radio(
    "Input Method",
    ["Paste Job Description", "Upload a File"],
    horizontal=True,
    key="input_method"
)


job_text = ""


# ==========================================
# OPTION 1: PASTE TEXT
# ==========================================

if input_method == "Paste Job Description":

    job_text = st.text_area(
        "Paste the job posting details here:",
        height=250,
        placeholder=(
            "Enter job title, company profile, "
            "description, requirements, benefits, etc."
        ),
        key="job_description_text"
    )


# ==========================================
# OPTION 2: UPLOAD FILE
# ==========================================

else:

    uploaded_file = st.file_uploader(
        "Upload a job posting file",
        type=["txt", "pdf", "docx"],
        help="Supported formats: TXT, PDF, DOCX",
        key="job_posting_file"
    )


    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()

        try:

            # TXT FILE
            if file_name.endswith(".txt"):

                job_text = uploaded_file.getvalue().decode(
                    "utf-8",
                    errors="ignore"
                )


            # PDF FILE
            elif file_name.endswith(".pdf"):

                pdf_reader = PdfReader(uploaded_file)

                extracted_pages = []

                for page in pdf_reader.pages:

                    page_text = page.extract_text()

                    if page_text:
                        extracted_pages.append(page_text)

                job_text = "\n".join(extracted_pages)


            # DOCX FILE
            elif file_name.endswith(".docx"):

                document = Document(uploaded_file)

                paragraphs = [
                    paragraph.text
                    for paragraph in document.paragraphs
                    if paragraph.text.strip()
                ]

                # Also extract text from tables
                table_text = []

                for table in document.tables:

                    for row in table.rows:

                        row_values = [
                            cell.text.strip()
                            for cell in row.cells
                        ]

                        table_text.append(
                            " | ".join(row_values)
                        )

                job_text = "\n".join(
                    paragraphs + table_text
                )


            # DISPLAY EXTRACTED TEXT
            if job_text.strip():

                st.success("File uploaded and text extracted successfully!")

                with st.expander("👁️ Preview Extracted Job Description"):

                    st.text_area(
                        "Extracted Text",
                        value=job_text,
                        height=250,
                        disabled=True
                    )

            else:

                st.warning(
                    "No readable text was found in this file. "
                    "If it is a scanned PDF, OCR may be required."
                )


        except Exception as e:

            st.error(
                f"Error reading file: {e}"
            )


st.divider()


# ==========================================
# ANALYZE AND REFRESH BUTTONS
# ==========================================

col1, col2 = st.columns([3, 1])


with col1:

    analyze_button = st.button(
        "🔍 Analyze Job Posting",
        use_container_width=True,
        type="primary"
    )


with col2:

    refresh_button = st.button(
        "🔄 Refresh",
        use_container_width=True
    )


# ==========================================
# REFRESH FUNCTIONALITY
# ==========================================

if refresh_button:

    for key in [
        "job_description_text",
        "job_posting_file",
        "input_method"
    ]:

        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


# ==========================================
# ANALYSIS
# ==========================================

if analyze_button:

    if not job_text or not job_text.strip():

        st.warning(
            "⚠️ Please enter a job description "
            "or upload a readable file first."
        )

    else:

        st.divider()

        st.header("📊 Analysis Results")


        # ----------------------------------
        # TEXT PREPROCESSING
        # ----------------------------------

        cleaned_text = re.sub(
            r"\s+",
            " ",
            job_text
        ).strip()


        # ----------------------------------
        # TF-IDF TRANSFORMATION
        # ----------------------------------

        text_features = tfidf_vectorizer.transform(
            [cleaned_text]
        )


        # ----------------------------------
        # MODEL PREDICTIONS
        # ----------------------------------

        svm_prediction = svm_model.predict(
            text_features
        )[0]

        lr_prediction = lr_model.predict(
            text_features
        )[0]


        # Logistic Regression probability
        fraud_probability = lr_model.predict_proba(
            text_features
        )[0][1]


        # ----------------------------------
        # MAIN RESULT
        # ----------------------------------

        if svm_prediction == 1:

            st.error(
                "🚨 Linear SVM Prediction: "
                "POTENTIALLY FRAUDULENT JOB"
            )

        else:

            st.success(
                "✅ Linear SVM Prediction: "
                "LIKELY LEGITIMATE JOB"
            )


        # ----------------------------------
        # PROBABILITY
        # ----------------------------------

        st.subheader("Estimated Fraud Probability")

        st.progress(
            float(fraud_probability)
        )

        st.write(
            f"Estimated probability: "
            f"**{fraud_probability * 100:.2f}%**"
        )


        # ----------------------------------
        # MODEL COMPARISON
        # ----------------------------------

        st.subheader("🤖 Model Comparison")

        col3, col4 = st.columns(2)


        with col3:

            if svm_prediction == 1:

                st.metric(
                    "Linear SVM",
                    "Fraudulent"
                )

            else:

                st.metric(
                    "Linear SVM",
                    "Legitimate"
                )


        with col4:

            if lr_prediction == 1:

                st.metric(
                    "Logistic Regression",
                    "Fraudulent"
                )

            else:

                st.metric(
                    "Logistic Regression",
                    "Legitimate"
                )


        # ----------------------------------
        # MODEL AGREEMENT
        # ----------------------------------

        if svm_prediction == lr_prediction:

            st.success(
                "Both models agree on the prediction."
            )

        else:

            st.warning(
                "The models disagree. "
                "Please review the job posting carefully."
            )


        # ----------------------------------
        # EXPLAINABLE RISK INDICATORS
        # ----------------------------------

        st.subheader("🔎 Explainable Risk Indicators")

        text_lower = cleaned_text.lower()

        found_any = False


        for category, patterns in risk_patterns.items():

            matched_patterns = [
                pattern
                for pattern in patterns
                if pattern in text_lower
            ]


            if matched_patterns:

                found_any = True

                st.warning(
                    f"**{category}:** "
                    + ", ".join(matched_patterns)
                )


        if not found_any:

            st.info(
                "No predefined risk indicators "
                "were detected in the text."
            )


        # ----------------------------------
        # DISCLAIMER
        # ----------------------------------

        st.caption(
            "⚠️ Disclaimer: This tool is for research and "
            "educational purposes. Predictions are not proof "
            "of fraud. Please verify job postings independently."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <p style="text-align:center; color:#94a3b8;">
    AI Job Scam Detector | Explainable & Robust ML Research Project
    </p>
    """,
    unsafe_allow_html=True
)
