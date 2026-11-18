
import streamlit as st
import joblib
import re
from urllib.parse import urlparse
from pypdf import PdfReader
from docx import Document


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Job Scam Detector",
    page_icon="🔍",
    layout="wide"
)


# ==========================================
# 2. LOAD MODELS
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
# 3. RISK INDICATORS
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
# 4. URL EXTRACTION
# ==========================================

def extract_urls(text):

    urls = re.findall(
        r'https?://[^\s<>"\']+',
        text,
        flags=re.IGNORECASE
    )

    cleaned_urls = []

    for url in urls:

        url = url.rstrip(".,;:!?)]}")

        if url not in cleaned_urls:

            cleaned_urls.append(url)

    return cleaned_urls


# ==========================================
# 5. URL SAFETY ANALYSIS
# ==========================================

def analyze_url(url):

    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower()

    if domain.startswith("www."):

        domain = domain[4:]


    findings = []


    # HTTPS CHECK

    if parsed_url.scheme.lower() == "https":

        https_status = "Yes"

    else:

        https_status = "No"

        findings.append(
            "Uses HTTP instead of secure HTTPS"
        )


    # URL SHORTENER CHECK

    shortened_domains = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "shorturl.at",
        "is.gd",
        "cutt.ly",
        "rb.gy",
        "rebrand.ly",
        "ow.ly",
        "buff.ly"
    ]

    is_shortened = any(
        domain == short_domain
        or domain.endswith("." + short_domain)
        for short_domain in shortened_domains
    )

    if is_shortened:

        findings.append(
            "Uses a URL shortening service"
        )


    # IP ADDRESS CHECK

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    if re.match(ip_pattern, domain):

        findings.append(
            "Website uses an IP address instead of a domain name"
        )


    # MULTIPLE HYPHENS CHECK

    if domain.count("-") >= 3:

        findings.append(
            "Domain contains multiple hyphens"
        )


    # @ SYMBOL CHECK

    if "@" in url:

        findings.append(
            "URL contains an unusual @ symbol"
        )


    # URL LENGTH CHECK

    if len(url) > 150:

        findings.append(
            "URL is unusually long"
        )


    return {
        "url": url,
        "domain": domain,
        "https": https_status,
        "shortened": is_shortened,
        "findings": findings
    }


# ==========================================
# 6. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

h1 {
    color: #60a5fa !important;
    font-weight: bold !important;
}

h2, h3 {
    color: #f8fafc !important;
}

p, label, .stMarkdown {
    color: #e2e8f0 !important;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #64748b !important;
    border-radius: 10px !important;
    font-size: 16px !important;
}

.stTextInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 2px solid #64748b !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] {
    background-color: #172033;
    border: 1px solid #64748b;
    border-radius: 10px;
    padding: 12px;
}

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

.stButton > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}

[data-testid="stMetricValue"] {
    color: #60a5fa !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 7. HEADER
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
    "Analyze job postings using machine learning, "
    "explainable risk indicators, and URL safety analysis."
)


# ==========================================
# 8. HOW TO USE
# ==========================================

with st.expander("📘 How to Use This Application"):

    st.markdown("""
    **1. Paste Job Description:** Enter the complete job posting.

    **2. Upload a File:** Upload TXT, PDF, or DOCX.

    **3. Check a Link:** Enter a website URL directly.

    **4. Click the relevant analysis button.**

    **5. Review the results and risk indicators.**

    Note: Predictions are not proof of fraud.
    Always verify job offers independently.
    """)


st.divider()


# ==========================================
# 9. INPUT SELECTION
# ==========================================

st.header("📄 Submit a Job Posting")

selected_input = st.radio(
    "Choose your input method:",
    [
        "📝 Paste Job Description",
        "📁 Upload a File",
        "🔗 Check a Link"
    ],
    horizontal=True,
    key="input_choice"
)


job_text = ""


# ==========================================
# 10. PASTE JOB DESCRIPTION
# ==========================================

if selected_input == "📝 Paste Job Description":

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
# 11. UPLOAD FILE
# ==========================================

elif selected_input == "📁 Upload a File":

    uploaded_file = st.file_uploader(
        "Upload a job posting file",
        type=["txt", "pdf", "docx"],
        key="job_posting_file"
    )


    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()

        try:

            if file_name.endswith(".txt"):

                job_text = uploaded_file.getvalue().decode(
                    "utf-8",
                    errors="ignore"
                )


            elif file_name.endswith(".pdf"):

                pdf_reader = PdfReader(uploaded_file)

                pages = []

                for page in pdf_reader.pages:

                    page_text = page.extract_text()

                    if page_text:

                        pages.append(page_text)

                job_text = "\n".join(pages)


            elif file_name.endswith(".docx"):

                document = Document(uploaded_file)

                paragraphs = [
                    p.text
                    for p in document.paragraphs
                    if p.text.strip()
                ]

                table_text = []

                for table in document.tables:

                    for row in table.rows:

                        table_text.append(
                            " | ".join(
                                cell.text.strip()
                                for cell in row.cells
                            )
                        )

                job_text = "\n".join(
                    paragraphs + table_text
                )


            if job_text.strip():

                st.success(
                    "File uploaded successfully!"
                )

                with st.expander("👁️ Preview Extracted Text"):

                    st.text_area(
                        "Extracted text",
                        value=job_text,
                        height=200,
                        disabled=True
                    )

            else:

                st.warning(
                    "No readable text found. "
                    "Scanned PDFs may require OCR."
                )


        except Exception as e:

            st.error(
                f"Error reading file: {e}"
            )


# ==========================================
# 12. CHECK A LINK
# ==========================================

elif selected_input == "🔗 Check a Link":

    st.subheader("🌐 Website Link Safety Checker")

    st.write(
        "Enter a website URL to check "
        "for suspicious characteristics."
    )

    manual_url = st.text_input(
        "Enter website URL:",
        placeholder="https://example.com/job-apply",
        key="manual_url_input"
    )


# ==========================================
# 13. ANALYZE / CHECK LINK / REFRESH BUTTONS
# ==========================================

st.divider()

col1, col2 = st.columns([3, 1])


with col1:

    # LINK MODE: ONLY CHECK URL BUTTON

    if selected_input == "🔗 Check a Link":

        check_link_button = st.button(
            "🔍 Check URL Safety",
            use_container_width=True,
            type="primary",
            key="check_url_button"
        )

        analyze_button = False


    # JOB MODE: ONLY ANALYZE BUTTON

    else:

        analyze_button = st.button(
            "🔍 Analyze Job Posting",
            use_container_width=True,
            type="primary",
            key="analyze_job_button"
        )

        check_link_button = False


with col2:

    refresh_button = st.button(
        "🔄 Refresh",
        use_container_width=True,
        key="refresh_button"
    )


# ==========================================
# 14. REFRESH
# ==========================================

if refresh_button:

    st.session_state.clear()

    st.rerun()


# ==========================================
# 15. CHECK URL BUTTON ACTION
# ==========================================

if check_link_button:

    if not manual_url.strip():

        st.warning(
            "Please enter a URL first."
        )

    else:

        entered_url = manual_url.strip()

        if not re.match(
            r"^https?://",
            entered_url,
            flags=re.IGNORECASE
        ):

            entered_url = "https://" + entered_url


        parsed_url = urlparse(entered_url)


        if not parsed_url.netloc or "." not in parsed_url.netloc:

            st.error(
                "Please enter a valid website URL."
            )

        else:

            result = analyze_url(entered_url)

            st.subheader("📊 URL Analysis Results")

            st.code(
                result["url"],
                language="text"
            )

            col3, col4 = st.columns(2)

            with col3:

                st.metric(
                    "HTTPS Protection",
                    result["https"]
                )

            with col4:

                st.metric(
                    "URL Shortener",
                    "Detected"
                    if result["shortened"]
                    else "Not Detected"
                )


            if result["findings"]:

                st.warning(
                    "⚠️ Potential suspicious characteristics detected:"
                )

                for finding in result["findings"]:

                    st.write(
                        f"⚠️ {finding}"
                    )

            else:

                st.success(
                    "No obvious suspicious URL patterns detected."
                )


            st.info(
                "This is a basic URL pattern analysis. "
                "It does not confirm whether the website "
                "is safe or malicious."
            )


# ==========================================
# 16. MACHINE LEARNING JOB ANALYSIS
# ==========================================

if analyze_button:

    if not job_text or not job_text.strip():

        st.warning(
            "Please paste a job description "
            "or upload a readable file first."
        )

    else:

        cleaned_text = re.sub(
            r"\s+",
            " ",
            job_text
        ).strip()


        # TF-IDF

        text_features = tfidf_vectorizer.transform(
            [cleaned_text]
        )


        # PREDICTIONS

        svm_prediction = svm_model.predict(
            text_features
        )[0]

        lr_prediction = lr_model.predict(
            text_features
        )[0]

        fraud_probability = lr_model.predict_proba(
            text_features
        )[0][1]


        st.divider()

        st.header("📊 Job Analysis Results")


        # SVM RESULT

        if svm_prediction == 1:

            st.error(
                "🚨 Linear SVM: POTENTIALLY FRAUDULENT"
            )

        else:

            st.success(
                "✅ Linear SVM: LIKELY LEGITIMATE"
            )


        # FRAUD PROBABILITY

        st.subheader("📈 Estimated Fraud Probability")

        st.progress(
            float(fraud_probability)
        )

        st.write(
            f"Estimated fraud probability: "
            f"**{fraud_probability * 100:.2f}%**"
        )


        # MODEL COMPARISON

        st.subheader("🤖 Model Comparison")

        col3, col4 = st.columns(2)


        with col3:

            st.metric(
                "Linear SVM",
                "Fraudulent"
                if svm_prediction == 1
                else "Legitimate"
            )


        with col4:

            st.metric(
                "Logistic Regression",
                "Fraudulent"
                if lr_prediction == 1
                else "Legitimate"
            )


        if svm_prediction == lr_prediction:

            st.success(
                "Both models agree."
            )

        else:

            st.warning(
                "Models disagree. Please review carefully."
            )


        # RISK INDICATORS

        st.subheader("🔎 Explainable Risk Indicators")

        text_lower = cleaned_text.lower()

        found_any = False


        for category, patterns in risk_patterns.items():

            matched = [
                pattern
                for pattern in patterns
                if pattern in text_lower
            ]

            if matched:

                found_any = True

                st.warning(
                    f"**{category}:** "
                    + ", ".join(matched)
                )


        if not found_any:

            st.info(
                "No predefined risk indicators detected."
            )


        # URLS IN JOB DESCRIPTION

        st.subheader("🔗 Links Found in Job Description")

        detected_urls = extract_urls(cleaned_text)


        if detected_urls:

            for i, url in enumerate(
                detected_urls,
                start=1
            ):

                result = analyze_url(url)

                with st.expander(
                    f"Link {i}: {result['domain']}"
                ):

                    st.code(
                        result["url"],
                        language="text"
                    )

                    st.write(
                        f"HTTPS: {result['https']}"
                    )

                    st.write(
                        "URL Shortener: "
                        + (
                            "Yes"
                            if result["shortened"]
                            else "No"
                        )
                    )

                    if result["findings"]:

                        for finding in result["findings"]:

                            st.warning(
                                f"⚠️ {finding}"
                            )

                    else:

                        st.success(
                            "No obvious suspicious patterns detected."
                        )

        else:

            st.info(
                "No URLs found in the job description."
            )


        # DISCLAIMER

        st.divider()

        st.caption(
            "Disclaimer: This application is for research "
            "and educational purposes. Predictions and URL "
            "pattern checks are not proof of fraud."
        )


# ==========================================
# 17. FOOTER
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
