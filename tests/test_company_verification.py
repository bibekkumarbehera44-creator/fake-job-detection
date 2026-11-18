import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.company_verification import assess_company


def test_genuine_registered_company(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTERNAL_COMPANY_CHECKS", "true")
    monkeypatch.setenv("MCA_API_KEY", "dummy-key")
    monkeypatch.setenv("BUSINESS_REGISTRY_PROVIDER", "MCA")
    monkeypatch.setattr(
        "src.company_verification._request_json",
        lambda *args, **kwargs: {"company_name": "Apex Systems India"},
    )

    assessment = assess_company(
        company_name="Apex Systems India",
        website_url="https://apexsystems.in",
        official_email="careers@apexsystems.in",
        company_registration_number="U12345TN2024PTC000123",
        company_address="Bengaluru, Karnataka, India",
        company_contact_number="+91-9876543210",
        recruiter_name="Nisha Reddy",
        recruiter_email="nisha@apexsystems.in",
        job_application_url="https://apexsystems.in/careers/apply",
        job_description="We are hiring a data analyst. No payment or security deposit is required.",
        allow_external_checks=True,
    )

    assert assessment.final_outcome == "GENUINE-LOOKING / LOW RISK"
    assert assessment.risk_score < 30
    assert assessment.check_results["registration_verification"].status == "verified"


def test_fake_company_with_incomplete_website():
    assessment = assess_company(
        company_name="NovaBridge Consultants",
        website_url="",
        official_email="info@novabridge-consulting.com",
        company_registration_number="",
        company_address="Dubai, UAE",
        company_contact_number="+971-555000111",
        recruiter_name="Aarav Khan",
        recruiter_email="aarav@novabridge-consulting.com",
        job_application_url="https://novabridge-consulting.com/apply",
        job_description="Immediate openings for remote roles with great pay."
    )

    assert assessment.risk_score >= 20
    assert assessment.check_results["website_authenticity"].status in {"failed", "inconclusive"}


def test_fake_company_using_free_email():
    assessment = assess_company(
        company_name="Summit Horizon Jobs",
        website_url="https://summithorizonjobs.com",
        official_email="jobs@gmail.com",
        company_registration_number="",
        company_address="India",
        company_contact_number="+91-9000000000",
        recruiter_name="Riya Gupta",
        recruiter_email="riya@gmail.com",
        job_application_url="https://summithorizonjobs.com/careers",
        job_description="We need candidates to pay a one-time registration fee."
    )

    assert assessment.check_results["email_authenticity"].status == "failed"
    assert assessment.check_results["payment_requests"].status == "failed"


def test_fake_company_with_lookalike_domain():
    assessment = assess_company(
        company_name="Microsoft",
        website_url="https://microsoft-jobs-support.net",
        official_email="recruiter@microsoft-jobs-support.net",
        company_address="India",
        company_contact_number="+1-555-123-4567",
        recruiter_name="Olivia James",
        recruiter_email="olivia@microsoft-jobs-support.net",
        job_application_url="https://microsoft-jobs-support.net/applications",
        job_description="Microsoft hiring process pays per application."
    )

    assert assessment.check_results["website_authenticity"].status in {"failed", "inconclusive"}
    assert any("lookalike" in reason.lower() or "impersonation" in reason.lower() for reason in assessment.reasons)


def test_complete_professional_looking_fake_job():
    assessment = assess_company(
        company_name="Northwind Analytics",
        website_url="https://northwindanalytics.com",
        official_email="careers@northwindanalytics.com",
        company_registration_number="",
        company_address="Austin, Texas",
        company_contact_number="+1-800-555-0199",
        recruiter_name="Daniel Price",
        recruiter_email="daniel@gmail.com",
        job_application_url="https://forms.gle/abc123",
        job_description="Northwind Analytics offers premium salaries and a quick hiring process. Please pay for the background verification before starting work.",
    )

    assert assessment.check_results["recruitment_behavior"].status == "failed"
    assert assessment.check_results["payment_requests"].status == "failed"


def test_company_impersonating_established_organization():
    assessment = assess_company(
        company_name="Google India Careers",
        website_url="https://googlejobs-support.in",
        official_email="careers@googlejobs-support.in",
        company_registration_number="",
        company_address="Bengaluru, India",
        company_contact_number="+91-800-100-2000",
        recruiter_name="Meera Singh",
        recruiter_email="meera@googlejobs-support.in",
        job_application_url="https://googlejobs-support.in/apply",
        job_description="Urgent hiring for senior engineering roles with a fast payout."
    )

    assert assessment.risk_score >= 45
    assert assessment.check_results["impersonation_reports"].status == "failed"


def test_genuine_new_startup_with_limited_online_information():
    assessment = assess_company(
        company_name="NovaForge Labs",
        website_url="",
        official_email="",
        company_registration_number="",
        company_address="Hyderabad, India",
        company_contact_number="+91-9876543213",
        recruiter_name="Ananya Nair",
        recruiter_email="ananya@novaforge.ai",
        job_application_url="https://novaforge.ai/jobs",
        job_description="We are a new startup hiring for engineering roles."
    )

    assert assessment.final_outcome == "UNABLE TO VERIFY"
    assert assessment.verification_confidence < 0.5


def test_registered_company_with_suspicious_recruitment_behavior(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTERNAL_COMPANY_CHECKS", "true")
    monkeypatch.setenv("MCA_API_KEY", "dummy-key")
    monkeypatch.setenv("BUSINESS_REGISTRY_PROVIDER", "MCA")
    monkeypatch.setattr(
        "src.company_verification._request_json",
        lambda *args, **kwargs: {"company_name": "Greenlattice Solutions"},
    )

    assessment = assess_company(
        company_name="Greenlattice Solutions",
        website_url="https://greenlattice.solutions",
        official_email="careers@greenlattice.solutions",
        company_registration_number="U12345TN2021PTC000987",
        company_address="Chennai, Tamil Nadu",
        company_contact_number="+91-9898989898",
        recruiter_name="Priya Raman",
        recruiter_email="priya@greenlattice.solutions",
        job_application_url="https://greenlattice.solutions/apply",
        job_description="We require a refundable security deposit and training fee before the interview.",
        allow_external_checks=True,
    )

    assert assessment.final_outcome in {"HIGH RISK / POTENTIAL FRAUD", "SUSPICIOUS / NEEDS VERIFICATION"}
    assert assessment.check_results["payment_requests"].status == "failed"


def test_external_verification_apis_unavailable(monkeypatch):
    monkeypatch.delenv("ENABLE_EXTERNAL_COMPANY_CHECKS", raising=False)
    monkeypatch.delenv("MCA_API_KEY", raising=False)
    monkeypatch.setattr(
        "src.company_verification._request_json",
        lambda *args, **kwargs: None,
    )

    assessment = assess_company(
        company_name="BluePeak Ventures",
        website_url="https://bluepeakventures.com",
        official_email="hr@bluepeakventures.com",
        company_registration_number="U99999KA2023PTC001234",
        company_address="Bengaluru, India",
        company_contact_number="+91-9888800001",
        recruiter_name="Aditya Shah",
        recruiter_email="aditya@bluepeakventures.com",
        job_application_url="https://bluepeakventures.com/careers/apply",
        job_description="No upfront fee. Email the hiring team.",
        allow_external_checks=True,
    )

    assert assessment.check_results["registration_verification"].status in {"unavailable", "inconclusive"}
    assert assessment.final_outcome in {"UNABLE TO VERIFY", "GENUINE-LOOKING / LOW RISK"}


def test_sophisticated_fake_company_passes_basic_checks(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTERNAL_COMPANY_CHECKS", "true")
    monkeypatch.setenv("MCA_API_KEY", "dummy-key")
    monkeypatch.setattr(
        "src.company_verification._request_json",
        lambda *args, **kwargs: None,
    )

    assessment = assess_company(
        company_name="Harborstar Labs",
        website_url="https://harborstarlabs.com",
        official_email="careers@harborstarlabs.com",
        company_registration_number="U11111MH2024PTC000010",
        company_address="Mumbai, Maharashtra",
        company_contact_number="+91-9822000011",
        recruiter_name="Sanjana Iyer",
        recruiter_email="sanjana@harborstarlabs.com",
        job_application_url="https://harborstarlabs.com/apply",
        job_description="We are hiring senior analysts. We offer flexible remote work and competitive salaries with no fees involved.",
        allow_external_checks=True,
    )

    assert assessment.final_outcome == "UNABLE TO VERIFY"
    assert assessment.verification_confidence < 0.7
