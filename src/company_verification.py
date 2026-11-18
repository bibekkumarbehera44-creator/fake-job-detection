from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "icloud.com",
    "protonmail.com",
    "aol.com",
}

RECOGNIZED_RECRUITMENT_PLATFORMS = {
    "linkedin.com",
    "greenhouse.io",
    "lever.co",
    "workday.com",
    "jobs.ashbyhq.com",
    "smartrecruiters.com",
    "icims.com",
    "jobvite.com",
    "hireology.com",
    "myworkdayjobs.com",
}

PAYMENT_KEYWORDS = {
    "registration fee",
    "interview fee",
    "security deposit",
    "training fee",
    "processing fee",
    "background check fee",
    "paid training",
    "deposit required",
    "buy equipment",
    "pay to proceed",
    "pay before interview",
    "mandatory fee",
    "pay for the background verification",
    "background verification fee",
    "must pay",
    "pay before starting work",
    "fee required",
    "deposit required before",
}

RISK_WEIGHTS = {
    "company_identity": 18,
    "website_authenticity": 18,
    "email_authenticity": 12,
    "registration_verification": 14,
    "contact_consistency": 12,
    "recruitment_behavior": 14,
    "payment_requests": 8,
    "impersonation_reports": 12,
}


@dataclass
class CheckResult:
    category: str
    status: str
    score: float
    weight: float
    evidence: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class CompanyAssessment:
    final_outcome: str
    risk_score: int
    verification_confidence: float
    positive_evidence: List[str] = field(default_factory=list)
    suspicious_indicators: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    check_results: Dict[str, CheckResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_outcome": self.final_outcome,
            "risk_score": self.risk_score,
            "verification_confidence": self.verification_confidence,
            "positive_evidence": list(self.positive_evidence),
            "suspicious_indicators": list(self.suspicious_indicators),
            "missing_information": list(self.missing_information),
            "reasons": list(self.reasons),
            "check_results": {
                key: {
                    "category": value.category,
                    "status": value.status,
                    "score": value.score,
                    "weight": value.weight,
                    "evidence": list(value.evidence),
                    "reasons": list(value.reasons),
                }
                for key, value in self.check_results.items()
            },
        }


def normalize_company_name(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    cleaned = re.sub(r"[^a-zA-Z0-9 &.-]", " ", cleaned)
    return " ".join(cleaned.split())


def normalize_domain(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).strip().lower().replace(" ", "")
    if "//" in text:
        parsed = urlparse(text if text.startswith(("http://", "https://")) else f"https://{text}")
        candidate = parsed.netloc or parsed.path
    else:
        candidate = text
    candidate = candidate.split("/", 1)[0]
    candidate = candidate.split("?", 1)[0]
    candidate = candidate.split("#", 1)[0]
    candidate = candidate.strip(".")
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate


def normalize_email(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def extract_email_domain(email: str) -> str:
    if not email or "@" not in email:
        return ""
    return email.rsplit("@", 1)[1].strip().lower()


def tokenize_text(value: Optional[str]) -> List[str]:
    if not value:
        return []
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(value).lower())
    return [part for part in text.split() if part]


def slugify(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def company_name_hits_domain(company_name: str, domain: str) -> bool:
    if not company_name or not domain:
        return False
    domain_base = normalize_domain(domain).split(".", 1)[0].lower()
    company_slug = slugify(company_name).replace("-", "")
    if not company_slug:
        return False

    if "jobs" in domain_base or "support" in domain_base or "careers" in domain_base or "hiring" in domain_base:
        if domain_base.startswith(company_slug[: max(3, min(8, len(company_slug)))]) or any(
            token in domain_base for token in [re.sub(r"[^a-z]", "", token.lower()) for token in tokenize_text(company_name) if token]
        ):
            return True
        return False

    if company_slug in domain_base:
        return True

    tokens = [re.sub(r"[^a-z]", "", token.lower()) for token in tokenize_text(company_name)]
    tokens = [token for token in tokens if token]
    return any(token in domain_base for token in tokens)


def lookalike_domain(company_name: str, domain: str) -> bool:
    if not company_name or not domain:
        return False
    domain_base = normalize_domain(domain).split(".", 1)[0].lower()
    suspicious_keywords = ("careers", "jobs", "hr", "verify", "support", "apply", "hiring")
    if any(keyword in domain_base for keyword in suspicious_keywords):
        name_base = slugify(company_name).replace("-", "")
        if not name_base:
            return True
        return domain_base != name_base
    return False


def is_free_email(email: str) -> bool:
    domain = extract_email_domain(email)
    return domain in FREE_EMAIL_DOMAINS


def validate_url(url: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "valid": False,
        "normalized": "",
        "status": "inconclusive",
        "issues": [],
        "https": False,
        "suspicious": False,
        "domain": "",
    }

    if not url:
        result["status"] = "inconclusive"
        result["issues"].append("No website URL supplied.")
        return result

    cleaned = str(url).strip()
    if not cleaned:
        result["status"] = "inconclusive"
        return result

    parsed = urlparse(cleaned if cleaned.startswith(("http://", "https://")) else f"https://{cleaned}")
    scheme = parsed.scheme.lower()
    host = parsed.hostname or ""

    if scheme not in {"http", "https"}:
        result["issues"].append("URL scheme is not supported.")
        result["status"] = "failed"
        return result

    if not host:
        result["issues"].append("URL host could not be parsed.")
        result["status"] = "failed"
        return result

    try:
        ipaddress.ip_address(host)
        result["suspicious"] = True
        result["issues"].append("URL uses a direct IP address instead of a registered domain.")
    except ValueError:
        pass

    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        result["suspicious"] = True
        result["issues"].append("URL points to a local or non-public network host.")

    if host.endswith(".local"):
        result["suspicious"] = True
        result["issues"].append("URL uses a local domain pattern.")

    if host.count(".") == 0 or len(host.split(".")[-1]) < 2:
        result["suspicious"] = True
        result["issues"].append("URL does not contain a valid public top-level domain.")

    normalized = normalize_domain(host)
    result["normalized"] = normalized
    result["domain"] = normalized
    result["https"] = scheme == "https"
    result["valid"] = True
    result["status"] = "verified" if scheme == "https" else "failed"
    if not result["https"]:
        result["issues"].append("Website does not use HTTPS.")
    return result


def _request_json(url: str, timeout: int = 8, headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    try:
        response = requests.get(url, timeout=timeout, headers=headers or {}, allow_redirects=False)
        if response.status_code >= 400:
            logger.warning("External lookup returned status code %s for %s", response.status_code, url)
            return None
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type and "text/plain" not in content_type:
            try:
                return response.json()
            except ValueError:
                logger.warning("External lookup for %s did not return JSON.", url)
                return None
        try:
            return response.json()
        except ValueError:
            return None
    except requests.RequestException as exc:
        logger.warning("External lookup failed for %s: %s", url, exc)
        return None


def _lookup_domain_age(domain: str) -> Tuple[str, Optional[int]]:
    if not domain:
        return "unavailable", None
    api_key = os.getenv("RDAP_API_KEY") or os.getenv("WHOIS_API_KEY")
    api_url = None
    if api_key:
        api_url = f"https://rdap.org/domain/{domain}?api_key={api_key}"
    else:
        api_url = f"https://rdap.org/domain/{domain}"
    payload = _request_json(api_url)
    if not payload:
        return "unavailable", None
    events = payload.get("events", []) if isinstance(payload, dict) else []
    if not events:
        return "inconclusive", None
    for event in events:
        if event.get("eventAction") == "last changed":
            date_value = event.get("eventDate")
            try:
                from datetime import datetime

                if date_value:
                    updated = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                    age_years = max(0.0, (datetime.utcnow() - updated.replace(tzinfo=None)).total_seconds() / (365.25 * 24 * 60 * 60))
                    return "verified", int(age_years)
            except ValueError:
                pass
    return "inconclusive", None


class CompanyVerifier:
    """Evidence-based company verification with conservative fallbacks."""

    def __init__(self, timeout: int = 8, allow_external_checks: bool = False):
        self.timeout = timeout
        self.allow_external_checks = allow_external_checks or bool(os.getenv("ENABLE_EXTERNAL_COMPANY_CHECKS"))

    def _new_check(self, category: str, weight: float, status: str, score: float, evidence: Sequence[str], reasons: Sequence[str]) -> CheckResult:
        return CheckResult(
            category=category,
            status=status,
            score=score,
            weight=weight,
            evidence=list(evidence),
            reasons=list(reasons),
        )

    def verify_business_registry(self, company_name: str, registration_number: Optional[str]) -> CheckResult:
        if not registration_number:
            return self._new_check(
                "registration_verification",
                RISK_WEIGHTS["registration_verification"],
                "unavailable",
                0.0,
                ["No registration number supplied."],
                ["Registration status was not checked because no registration number was provided."],
            )

        registry_key = os.getenv("MCA_API_KEY") or os.getenv("BUSINESS_REGISTRY_API_KEY")
        if not self.allow_external_checks or not registry_key:
            return self._new_check(
                "registration_verification",
                RISK_WEIGHTS["registration_verification"],
                "unavailable",
                0.15,
                ["No trusted registry API configured."],
                ["Business registry lookup was unavailable; no claim of registration verification is made."],
            )

        registry_url = (
            f"https://api.mca.gov.in/registry/{registration_number}"
            if "MCA" in (os.getenv("BUSINESS_REGISTRY_PROVIDER", "").upper())
            else f"https://example-verification-api.example/registry/{registration_number}"
        )
        payload = _request_json(registry_url, timeout=self.timeout, headers={"Authorization": f"Bearer {registry_key}"})
        if not payload:
            return self._new_check(
                "registration_verification",
                RISK_WEIGHTS["registration_verification"],
                "inconclusive",
                0.25,
                ["Registry API unavailable or returned no data."],
                ["Registry verification could not be completed; the company is not treated as verified."],
            )

        company_match = payload.get("company_name") or payload.get("name") or ""
        if company_match and normalize_company_name(company_name) and normalize_company_name(company_match) and normalize_company_name(company_match) != normalize_company_name(company_name):
            return self._new_check(
                "registration_verification",
                RISK_WEIGHTS["registration_verification"],
                "failed",
                0.9,
                ["Registered company name does not match the supplied company name."],
                ["The registration details do not match the claimed company identity."],
            )

        return self._new_check(
            "registration_verification",
            RISK_WEIGHTS["registration_verification"],
            "verified",
            0.05,
            ["Business registry lookup succeeded and matched identity."],
            ["A valid registry record was found and matched the supplied company details."],
        )

    def compare_company_identity(self, company_name: str, website_url: str, company_address: str, registration_number: Optional[str]) -> CheckResult:
        url_result = validate_url(website_url)
        domain = url_result.get("domain") or ""
        normalized_name = normalize_company_name(company_name)
        issues: List[str] = []
        evidence: List[str] = []

        if not normalized_name:
            return self._new_check(
                "company_identity",
                RISK_WEIGHTS["company_identity"],
                "inconclusive",
                0.45,
                ["Company name missing."],
                ["The company name is absent, so identity checks are inconclusive."],
            )

        if domain and company_name_hits_domain(normalized_name, domain):
            evidence.append(f"The company name aligns with the website domain '{domain}'.")
        else:
            issues.append("The website domain does not clearly match the claimed company name.")

        if company_address and normalized_name and len(company_address) > 15:
            evidence.append("Address information was supplied for review.")
        elif not company_address:
            issues.append("No address information was supplied for the identity check.")

        if registration_number:
            evidence.append("A company registration number was provided for validation.")

        if issues:
            score = 0.85 if url_result.get("status") == "failed" else 0.6
            return self._new_check(
                "company_identity",
                RISK_WEIGHTS["company_identity"],
                "failed" if score >= 0.7 else "inconclusive",
                score,
                evidence or ["Identity mismatch detected."],
                issues,
            )

        return self._new_check(
            "company_identity",
            RISK_WEIGHTS["company_identity"],
            "verified",
            0.1,
            evidence,
            ["Company name and website identity appear consistent based on the supplied information."],
        )

    def verify_website(self, company_name: str, website_url: str) -> CheckResult:
        result = validate_url(website_url)
        evidence: List[str] = []
        reasons: List[str] = []
        domain = result.get("domain") or normalize_domain(website_url)
        lookalike_detected = False

        if not website_url:
            return self._new_check(
                "website_authenticity",
                RISK_WEIGHTS["website_authenticity"],
                "inconclusive",
                0.4,
                ["No website was supplied."],
                ["No website evidence was available; website authenticity remains inconclusive."],
            )

        if result.get("status") == "failed":
            return self._new_check(
                "website_authenticity",
                RISK_WEIGHTS["website_authenticity"],
                "failed",
                0.9,
                result.get("issues", ["Website validation failed."]),
                result.get("issues", ["The website format or security settings are suspicious." ]),
            )

        if result.get("https"):
            evidence.append("The website is served over HTTPS.")
        else:
            reasons.append("The website does not use HTTPS.")

        if domain and lookalike_domain(company_name, domain):
            lookalike_detected = True
            reasons.append(f"The domain '{domain}' resembles a lookalike or impostor domain for '{company_name}'.")
        elif company_name and company_name_hits_domain(company_name, domain):
            evidence.append(f"The domain '{domain}' matches the claimed brand identity.")
        elif company_name:
            reasons.append(f"The website domain '{domain}' does not clearly align with the reported company identity.")

        if domain:
            try:
                if self.allow_external_checks:
                    status, age_value = _lookup_domain_age(domain)
                    if status == "verified":
                        evidence.append(f"Domain registration check succeeded; approximate domain age: {age_value} years.")
                    elif status in {"inconclusive", "unavailable"}:
                        reasons.append("Domain age could not be confirmed from a trusted source.")
            except Exception as exc:
                logger.warning("Unable to check domain age for %s: %s", domain, exc)
                reasons.append("Domain age check was unavailable.")

        if reasons and not evidence:
            return self._new_check(
                "website_authenticity",
                RISK_WEIGHTS["website_authenticity"],
                "failed" if lookalike_detected else "failed",
                0.9 if lookalike_detected else 0.75,
                reasons,
                reasons,
            )

        final_status = "failed" if lookalike_detected else ("verified" if not reasons else "inconclusive")
        final_score = 0.9 if lookalike_detected else (0.2 if not reasons else 0.55)
        return self._new_check(
            "website_authenticity",
            RISK_WEIGHTS["website_authenticity"],
            final_status,
            final_score,
            evidence or reasons,
            reasons or ["Website identity generally looks consistent."],
        )

    def verify_email(self, company_name: str, company_website_url: str, company_email: str) -> CheckResult:
        email = normalize_email(company_email)
        website_domain = normalize_domain(company_website_url)
        email_domain = extract_email_domain(email)

        if not email:
            return self._new_check(
                "email_authenticity",
                RISK_WEIGHTS["email_authenticity"],
                "inconclusive",
                0.4,
                ["No company email was supplied."],
                ["Email verification is inconclusive because no official email was provided."],
            )

        if is_free_email(email):
            return self._new_check(
                "email_authenticity",
                RISK_WEIGHTS["email_authenticity"],
                "failed",
                0.85,
                [f"The provided email uses the free service '{email_domain}'."],
                ["Free email addresses can be used by scammers, so this alone is not evidence of legitimacy."],
            )

        if website_domain and email_domain and website_domain == email_domain:
            return self._new_check(
                "email_authenticity",
                RISK_WEIGHTS["email_authenticity"],
                "verified",
                0.08,
                [f"The email domain '{email_domain}' matches the website domain '{website_domain}'."],
                ["The supplied company email aligns with the official website domain."],
            )

        return self._new_check(
            "email_authenticity",
            RISK_WEIGHTS["email_authenticity"],
            "inconclusive",
            0.45,
            [f"The email domain '{email_domain}' does not match the website domain '{website_domain}' or there was no website domain to compare."],
            ["The email could be legitimate but the domain mismatch warrants additional verification."],
        )

    def verify_contact_and_address(
        self,
        company_address: str,
        company_contact_number: str,
        recruiter_name: str,
        recruiter_email: str,
        historical_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CheckResult:
        evidence: List[str] = []
        reasons: List[str] = []

        if not company_address and not company_contact_number and not recruiter_email:
            return self._new_check(
                "contact_consistency",
                RISK_WEIGHTS["contact_consistency"],
                "inconclusive",
                0.3,
                ["No contact details were supplied."],
                ["No independent contact evidence was available for validation."],
            )

        if company_address and len(company_address) > 10:
            evidence.append("A company address was supplied.")
        else:
            reasons.append("No meaningful company address was supplied.")

        if company_contact_number and re.search(r"\d{7,}", company_contact_number):
            evidence.append("A contact number was supplied.")
        else:
            reasons.append("No usable company contact number was provided.")

        if recruiter_email and "@" in recruiter_email:
            evidence.append("Recruiter contact details were supplied.")

        if historical_records:
            for record in historical_records:
                if not isinstance(record, dict):
                    continue
                if record.get("company_contact_number") == company_contact_number:
                    reasons.append("The same phone number has appeared in previous suspicious records.")
                if record.get("recruiter_email") == recruiter_email:
                    reasons.append("The same recruiter email appears in prior suspicious records.")

        if reasons and not evidence:
            return self._new_check(
                "contact_consistency",
                RISK_WEIGHTS["contact_consistency"],
                "failed",
                0.8,
                reasons,
                reasons,
            )

        if reasons:
            return self._new_check(
                "contact_consistency",
                RISK_WEIGHTS["contact_consistency"],
                "inconclusive",
                0.55,
                evidence or ["Contact details are incomplete."],
                reasons,
            )

        return self._new_check(
            "contact_consistency",
            RISK_WEIGHTS["contact_consistency"],
            "verified",
            0.1,
            evidence,
            ["Company contact details appear present and internally consistent."],
        )

    def evaluate_recruitment_behavior(
        self,
        company_name: str,
        company_website_url: str,
        job_application_url: str,
        job_description: str,
        recruiter_email: str,
    ) -> CheckResult:
        evidence: List[str] = []
        reasons: List[str] = []
        if job_application_url:
            parsed = urlparse(job_application_url)
            host = (parsed.hostname or "").lower()
            site_domain = normalize_domain(company_website_url)
            if host and site_domain and host == site_domain:
                evidence.append("The application URL belongs to the company's own website domain.")
            elif host and host.endswith(".linkedin.com"):
                evidence.append("The application URL belongs to a recognized professional platform.")
            elif host and any(platform in host for platform in RECOGNIZED_RECRUITMENT_PLATFORMS):
                evidence.append(f"The application URL belongs to a recognized recruitment platform: {host}.")
            elif host:
                reasons.append(f"The application URL uses a domain outside the company's own site: {host}.")
        else:
            reasons.append("No application URL was supplied for verification.")

        if recruiter_email and extract_email_domain(recruiter_email) in FREE_EMAIL_DOMAINS:
            reasons.append("The recruiter uses a free email address, which raises caution.")

        description_text = (job_description or "").lower()
        if any(phrase in description_text for phrase in ["no payment", "no fees", "no fee", "not required", "no security deposit", "no registration fee", "no upfront fee", "no interview fee", "no deposit required", "no charge required"]):
            payment_hits = []
        else:
            payment_hits = [keyword for keyword in PAYMENT_KEYWORDS if keyword.lower() in description_text]
        if payment_hits:
            reasons.extend(f"Payment keyword detected: '{hit}'." for hit in payment_hits)

        if evidence and not reasons:
            return self._new_check(
                "recruitment_behavior",
                RISK_WEIGHTS["recruitment_behavior"],
                "verified",
                0.1,
                evidence,
                ["The recruitment process appears consistent with a standard application flow."],
            )

        if reasons:
            return self._new_check(
                "recruitment_behavior",
                RISK_WEIGHTS["recruitment_behavior"],
                "failed",
                0.8 if payment_hits else 0.6,
                evidence or ["Recruitment behavior was suspicious."],
                reasons,
            )

        return self._new_check(
            "recruitment_behavior",
            RISK_WEIGHTS["recruitment_behavior"],
            "inconclusive",
            0.35,
            ["Recruitment behavior could not be independently confirmed."],
            ["Application process details are incomplete or unavailable."],
        )

    def evaluate_payment_requests(self, job_description: str) -> CheckResult:
        if not job_description:
            return self._new_check(
                "payment_requests",
                RISK_WEIGHTS["payment_requests"],
                "inconclusive",
                0.25,
                ["No opportunity description was supplied."],
                ["Payment requests cannot be checked without an application description."],
            )

        text = job_description.lower()
        negated = any(phrase in text for phrase in ["no payment", "no fees", "no fee", "not required", "no security deposit", "no registration fee", "no upfront fee", "no interview fee", "no deposit required", "no charge required"]) 
        if negated:
            return self._new_check(
                "payment_requests",
                RISK_WEIGHTS["payment_requests"],
                "verified",
                0.05,
                ["The description explicitly says no fees or deposits are required."],
                ["No obvious charge or deposit request was found in the supplied job description."],
            )

        hits = [keyword for keyword in PAYMENT_KEYWORDS if keyword in text]
        if hits:
            return self._new_check(
                "payment_requests",
                RISK_WEIGHTS["payment_requests"],
                "failed",
                0.95,
                [f"Payment request keywords detected: {', '.join(hits)}."],
                ["The opportunity asks for a fee or deposit, a common scam signal."],
            )
        return self._new_check(
            "payment_requests",
            RISK_WEIGHTS["payment_requests"],
            "verified",
            0.05,
            ["No suspicious payment keywords were detected."],
            ["No obvious charge or deposit request was found in the supplied job description."],
        )

    def verify_impersonation(self, company_name: str, website_url: str, official_email: str, company_address: str) -> CheckResult:
        reasons: List[str] = []
        evidence: List[str] = []
        domain = normalize_domain(website_url)

        if not company_name and not website_url and not official_email:
            return self._new_check(
                "impersonation_reports",
                RISK_WEIGHTS["impersonation_reports"],
                "unavailable",
                0.1,
                ["No impersonation evidence source is configured."],
                ["No trustworthy reputation or impersonation signal was available."],
            )

        if domain and lookalike_domain(company_name, domain):
            reasons.append(f"The domain '{domain}' resembles an impersonation or lookalike of '{company_name}'.")

        if company_name and website_url:
            company_name_lower = company_name.lower()
            domain_lower = domain.lower()
            if "google" in company_name_lower and ("googlejobs" in domain_lower or "google" in domain_lower and "jobs" in domain_lower):
                reasons.append("The company name appears to impersonate a major organization through a suspicious domain name.")
            if "microsoft" in company_name_lower and ("microsoft" in domain_lower and ("jobs" in domain_lower or "support" in domain_lower)):
                reasons.append("The domain structure suggests an impersonation attack against a known brand.")

        if official_email and company_name and official_email.lower().endswith("gmail.com"):
            reasons.append("The official email domain is a free email provider and does not match a company-owned website.")

        if company_address and "india" in company_address.lower() and "support" in domain.lower():
            reasons.append("Address and domain do not align with a consistent corporate identity.")

        if reasons:
            return self._new_check(
                "impersonation_reports",
                RISK_WEIGHTS["impersonation_reports"],
                "failed",
                1.0,
                reasons,
                reasons,
            )

        return self._new_check(
            "impersonation_reports",
            RISK_WEIGHTS["impersonation_reports"],
            "unavailable",
            0.15,
            ["No reliable impersonation or scam-report source was available."],
            ["Reputation and impersonation sources were not configured; a clean result is not a proof of legitimacy."],
        )

    def assess_company(
        self,
        company_name: str,
        website_url: str,
        official_email: str,
        company_registration_number: Optional[str] = None,
        company_address: str = "",
        company_contact_number: str = "",
        recruiter_name: str = "",
        recruiter_email: str = "",
        job_application_url: str = "",
        job_description: str = "",
        historical_records: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CompanyAssessment:
        check_results: Dict[str, CheckResult] = {}
        missing_information: List[str] = []
        positive_evidence: List[str] = []
        suspicious_indicators: List[str] = []
        reasons: List[str] = []

        if not company_name:
            missing_information.append("Company name")
        if not website_url:
            missing_information.append("Company website URL")
        if not official_email:
            missing_information.append("Official company email")
        if not company_address:
            missing_information.append("Company address")
        if not company_contact_number:
            missing_information.append("Company contact number")

        identity_result = self.compare_company_identity(company_name, website_url, company_address, company_registration_number)
        check_results["company_identity"] = identity_result
        if identity_result.status == "verified":
            positive_evidence.extend(identity_result.evidence)
        if identity_result.status in {"failed", "inconclusive"}:
            suspicious_indicators.extend(identity_result.reasons)
            reasons.extend(identity_result.reasons)

        website_result = self.verify_website(company_name, website_url)
        check_results["website_authenticity"] = website_result
        if website_result.status == "verified":
            positive_evidence.extend(website_result.evidence)
        if website_result.status in {"failed", "inconclusive"}:
            suspicious_indicators.extend(website_result.reasons)
            reasons.extend(website_result.reasons)

        email_result = self.verify_email(company_name, website_url, official_email)
        check_results["email_authenticity"] = email_result
        if email_result.status == "verified":
            positive_evidence.extend(email_result.evidence)
        if email_result.status in {"failed", "inconclusive"}:
            suspicious_indicators.extend(email_result.reasons)
            reasons.extend(email_result.reasons)

        registration_result = self.verify_business_registry(company_name, company_registration_number)
        check_results["registration_verification"] = registration_result
        if registration_result.status == "verified":
            positive_evidence.extend(registration_result.evidence)
        if registration_result.status in {"failed", "inconclusive"}:
            suspicious_indicators.extend(registration_result.reasons)
            reasons.extend(registration_result.reasons)

        contact_result = self.verify_contact_and_address(company_address, company_contact_number, recruiter_name, recruiter_email, historical_records)
        check_results["contact_consistency"] = contact_result
        if contact_result.status == "verified":
            positive_evidence.extend(contact_result.evidence)
        if contact_result.status in {"failed", "inconclusive"}:
            suspicious_indicators.extend(contact_result.reasons)
            reasons.extend(contact_result.reasons)

        recruitment_result = self.evaluate_recruitment_behavior(company_name, website_url, job_application_url, job_description, recruiter_email)
        check_results["recruitment_behavior"] = recruitment_result
        if recruitment_result.status == "verified":
            positive_evidence.extend(recruitment_result.evidence)
        if recruitment_result.status in {"failed", "inconclusive"}:
            suspicious_indicators.extend(recruitment_result.reasons)
            reasons.extend(recruitment_result.reasons)

        payment_result = self.evaluate_payment_requests(job_description)
        check_results["payment_requests"] = payment_result
        if payment_result.status == "failed":
            suspicious_indicators.extend(payment_result.reasons)
            reasons.extend(payment_result.reasons)
        elif payment_result.status == "verified":
            positive_evidence.extend(payment_result.evidence)

        impersonation_result = self.verify_impersonation(company_name, website_url, official_email, company_address)
        check_results["impersonation_reports"] = impersonation_result
        if impersonation_result.status == "failed":
            suspicious_indicators.extend(impersonation_result.reasons)
            reasons.extend(impersonation_result.reasons)
        elif impersonation_result.status == "verified":
            positive_evidence.extend(impersonation_result.evidence)

        risk_score = 0
        for result in check_results.values():
            if result.status == "failed":
                risk_score += int(round(result.weight * result.score))
            elif result.status == "inconclusive":
                risk_score += int(round(result.weight * result.score * 0.7))
            elif result.status == "unavailable":
                risk_score += int(round(result.weight * result.score * 0.3))
        if any(term in reason.lower() for reason in reasons for term in ["lookalike", "impersonation", "impersonate", "major organization"]):
            risk_score += 15
        risk_score = max(0, min(100, risk_score))

        verified_count = sum(1 for result in check_results.values() if result.status == "verified")
        unavailable_count = sum(1 for result in check_results.values() if result.status == "unavailable")
        total_checks = max(1, len(check_results))
        confidence = (verified_count / total_checks) * 0.6
        if check_results.get("registration_verification", CheckResult("registration_verification", "unavailable", 0.0, 0.0)).status == "verified":
            confidence += 0.25
        else:
            confidence -= 0.18 * (unavailable_count / total_checks)
        if not company_name or not website_url or not official_email:
            confidence -= 0.12
        verification_confidence = min(1.0, max(0.0, round(confidence, 2)))

        registration_status = check_results.get("registration_verification", CheckResult("registration_verification", "unavailable", 0.0, 0.0)).status

        if risk_score >= 70:
            final_outcome = "HIGH RISK / POTENTIAL FRAUD"
        elif risk_score >= 35 or any("fee" in reason.lower() or "deposit" in reason.lower() or "lookalike" in reason.lower() for reason in reasons):
            final_outcome = "SUSPICIOUS / NEEDS VERIFICATION"
        elif verification_confidence < 0.4 and missing_information:
            final_outcome = "UNABLE TO VERIFY"
        elif risk_score <= 30 and verification_confidence >= 0.5 and registration_status == "verified":
            final_outcome = "GENUINE-LOOKING / LOW RISK"
        elif risk_score <= 30 and verification_confidence >= 0.5:
            final_outcome = "UNABLE TO VERIFY"
        else:
            final_outcome = "UNABLE TO VERIFY"

        deduplicated_reasons = []
        for reason in reasons:
            if reason not in deduplicated_reasons:
                deduplicated_reasons.append(reason)

        return CompanyAssessment(
            final_outcome=final_outcome,
            risk_score=risk_score,
            verification_confidence=verification_confidence,
            positive_evidence=list(dict.fromkeys(positive_evidence)),
            suspicious_indicators=list(dict.fromkeys(suspicious_indicators)),
            missing_information=list(dict.fromkeys(missing_information)),
            reasons=deduplicated_reasons,
            check_results=check_results,
        )


def assess_company(
    company_name: str,
    website_url: str,
    official_email: str,
    company_registration_number: Optional[str] = None,
    company_address: str = "",
    company_contact_number: str = "",
    recruiter_name: str = "",
    recruiter_email: str = "",
    job_application_url: str = "",
    job_description: str = "",
    historical_records: Optional[Sequence[Dict[str, Any]]] = None,
    allow_external_checks: Optional[bool] = None,
) -> CompanyAssessment:
    verifier = CompanyVerifier(
        timeout=8,
        allow_external_checks=bool(allow_external_checks) if allow_external_checks is not None else bool(os.getenv("ENABLE_EXTERNAL_COMPANY_CHECKS")),
    )
    return verifier.assess_company(
        company_name=company_name,
        website_url=website_url,
        official_email=official_email,
        company_registration_number=company_registration_number,
        company_address=company_address,
        company_contact_number=company_contact_number,
        recruiter_name=recruiter_name,
        recruiter_email=recruiter_email,
        job_application_url=job_application_url,
        job_description=job_description,
        historical_records=historical_records,
    )


def summarize_company_assessment(assessment: CompanyAssessment) -> Dict[str, Any]:
    return assessment.to_dict()


if __name__ == "__main__":
    sample = assess_company(
        company_name="Acme Labs",
        website_url="https://acmelabs.com",
        official_email="careers@acmelabs.com",
        company_registration_number="U12345TN2024PTC000001",
        company_address="Bengaluru, India",
        company_contact_number="+91-9876543210",
        recruiter_name="Asha Rao",
        recruiter_email="asha@acmelabs.com",
        job_application_url="https://acmelabs.com/careers/apply",
        job_description="We are hiring a data analyst. No fees involved.",
    )
    print(json.dumps(sample.to_dict(), indent=2, default=str))
