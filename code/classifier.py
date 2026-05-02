"""
Classifier — Identifies request type, product area, and initial status.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSIFIER_PROMPT, VALID_REQUEST_TYPES, VALID_STATUSES
from llm_client import call_llm_json
from logger import log


def classify(issue: str, subject: str, company: str, issue_id: str = None) -> dict:
    """
    Classify a support ticket.
    Returns: {"request_type", "product_area", "status", "company_inferred"}
    """
    prompt = CLASSIFIER_PROMPT.format(issue=issue, subject=subject, company=company)
    result = call_llm_json(prompt)

    # Validate
    req_type = result.get("request_type", "product_issue").lower().strip()
    if req_type not in VALID_REQUEST_TYPES:
        req_type = _fuzzy_match(req_type, VALID_REQUEST_TYPES, "product_issue")

    status = result.get("status", "replied").lower().strip()
    if status not in VALID_STATUSES:
        status = "escalated" if "escal" in status else "replied"

    product_area = result.get("product_area", "general-help").lower().strip()
    company_inferred = result.get("company_inferred", company or "").strip()

    classification = {
        "request_type": req_type,
        "product_area": product_area,
        "status": status,
        "company_inferred": company_inferred,
    }

    log("CLASSIFIER", f"{classification}", issue_id)
    return classification


def _fuzzy_match(value: str, valid: list, default: str) -> str:
    v = value.replace("_", "").replace("-", "").replace(" ", "")
    for opt in valid:
        if v in opt.replace("_", "") or opt.replace("_", "") in v:
            return opt
    return default
