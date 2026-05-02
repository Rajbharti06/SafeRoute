"""
Risk Engine — Deterministic keyword-based risk detection.
No LLM calls here: avoids false positives on legitimate support questions.
The self-check layer handles quality gating for ambiguous cases.
"""
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HIGH_RISK_KEYWORDS, VAGUE_ESCALATION_PATTERNS
from logger import log


def assess_risk(issue: str, subject: str, company: str, classification: dict = None, issue_id: str = None) -> dict:
    """
    Deterministic risk assessment using keyword and pattern matching.
    Returns: {risk_level, should_escalate, reason, keyword_flags}
    """
    full_text = issue + " " + (subject or "")

    # Phase 0: Vague no-info report — escalate (corpus can't help).
    # Check issue only: subject text would break end-anchored patterns.
    if _is_vague_report(issue):
        result = {"risk_level": "high", "should_escalate": True,
                  "reason": "Vague issue with insufficient details to resolve from support corpus.",
                  "keyword_flags": ["vague_report"]}
        log("RISK_ENGINE", "HIGH (vague report)", issue_id)
        return result

    # Phase 1: Keyword + pattern scan
    keyword_flags = _keyword_scan(full_text)

    if keyword_flags:
        reason = f"High-risk indicators detected: {', '.join(keyword_flags[:3])}."
        result = {"risk_level": "high", "should_escalate": True,
                  "reason": reason, "keyword_flags": keyword_flags}
        log("RISK_ENGINE", f"HIGH (keywords: {keyword_flags[:3]})", issue_id)
        return result

    # No risk signals — let corpus retrieval + self-check decide
    result = {"risk_level": "low", "should_escalate": False,
              "reason": "No high-risk indicators found.", "keyword_flags": []}
    log("RISK_ENGINE", "LOW (no risk indicators)", issue_id)
    return result


def _keyword_scan(text: str) -> list:
    text_lower = text.lower()
    matches = []
    for kw in HIGH_RISK_KEYWORDS:
        if kw.lower() in text_lower:
            matches.append(kw)
    patterns = [
        (r'\bunauthorized\s+(?:access|transaction|charge)', "unauthorized_activity"),
        (r'\baccount\s+(?:hacked|compromised|breached)', "account_breach"),
        (r'\bidentity\s+(?:theft|stolen)', "identity_theft"),
        (r'\bcharge(?:d|back)?\s+(?:twice|again|double|unauthorized)', "duplicate_charge"),
        (r'\brefund\b', "refund_request"),
        (r'\b(?:payment|billing)\s+(?:issue|fail|error|problem|dispute)|(?:issue|problem)\s+with\s+(?:my\s+)?(?:payment|billing)', "payment_issue"),
        (r'\beven\s+though\s+i\s+am\s+not\s+the\s+(?:admin|owner|workspace)', "non_admin_access_request"),
        (r'\bnot\s+the\s+workspace\s+owner\s+or\s+admin', "non_admin_access_request"),
        (r'\brestore\s+(?:my\s+)?access\b', "access_restore_request"),
        # Security research — must reach security team, not corpus
        (r'\b(?:major\s+)?security\s+vulnerabilit(?:y|ies)\b', "security_vulnerability"),
        (r'\bbug\s+bounty\b', "bug_bounty"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text_lower) and label not in matches:
            matches.append(label)
    return matches


def _is_vague_report(text: str) -> bool:
    """Detect vague platform-outage or no-info reports that corpus cannot help with."""
    text_lower = text.lower().strip().replace('’', "'").replace('‘', "'")
    for pattern in VAGUE_ESCALATION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False
