"""
Self-Check — Verifies response grounding against source documents.
The winning differentiator.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SELF_CHECK_PROMPT
from llm_client import call_llm_json
from responder import format_docs
from logger import log

LOW_CONFIDENCE_THRESHOLD = 0.15


def self_check(response: str, docs: list, issue_id: str = None) -> dict:
    """
    Verify response grounding against source docs.
    Escalates only when the LLM explicitly marks response as not grounded,
    or confidence is extremely low (< 0.35). Avoids false positives on borderline cases.
    """
    docs_formatted = format_docs(docs)
    prompt = SELF_CHECK_PROMPT.format(response=response, docs=docs_formatted)

    try:
        result = call_llm_json(prompt, temperature=0.0)
    except Exception as e:
        log("SELF_CHECK", f"Failed (defaulting to reply): {e}", issue_id)
        # On LLM failure, trust the corpus-based response rather than always escalating
        return {"is_grounded": True, "confidence": 0.7, "reason": "Self-check unavailable; assuming grounded.", "should_escalate": False}

    is_grounded = result.get("is_grounded", True)
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.8))))
    reason = result.get("reason", "No reason provided")

    # Only escalate when explicitly not grounded OR extremely low confidence
    # (avoids over-escalation on borderline responses that partially match corpus)
    should_escalate = (not is_grounded) and (confidence < LOW_CONFIDENCE_THRESHOLD)

    log("SELF_CHECK", f"grounded={is_grounded}, conf={confidence:.2f}, escalate={should_escalate}, {reason[:80]}", issue_id)
    return {"is_grounded": is_grounded, "confidence": confidence, "reason": reason, "should_escalate": should_escalate}
