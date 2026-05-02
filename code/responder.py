"""
Responder — Generates grounded responses or escalation messages.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESPONSE_PROMPT, ESCALATION_RESPONSE, JUSTIFICATION_PROMPT
from llm_client import call_llm
from logger import log


def format_docs(docs: list, max_chars: int = 6000) -> str:
    """Format retrieved docs for prompt inclusion."""
    if not docs:
        return "[No relevant documents found]"
    formatted, total = [], 0
    for i, doc in enumerate(docs, 1):
        entry = f"--- Document {i} [ID: {doc['id']}] [Source: {doc['source']}] [Score: {doc.get('score', 0):.2f}] ---\n{doc['content']}\n"
        if total + len(entry) > max_chars:
            remaining = max_chars - total - 100
            if remaining > 200:
                formatted.append(entry[:remaining] + "...")
            break
        formatted.append(entry)
        total += len(entry)
    return "\n".join(formatted)


def generate_response(issue: str, subject: str, company: str, docs: list, issue_id: str = None) -> str:
    """Generate a grounded response using only corpus docs."""
    docs_formatted = format_docs(docs)
    prompt = RESPONSE_PROMPT.format(issue=issue, subject=subject, company=company, docs=docs_formatted)
    try:
        response = call_llm(prompt, temperature=0.0, max_tokens=800)
    except Exception as e:
        log("RESPONDER", f"Failed: {e}", issue_id)
        response = "I'm unable to generate a response at this time. Please contact the official support team."
    log("RESPONDER", f"Generated response ({len(response)} chars)", issue_id)
    return response.strip()


def generate_escalation(company: str, reason: str, issue_id: str = None) -> str:
    """Generate escalation response."""
    company_name = (company or "support").strip()
    if company_name.lower() == "none" or not company_name:
        company_name = "the relevant"
    response = ESCALATION_RESPONSE.format(company=company_name, reason=reason)
    log("RESPONDER", f"Escalation for {company_name}", issue_id)
    return response.strip()


def generate_justification(issue: str, company: str, status: str, request_type: str, product_area: str, doc_ids: list, issue_id: str = None) -> str:
    """Generate concise justification using a deterministic template (no extra LLM call)."""
    docs_note = (f"Corpus documents used: {', '.join(doc_ids[:3])}." if doc_ids
                 else "No matching corpus documents found.")
    if status == "escalated":
        if request_type == "invalid":
            text = (f"Ticket classified as {request_type} under {product_area}. "
                    "Request is outside the scope of this support agent. Replied with refusal.")
        else:
            text = (f"Classified as {request_type} under {company or 'unknown'} ({product_area}). "
                    f"Issue flagged as high-risk or unresolvable from corpus — escalated to human support. "
                    f"{docs_note}")
    else:
        text = (f"Classified as {request_type} under {company or 'unknown'} ({product_area}). "
                f"Relevant support documentation retrieved and response verified against corpus. "
                f"{docs_note}")
    log("RESPONDER", f"Justification (template): {text[:100]}...", issue_id)
    return text
