"""
SafeRoute: Multi-Domain Support Triage Agent
Main Pipeline — Orchestrates classification, retrieval, risk assessment,
response generation, and self-check verification.

Usage:
  python main.py                    # Process support_tickets.csv → output.csv
  python main.py --interactive      # Interactive chat mode
  python main.py --sample           # Run on sample_support_tickets.csv (dev/test)
"""
import os
import sys
import csv
import argparse
import datetime
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV
from logger import init_log, log, log_separator, log_final_summary
from classifier import classify
from retriever import retrieve_docs, get_retriever
from risk_engine import assess_risk
from responder import generate_response, generate_escalation, generate_justification
from self_check import self_check


def process_ticket(issue: str, subject: str, company: str, issue_id: str) -> dict:
    """
    Process a single support ticket through the full pipeline.

    Pipeline: PreScreen → Classify → Risk → [Escalation] → Retrieve → Respond → Self-Check → Decision
    """
    log_separator(issue_id)
    log("INPUT", f"Issue: {issue[:200]}... | Subject: {subject} | Company: {company}", issue_id)

    # Step 0: Pre-screen for prompt injection before LLM sees the text.
    # Catches multilingual attempts (French/Spanish) that regex misses in post-processing.
    if _is_injection_attempt(issue):
        response = _generate_invalid_response(issue)
        justification = "Pre-screening detected prompt injection attempt. Request refused without LLM processing."
        log("DECISION", "REPLIED (pre-screen: injection detected)", issue_id)
        return _build_result(issue, subject, company, response, company or "general-help", "replied", "invalid", justification)

    # Step 1: Classify
    classification = classify(issue, subject, company, issue_id)
    req_type = classification["request_type"]
    product_area = classification["product_area"]
    status = classification["status"]
    company_resolved = classification.get("company_inferred", company) or company or ""

    # Step 2: Handle invalid/off-topic immediately
    # Safety net: payment/billing/order issues must never be silently marked invalid —
    # they need to reach the risk engine even if the classifier misfired.
    if req_type == "invalid" and _has_payment_signal(issue + " " + (subject or "")):
        req_type = "product_issue"
        status = "escalated"
        log("DECISION", "OVERRIDING invalid→product_issue (payment signal detected)", issue_id)

    if req_type == "invalid":
        response = _generate_invalid_response(issue)
        justification = generate_justification(issue, company_resolved, "replied", req_type, product_area, [], issue_id)
        log("DECISION", "REPLIED (invalid/off-topic)", issue_id)
        return _build_result(issue, subject, company, response, product_area, "replied", req_type, justification)

    # Step 3: Assess risk
    risk = assess_risk(issue, subject, company_resolved, classification, issue_id)

    # Step 4: Explicitly high-risk → escalate without attempting an answer
    # Covers: fraud, billing disputes, identity theft, unauthorized charges, security breaches
    if risk["risk_level"] == "high":
        docs = retrieve_docs(issue, subject, company_resolved, top_k=3, issue_id=issue_id)
        doc_ids = [d["id"] for d in docs[:3]]
        response = generate_escalation(company_resolved, risk["reason"], issue_id)
        justification = generate_justification(issue, company_resolved, "escalated", req_type, product_area, doc_ids, issue_id)
        log("DECISION", f"ESCALATED (high risk: {risk['reason'][:80]})", issue_id)
        return _build_result(issue, subject, company, response, product_area, "escalated", req_type, justification)

    # Step 4b: Honor classifier's escalation judgment — catches vague/unanswerable reports
    # that the keyword engine misses (e.g. "none of the submissions are working",
    # "Claude has stopped working completely").
    if status == "escalated":
        response = generate_escalation(company_resolved, "Issue cannot be resolved from available support documentation — human review required.", issue_id)
        justification = generate_justification(issue, company_resolved, "escalated", req_type, product_area, [], issue_id)
        log("DECISION", "ESCALATED (classifier status=escalated)", issue_id)
        return _build_result(issue, subject, company, response, product_area, "escalated", req_type, justification)

    # Step 5: Retrieve relevant corpus docs
    docs = retrieve_docs(issue, subject, company_resolved, issue_id=issue_id)
    doc_ids = [d["id"] for d in docs[:5]]

    # Step 6: Generate grounded response
    response = generate_response(issue, subject, company_resolved, docs, issue_id)

    # Step 7: Verify response is grounded — escalate if not
    check = self_check(response, docs, issue_id)

    if check["should_escalate"]:
        response = generate_escalation(company_resolved, check["reason"], issue_id)
        justification = generate_justification(issue, company_resolved, "escalated", req_type, product_area, doc_ids, issue_id)
        log("DECISION", f"ESCALATED (ungrounded: {check['reason'][:60]})", issue_id)
        return _build_result(issue, subject, company, response, product_area, "escalated", req_type, justification)

    # Safe to reply
    justification = generate_justification(issue, company_resolved, "replied", req_type, product_area, doc_ids, issue_id)
    log("DECISION", f"REPLIED (grounded: {check['is_grounded']}, conf: {check['confidence']:.2f})", issue_id)
    return _build_result(issue, subject, company, response, product_area, "replied", req_type, justification)


def _has_payment_signal(text: str) -> bool:
    """Return True if text contains payment/billing/order signals that must not be silently dismissed."""
    import re
    t = text.lower()
    hard_tokens = ["payment", "billing", "invoice", "order id", "transaction", "refund", "charge", "chargeback"]
    if any(tok in t for tok in hard_tokens):
        return True
    return bool(re.search(r'\border\s+id\b|\bcs_(?:live|test)_', t))


def _generate_invalid_response(issue: str) -> str:
    """Return a targeted refusal based on the nature of the invalid request."""
    issue_lower = issue.lower()
    # Prompt injection: asking for internal rules, corpus, or system logic
    injection_signals = [
        "internal", "rules", "corpus", "retrieved", "decision logic", "prompt",
        "affiche", "toutes les", "interne", "montre", "show me your", "ignore previous",
        "override", "system prompt", "instructions", "reveal your",
    ]
    if any(sig in issue_lower for sig in injection_signals):
        return (
            "This request appears to ask for internal system information or attempts to alter "
            "system behavior. I'm unable to fulfill it. If you have a legitimate support "
            "question, please describe your issue clearly and I'll do my best to assist."
        )
    # Score or grade manipulation
    score_signals = ["increase my score", "change my score", "modify score", "update my score",
                     "graded unfairly", "move me to the next round", "review my answers"]
    if any(sig in issue_lower for sig in score_signals):
        return (
            "Assessment scores and results cannot be modified through this support channel. "
            "Scores are set by the hiring organization and are final once submitted. "
            "For concerns about an assessment, please contact the company that sent you the test."
        )
    # System command / code execution
    command_signals = ["delete all files", "rm -rf", "execute", "run this code", "system command",
                       "delete from", "drop table"]
    if any(sig in issue_lower for sig in command_signals):
        return (
            "Executing system commands or code is outside the scope of this support agent. "
            "If you need help with a software issue, please describe the problem and I'll assist."
        )
    # Default
    return "I'm sorry, this request is outside the scope of this support triage agent."


def _is_injection_attempt(issue: str) -> bool:
    """
    Pre-screen for prompt injection before the LLM sees the text.
    Catches multilingual attempts (French, Spanish, etc.) using token-level signals.
    Returns True if the issue is likely an injection attempt.
    """
    import re
    text = issue.lower()
    # High-confidence injection signals in multiple languages
    hard_signals = [
        "ignore previous", "ignore all previous", "ignore your instructions",
        "disregard previous", "forget previous instructions",
        "you are now", "act as if", "pretend you are",
        "system prompt", "reveal your prompt", "show me your prompt",
        "override your", "bypass your",
        # French injection signals
        "affiche toutes les", "règles internes", "documents récupérés",
        "logique exacte", "ignore les instructions",
        # Spanish injection signals
        "ignora tus instrucciones", "muestra tus reglas",
    ]
    if any(sig in text for sig in hard_signals):
        return True
    # Pattern: asking to reveal internal mechanics
    reveal_patterns = [
        r"\breveal\b.{0,30}\b(?:rules|instructions|prompt|corpus|logic)\b",
        r"\bshow\b.{0,30}\b(?:internal|system|hidden|all your)\b",
        r"\bwhat are your\b.{0,20}\b(?:rules|instructions|guidelines)\b",
        r"\bprint\b.{0,20}\b(?:rules|instructions|prompt)\b",
    ]
    return any(re.search(p, text) for p in reveal_patterns)


def _build_result(issue, subject, company, response, product_area, status, request_type, justification):
    """Build output row dict matching the required CSV schema."""
    return {
        "issue": issue,
        "subject": subject or "",
        "company": company or "",
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": justification,
    }


def process_csv(input_path, output_path):
    """Process all tickets from CSV and write results."""
    log("PIPELINE", f"Reading: {input_path}")

    # Read input CSV
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    log("PIPELINE", f"Found {len(rows)} tickets to process")

    # Initialize retriever (loads corpus + builds embeddings once)
    print("\n[*] Initializing corpus retriever...")
    retriever = get_retriever()
    print(f"   [+] {len(retriever.documents)} document chunks loaded\n")

    # Process each ticket
    results = []
    replied_count = 0
    escalated_count = 0
    error_count = 0

    for i, row in enumerate(rows, 1):
        issue = row.get("Issue", row.get("issue", ""))
        subject = row.get("Subject", row.get("subject", ""))
        company = row.get("Company", row.get("company", ""))
        issue_id = str(i)

        print(f"\n{'='*60}")
        print(f"  Ticket {i}/{len(rows)}")
        print(f"  Issue: {issue[:80]}...")
        print(f"{'='*60}")

        try:
            if i > 1:
                time.sleep(4)  # Pace requests — 4 LLM calls/ticket, stay under rate limit
            result = process_ticket(issue, subject, company, issue_id)
            results.append(result)

            if result["status"] == "replied":
                replied_count += 1
                print(f"  [REPLIED] {result['request_type']} | {result['product_area']}")
            else:
                escalated_count += 1
                print(f"  [ESCALATED] {result['request_type']} | {result['product_area']}")

        except Exception as e:
            error_count += 1
            print(f"  [ERROR]: {e}")
            log("ERROR", f"Ticket {issue_id}: {e}")
            results.append(_build_result(
                issue, subject, company,
                "Unable to process. Please contact support directly.",
                "general", "escalated", "product_issue",
                f"Processing error: {str(e)}"
            ))

    # Write output CSV
    _write_output(output_path, results)
    log_final_summary(len(rows), replied_count, escalated_count, error_count)

    print(f"\n{'='*60}")
    print(f"  DONE! {len(rows)} tickets processed")
    print(f"  Replied: {replied_count} | Escalated: {escalated_count} | Errors: {error_count}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")


def _write_output(output_path, results):
    """Write results to output CSV in the exact required format."""
    fieldnames = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    log("OUTPUT", f"Wrote {len(results)} results to {output_path}")


def interactive_mode():
    """Interactive chat mode for testing."""
    print("\n" + "="*60)
    print("SafeRoute: Multi-Domain Support Triage Agent")
    print("   Interactive Mode -- type 'quit' to exit")
    print("="*60 + "\n")

    print("[*] Loading corpus...")
    retriever = get_retriever()
    print(f"[+] {len(retriever.documents)} document chunks loaded\n")

    counter = 0
    while True:
        try:
            issue = input("\nIssue: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not issue or issue.lower() in ("quit", "exit", "q"):
            break

        subject = input("Subject (optional): ").strip()
        company = input("Company (HackerRank/Claude/Visa/None): ").strip() or "None"

        counter += 1
        result = process_ticket(issue, subject, company, f"chat_{counter}")

        print(f"\n{'─'*50}")
        print(f"Type: {result['request_type']} | Area: {result['product_area']}")
        print(f"Status: {result['status']}")
        print(f"{'─'*50}")
        print(f"\nResponse:\n{result['response']}")
        print(f"\nJustification:\n{result['justification']}")
        print(f"{'─'*50}")


def main():
    parser = argparse.ArgumentParser(description="SafeRoute: Multi-Domain Support Triage Agent")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--sample", "-s", action="store_true", help="Run on sample tickets (dev mode)")
    parser.add_argument("--input", type=str, default=None, help="Custom input CSV path")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    args = parser.parse_args()

    init_log()

    print("\n" + "="*60)
    print("SafeRoute: Multi-Domain Support Triage Agent")
    print(f"   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if args.interactive:
        interactive_mode()
    else:
        if args.sample:
            input_path = str(SAMPLE_CSV)
        else:
            input_path = args.input or str(INPUT_CSV)
        output_path = args.output or str(OUTPUT_CSV)

        if not os.path.exists(input_path):
            print(f"\n[!] Input not found: {input_path}")
            sys.exit(1)

        process_csv(input_path, output_path)


if __name__ == "__main__":
    main()
