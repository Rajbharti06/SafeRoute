"""
Logger — Structured logging for transparency.
Writes to both console and the AGENTS.md mandated log file.
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOG_DIR, LOG_FILE


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def init_log():
    """Initialize log file."""
    ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SafeRoute: Multi-Domain Support Triage Agent — Pipeline Log\n")
        f.write(f"Started: {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")


def log(section: str, content: str, issue_id: str = None):
    """Append a structured log entry."""
    ensure_log_dir()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    header = f"[{timestamp}]"
    if issue_id:
        header += f" [Issue: {issue_id}]"
    header += f" [{section}]"

    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{header}\n")
        f.write(f"  {content}\n")


def log_separator(issue_id: str = None):
    ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n" + "-" * 60 + "\n")
        if issue_id:
            f.write(f"Processing Issue: {issue_id}\n")
        f.write("-" * 60 + "\n")


def log_final_summary(total, replied, escalated, errors):
    ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write(f"Total: {total} | Replied: {replied} | Escalated: {escalated} | Errors: {errors}\n")
        f.write(f"Completed: {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n")
