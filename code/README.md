# SafeRoute: Multi-Domain Support Triage Agent

A trust-first, terminal-based AI agent that triages support tickets across **HackerRank**, **Claude**, and **Visa** ecosystems using only the provided corpus.

## Architecture

```
Ticket Input (issue, subject, company)
         ↓
    Classifier (LLM → request_type, product_area, initial status)
         ↓
  [invalid?] → Targeted refusal response → Output
         ↓
    Risk Engine (deterministic keyword + pattern scan)
         ↓
  [high risk?] → Escalation response → Output
         ↓
    Retriever (FAISS vector search, ~4700+ corpus chunks)
         ↓
    Response Generator (strictly grounded in corpus only)
         ↓
    Self-Check Verifier (response ↔ source docs validation)
         ↓
  [ungrounded?] → Escalation response → Output
         ↓
    Output (replied + justification)
```

## Design Principles

> "A wrong answer in support is worse than no answer."

- **Corpus-Only Responses**: Zero parametric knowledge — every answer cites source documents
- **Deterministic Risk Engine**: Keyword + regex patterns, no LLM in risk path (avoids false positives)
- **Self-Check Layer**: Every response is re-verified against source docs before delivery
- **Domain Boosting**: Docs from the ticket's own company are weighted higher in retrieval
- **Conservative Escalation**: Vague outages, billing disputes, identity theft → always escalate

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Set your API key via environment variable — never hardcode keys:

```bash
# NVIDIA NIM (default — recommended)
export LLM_PROVIDER="nvidia"
export NVIDIA_API_KEY="nvapi-your-key-here"

# Windows PowerShell
$env:LLM_PROVIDER="nvidia"
$env:NVIDIA_API_KEY="nvapi-your-key-here"

# OpenAI
export OPENAI_API_KEY="sk-your-key"

# Anthropic Claude
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="sk-ant-your-key"

# Local Ollama (no API key needed)
export LLM_PROVIDER="ollama"
export LLM_MODEL="llama3"
```

Copy `.env.example` → `.env` and fill in your key (`.env` is gitignored):

```bash
cp ../.env.example ../.env
```

## Usage

### Process all support tickets (submission mode)
```bash
python code/main.py
```

### Run on sample tickets (development / testing)
```bash
python code/main.py --sample
```

### Interactive chat mode
```bash
python code/main.py --interactive
```

## Output

Writes to `support_tickets/output.csv` with columns:

| Column | Description |
|--------|-------------|
| `issue` | Original ticket text |
| `subject` | Ticket subject |
| `company` | HackerRank / Claude / Visa / None |
| `response` | Grounded answer or escalation message |
| `product_area` | Support category (e.g. interviews, privacy, card_services) |
| `status` | `replied` or `escalated` |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |
| `justification` | Decision reasoning with corpus doc IDs |

## Modules

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestrator — full ticket processing flow |
| `config.py` | All prompts, paths, and settings in one place |
| `classifier.py` | LLM-based request type + product area classification |
| `retriever.py` | FAISS corpus search with LLM query rewriting |
| `risk_engine.py` | Deterministic keyword + regex risk detection (no LLM) |
| `responder.py` | Corpus-grounded response + escalation generation |
| `self_check.py` | LLM response verification against source documents |
| `llm_client.py` | Unified LLM client (NVIDIA NIM / OpenAI / Anthropic / Ollama) |
| `logger.py` | Structured logging to AGENTS.md-mandated log file |

## Key Differentiators

1. **Self-Check Layer**: Responses are re-verified by a second LLM pass before delivery — catches hallucinations before they reach users
2. **Deterministic Risk Engine**: No LLM in the risk path means zero false positives from prompt injection or adversarial tickets
3. **NVIDIA NIM Embeddings**: Uses `input_type="passage"` for corpus indexing and `input_type="query"` for search — the distinction NVIDIA requires for optimal retrieval
4. **Exponential Backoff**: Handles API rate limits gracefully (8s → 16s → 32s → 64s → 90s retries)
5. **4 LLM Calls/Ticket**: Classifier + query rewriter + responder + self-check — template-based justification saves one call per ticket
