"""
LLM Client — Unified interface for OpenAI / Anthropic / Ollama.
Deterministic outputs via temperature=0.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_TEMPERATURE,
    EMBEDDING_MODEL, SEED,
    NVIDIA_API_KEY, NVIDIA_MODEL, NVIDIA_BASE_URL, NVIDIA_EMBED_MODEL
)


def call_llm(prompt: str, temperature: float = None, max_tokens: int = 1024, _retries: int = 6) -> str:
    """Call LLM with exponential backoff retry on rate-limit and connection errors."""
    import time
    if temperature is None:
        temperature = LLM_TEMPERATURE
    provider = LLM_PROVIDER.lower()
    for attempt in range(_retries):
        try:
            if provider == "openai":
                return _call_openai(prompt, temperature, max_tokens)
            elif provider == "anthropic":
                return _call_anthropic(prompt, temperature, max_tokens)
            elif provider == "nvidia":
                return _call_nvidia(prompt, temperature, max_tokens)
            elif provider == "ollama":
                return _call_ollama(prompt, temperature, max_tokens)
            raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            msg = str(e).lower()
            transient = ("429" in msg or "too many requests" in msg or "rate" in msg
                         or "connection" in msg or "timeout" in msg or "503" in msg
                         or "502" in msg or "500" in msg)
            if transient:
                wait = min(8 * (2 ** attempt), 90)  # 8s, 16s, 32s, 64s, 90s, 90s
                print(f"   [retry {attempt+1}/{_retries}] waiting {wait}s ({str(e)[:60]})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"LLM call failed after {_retries} retries")


def call_llm_json(prompt: str, temperature: float = None, max_tokens: int = 1024) -> dict:
    """Call LLM and parse response as JSON."""
    raw = call_llm(prompt, temperature, max_tokens)
    return extract_json(raw)


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown code block
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    # Find JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end])
        except json.JSONDecodeError:
            pass
    return {"error": "Failed to parse JSON", "raw": text[:500]}


def _call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise support triage system. Always respond in the exact format requested. No extra commentary."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=SEED,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system="You are a precise support triage system. Always respond in the exact format requested. No extra commentary.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_ollama(prompt: str, temperature: float, max_tokens: int) -> str:
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature, "num_predict": max_tokens, "seed": SEED}},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _call_nvidia(prompt: str, temperature: float, max_tokens: int) -> str:
    """Call NVIDIA NIM API (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise support triage system. Always respond in the exact format requested. No extra commentary."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def get_embeddings(texts: list, input_type: str = "query") -> list:
    """Get embeddings for texts. input_type: 'query' for search queries, 'passage' for corpus docs."""
    provider = LLM_PROVIDER.lower()
    if provider == "nvidia" and NVIDIA_API_KEY:
        return _get_nvidia_embeddings(texts, input_type=input_type)
    if provider in ("openai",) or OPENAI_API_KEY:
        return _get_openai_embeddings(texts)
    return _get_tfidf_embeddings(texts)


def _get_nvidia_embeddings(texts: list, input_type: str = "query") -> list:
    """Get embeddings via NVIDIA NIM API. input_type='passage' for corpus, 'query' for search."""
    from openai import OpenAI
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [t[:2048] if len(t) > 2048 else t for t in batch]
        response = client.embeddings.create(
            model=NVIDIA_EMBED_MODEL,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"}
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def _get_openai_embeddings(texts: list) -> list:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate very long texts to avoid token limits
        batch = [t[:8000] if len(t) > 8000 else t for t in batch]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def _get_tfidf_embeddings(texts: list) -> list:
    """Fallback TF-IDF pseudo-embeddings."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1536)
    tfidf_matrix = vectorizer.fit_transform(texts)
    result = []
    for i in range(tfidf_matrix.shape[0]):
        vec = tfidf_matrix[i].toarray().flatten().tolist()
        if len(vec) < 1536:
            vec.extend([0.0] * (1536 - len(vec)))
        result.append(vec[:1536])
    return result
