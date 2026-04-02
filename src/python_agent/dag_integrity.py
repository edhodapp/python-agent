"""HMAC integrity verification and injection scanning."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from typing import Any

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"you\s+are\s+now\s+a",
        r"new\s+instructions:",
        r"system\s+prompt:",
        r"</ontology-data>",
        r"</strategy-data>",
        r"</candidate-summaries>",
        r"</context-data>",
        r"</user-input>",
    ]
]


def generate_key() -> str:
    """Generate 32-byte random key, hex-encoded."""
    return os.urandom(32).hex()


def load_or_create_key(path: str) -> str:
    """Load hex key from file, or create file with new key."""
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        key = generate_key()
        with open(path, "w") as f:
            f.write(key)
        return key


def compute_hash(
    ontology_dict: dict[str, Any], key: str,
) -> str:
    """HMAC-SHA256 hex digest of deterministic JSON."""
    payload = json.dumps(ontology_dict, sort_keys=True)
    return hmac.new(
        bytes.fromhex(key),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()


def sign_node(node: Any, key: str) -> None:
    """Set node.integrity_hash from ontology content."""
    node.integrity_hash = compute_hash(
        node.ontology.model_dump(), key,
    )


def verify_node(node: Any, key: str) -> bool:
    """Return True if hash matches. False if tampered.

    Returns False for unsigned nodes (empty hash).
    """
    if not node.integrity_hash:
        return False
    expected = compute_hash(
        node.ontology.model_dump(), key,
    )
    return hmac.compare_digest(
        node.integrity_hash, expected,
    )


def verify_dag(dag: Any, key: str) -> list[str]:
    """Return IDs of signed nodes that fail verification.

    Unsigned nodes (empty hash) are skipped.
    """
    failed: list[str] = []
    for n in dag.nodes:
        if not n.integrity_hash:
            continue
        if not verify_node(n, key):
            failed.append(n.id)
    return failed


def scan_text_for_injection(text: str) -> list[str]:
    """Scan a string for common injection patterns.

    Returns list of matched pattern descriptions.
    """
    matches: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def _collect_text_fields(
    ontology_dict: dict[str, Any],
) -> list[str]:
    """Extract all free-text fields from an ontology dict."""
    texts: list[str] = []
    for entity in ontology_dict.get("entities", []):
        texts.append(entity.get("description", ""))
    for c in ontology_dict.get("domain_constraints", []):
        texts.append(c.get("description", ""))
        texts.append(c.get("expression", ""))
    for q in ontology_dict.get("open_questions", []):
        texts.append(q.get("text", ""))
        texts.append(q.get("context", ""))
        texts.append(q.get("resolution", ""))
    return [t for t in texts if t]


def scan_ontology_for_injection(
    ontology_dict: dict[str, Any],
) -> list[str]:
    """Scan all text fields in an ontology for injection.

    Returns list of warnings (empty if clean).
    """
    warnings_list: list[str] = []
    for text in _collect_text_fields(ontology_dict):
        hits = scan_text_for_injection(text)
        for pattern in hits:
            preview = text[:80]
            warnings_list.append(
                f"Suspicious pattern {pattern!r} "
                f"in: {preview!r}",
            )
    return warnings_list
