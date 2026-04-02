"""HMAC integrity verification for DAG nodes."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any


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
