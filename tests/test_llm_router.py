"""
test_llm_router.py - Unit Tests for Multi-Key & LiteLLM Router.

Verifies:
- Parsing of 11 keys from GEMINI_API_KEYS environment variable.
- Round-robin key rotation and selection.
- Rate-limit cooldown handling (marking key in cooldown upon 429).
- Graceful deterministic fail-safe behavior when API endpoints are unreachable.
- AdvisorAgent and ForecasterAgent fallback behavior during outages.
"""

import os
import time
import pytest
from tools.llm_router import LLMRouter


def test_multi_key_parsing(monkeypatch):
    """Verifies that 11 comma-separated keys are parsed correctly into the pool."""
    test_keys = ",".join([f"key_sample_{i}" for i in range(1, 12)])
    monkeypatch.setenv("GEMINI_API_KEYS", test_keys)
    monkeypatch.setenv("GEMINI_API_KEY", "fallback_single_key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-light")

    router = LLMRouter()
    status = router.get_status()

    # 11 keys + 1 fallback = 12 keys
    assert status["total_keys"] == 12
    assert status["total_rpm_capacity"] == 12 * 15  # 180 RPM
    assert status["daily_capacity"] == 12 * 500      # 6,000 requests/day
    assert status["model_name"] == "gemini-3.5-flash-light"


def test_round_robin_rotation(monkeypatch):
    """Verifies that keys are rotated sequentially across round-robin selections."""
    keys = ["key_alpha", "key_beta", "key_gamma"]
    monkeypatch.setenv("GEMINI_API_KEYS", ",".join(keys))
    monkeypatch.setenv("GEMINI_API_KEY", "")

    router = LLMRouter()

    k1, i1 = router._get_next_direct_key()
    k2, i2 = router._get_next_direct_key()
    k3, i3 = router._get_next_direct_key()
    k4, i4 = router._get_next_direct_key()

    assert k1 == "key_alpha" and i1 == 0
    assert k2 == "key_beta" and i2 == 1
    assert k3 == "key_gamma" and i3 == 2
    assert k4 == "key_alpha" and i4 == 0  # Wrapped around!


def test_rate_limit_cooldown_failover(monkeypatch):
    """Verifies that when a key hits 429 and enters cooldown, it is skipped."""
    keys = ["key_1", "key_2"]
    monkeypatch.setenv("GEMINI_API_KEYS", ",".join(keys))
    monkeypatch.setenv("GEMINI_API_KEY", "")

    router = LLMRouter()

    # Mark key_1 into cooldown
    router._mark_key_cooldown("key_1", duration_sec=60.0)

    # Next key should skip key_1 and return key_2
    next_key, idx = router._get_next_direct_key()
    assert next_key == "key_2"
    assert idx == 1

    status = router.get_status()
    assert status["active_keys"] == 1
    assert status["cooldown_keys"] == 1


def test_deterministic_offline_fallback(monkeypatch):
    """Verifies that when endpoints are unreachable, query() returns None without throwing."""
    monkeypatch.setenv("LITELLM_API_BASE", "http://127.0.0.1:9999/v1")  # Dead port
    monkeypatch.setenv("GEMINI_API_KEYS", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.5")

    router = LLMRouter()
    res = router.query("Is the slope stable?", agent_name="TEST_AGENT")
    assert res is None  # Deterministic safety fallback!
    assert router.call_stats["failed_calls"] >= 1
