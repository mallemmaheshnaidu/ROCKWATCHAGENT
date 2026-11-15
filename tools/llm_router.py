"""
tools/llm_router.py - Enterprise LLM Router for RockWatch.

Dual-Engine Architecture:
1. Primary: LiteLLM Local Proxy (OpenAI-compatible /v1/chat/completions)
   - Connects to LiteLLM instance running on user's machine (e.g. http://localhost:4000/v1).
   - Pools every key configured in litellm_config.yaml (the example ships with 11).
   - LiteLLM natively handles 15 RPM limits, load balancing, key cooldowns, and retries.
2. Built-in Multi-Key Round-Robin Fallback:
   - If LiteLLM is not running or unreachable, directly rotates through every key in
     GEMINI_API_KEYS plus the optional GEMINI_API_KEY fallback via the Google Gemini REST API.
   - Built-in HTTP 429 detection: marks key in 60s cooldown and immediately fails over to the next key.
3. Fail-Closed Deterministic Safety:
   - If all APIs/network are unavailable, returns None without raising exceptions.
   - Agents automatically fall back to deterministic safety logic (passes Scenario 6 tests).
"""

from __future__ import annotations
import os
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("RockWatch.LLMRouter")


class LLMRouter:
    """
    Intelligent LLM router supporting LiteLLM proxy and direct multi-key Gemini pool.
    """

    def __init__(self):
        self._load_config()
        self._key_index = 0
        self._key_cooldowns: Dict[str, float] = {}  # key -> cooldown_until_timestamp
        self._litellm_cooldown_until: float = 0.0  # Fast-fail cooldown if LiteLLM is not running
        self.call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "litellm_calls": 0,
            "direct_pool_calls": 0,
            "last_latency_ms": 0.0,
            "active_endpoint": "INITIALIZING",
        }
        self.call_history: List[Dict[str, Any]] = []

    @property
    def api_model_name(self) -> str:
        """Normalizes model identifier for Google Generative Language REST API."""
        m = (self.model_name or "").lower().strip()
        aliases = {
            "gemini-3.5-flash-light": "gemini-3.5-flash-lite",
            "gemini-3.5-flashlight": "gemini-3.5-flash-lite",
            "gemini-3.5-flash": "gemini-3.5-flash-lite",
        }
        return aliases.get(m, self.model_name)

    def _load_config(self) -> None:
        """Loads configuration from environment variables."""
        self.litellm_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000/v1").rstrip("/")
        self.litellm_key = os.getenv("LITELLM_API_KEY", "sk-litellm-local")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "6.0"))

        # Parse keys for direct multi-key pool
        raw_keys = os.getenv("GEMINI_API_KEYS", "")
        single_key = os.getenv("GEMINI_API_KEY", "")

        keys: List[str] = []
        if raw_keys:
            # Comma, newline, or semicolon delimited
            for k in raw_keys.replace(";", ",").replace("\n", ",").split(","):
                k_clean = k.strip().strip('"').strip("'")
                if k_clean and k_clean not in keys:
                    keys.append(k_clean)

        if single_key:
            k_clean = single_key.strip().strip('"').strip("'")
            if k_clean and k_clean not in keys:
                keys.append(k_clean)

        self.keys = keys

    def reload(self) -> None:
        """Refreshes configuration from environment (useful when .env changes)."""
        self._load_config()

    def get_status(self) -> Dict[str, Any]:
        """Returns runtime status for UI telemetry display."""
        now = time.time()
        active_keys = [k for k in self.keys if self._key_cooldowns.get(k, 0.0) <= now]
        cooldown_keys = [k for k in self.keys if self._key_cooldowns.get(k, 0.0) > now]

        key_count = len(self.keys)
        # Documented free-tier planning assumption, not a guaranteed quota.
        rpm_per_key = int(os.getenv("GEMINI_RPM_PER_KEY", "15"))
        daily_per_key = int(os.getenv("GEMINI_DAILY_PER_KEY", "500"))
        total_rpm = key_count * rpm_per_key
        daily_capacity = key_count * daily_per_key

        # Mode reflects the most recent attempt, not an all-time success counter.
        endpoint = self.call_stats["active_endpoint"]
        if endpoint.startswith("LiteLLM"):
            mode = "LITELLM_PROXY"
        elif endpoint.startswith("Gemini Direct Pool"):
            mode = "DIRECT_POOL"
        elif endpoint == "OFFLINE_DETERMINISTIC_FALLBACK":
            mode = "OFFLINE"
        elif key_count > 0:
            mode = "READY_DIRECT_POOL"
        else:
            mode = "OFFLINE"

        return {
            "mode": mode,
            "litellm_base": self.litellm_base,
            "model_name": self.model_name,
            "total_keys": key_count,
            "active_keys": len(active_keys),
            "cooldown_keys": len(cooldown_keys),
            "total_rpm_capacity": total_rpm,
            "daily_capacity": daily_capacity,
            "total_calls": self.call_stats["total_calls"],
            "successful_calls": self.call_stats["successful_calls"],
            "failed_calls": self.call_stats["failed_calls"],
            "last_latency_ms": self.call_stats["last_latency_ms"],
            "active_endpoint": self.call_stats["active_endpoint"],
        }

    def _get_next_direct_key(self) -> Optional[Tuple[str, int]]:
        """Round-robin selection of an active (non-cooled) key from the configured pool."""
        if not self.keys:
            return None

        now = time.time()
        n = len(self.keys)
        for i in range(n):
            idx = (self._key_index + i) % n
            key = self.keys[idx]
            if self._key_cooldowns.get(key, 0.0) <= now:
                self._key_index = (idx + 1) % n
                return key, idx

        # Every key is cooling down: fail over to the deterministic path rather
        # than hammering a key that is known to be rate-limited.
        return None

    def _mark_key_cooldown(self, key: str, duration_sec: float = 60.0) -> None:
        """Places a key into cooldown following a 429 rate limit."""
        self._key_cooldowns[key] = time.time() + duration_sec
        logger.warning(f"Gemini key ending ...{key[-4:]} entered {duration_sec}s cooldown due to 429.")

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_name: str = "COPILOT",
        temperature: float = 0.25,
        max_tokens: int = 600,
    ) -> Optional[str]:
        """
        Routes the prompt to LiteLLM first, falling back to the direct Gemini key pool,
        and finally returning None on complete outage (enabling deterministic failover).
        """
        import requests

        self.call_stats["total_calls"] += 1
        t0 = time.time()

        # ----------------------------------------------------------------------
        # Strategy 1: Attempt LiteLLM Proxy (OpenAI-compatible)
        # ----------------------------------------------------------------------
        now = time.time()
        if now >= self._litellm_cooldown_until:
            litellm_url = f"{self.litellm_base}/chat/completions"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            litellm_payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            litellm_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.litellm_key}",
            }

            try:
                # Fast connect timeout (0.8s) avoids freezing UI when LiteLLM is not running
                resp = requests.post(
                    litellm_url,
                    headers=litellm_headers,
                    json=litellm_payload,
                    timeout=(0.8, self.timeout_seconds),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    dt_ms = round((time.time() - t0) * 1000, 1)

                    self.call_stats["successful_calls"] += 1
                    self.call_stats["litellm_calls"] += 1
                    self.call_stats["last_latency_ms"] = dt_ms
                    self.call_stats["active_endpoint"] = f"LiteLLM ({self.model_name})"

                    self._record_log(agent_name, prompt, content, dt_ms, "SUCCESS_LITELLM")
                    return content
                else:
                    self._litellm_cooldown_until = now + 15.0
            except Exception:
                # LiteLLM connection timed out or refused -> cool down for 30s to keep UI instant
                self._litellm_cooldown_until = now + 30.0

        # ----------------------------------------------------------------------
        # Strategy 2: Direct Multi-Key Gemini REST Pool
        # ----------------------------------------------------------------------
        if self.keys:
            max_attempts = len(self.keys)
            for attempt in range(max_attempts):
                key_info = self._get_next_direct_key()
                if not key_info:
                    break
                key, key_idx = key_info

                gemini_url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self.api_model_name}:generateContent"
                )
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

                gemini_body = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    },
                }

                try:
                    resp = requests.post(
                        gemini_url,
                        headers={
                            "Content-Type": "application/json",
                            # Key travels in a header, never in the URL (avoids proxy/HAR leaks).
                            "x-goog-api-key": key,
                        },
                        json=gemini_body,
                        timeout=(2.5, self.timeout_seconds),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        dt_ms = round((time.time() - t0) * 1000, 1)

                        self.call_stats["successful_calls"] += 1
                        self.call_stats["direct_pool_calls"] += 1
                        self.call_stats["last_latency_ms"] = dt_ms
                        self.call_stats["active_endpoint"] = f"Gemini Direct Pool (Key #{key_idx + 1}/{len(self.keys)})"

                        self._record_log(agent_name, prompt, content, dt_ms, f"SUCCESS_KEY_{key_idx + 1}")
                        return content
                    elif resp.status_code == 429:
                        # Rate limit hit: put key in cooldown and retry next key
                        self._mark_key_cooldown(key, duration_sec=60.0)
                        continue
                    elif resp.status_code in (400, 404):
                        # Request/model error: retrying other keys cannot help.
                        self._mark_key_cooldown(key, duration_sec=300.0)
                        logger.error(
                            f"Gemini rejected the request (HTTP {resp.status_code}); "
                            f"check model id {self.api_model_name!r}: {resp.text[:160]}"
                        )
                        break
                    elif resp.status_code in (401, 403):
                        # Invalid or unauthorized key: long cooldown, try the next one.
                        self._mark_key_cooldown(key, duration_sec=3600.0)
                        continue
                    else:
                        # Transient server error: short jittered cooldown.
                        self._mark_key_cooldown(key, duration_sec=5.0 + (attempt * 2.0))
                        logger.warning(
                            f"Gemini API returned HTTP {resp.status_code}: {resp.text[:120]}"
                        )
                        continue
                except Exception as exc:
                    logger.warning(f"Gemini direct call failed: {exc}")
                    continue

        # ----------------------------------------------------------------------
        # Strategy 3: Graceful Offline Fallback (Deterministic safety path)
        # ----------------------------------------------------------------------
        dt_ms = round((time.time() - t0) * 1000, 1)
        self.call_stats["failed_calls"] += 1
        self.call_stats["last_latency_ms"] = dt_ms
        self.call_stats["active_endpoint"] = "OFFLINE_DETERMINISTIC_FALLBACK"
        self._record_log(agent_name, prompt, None, dt_ms, "OUTAGE_FALLBACK")
        return None

    def _record_log(
        self,
        agent_name: str,
        prompt: str,
        content: Optional[str],
        latency_ms: float,
        status: str,
    ) -> None:
        """Maintains ring buffer of recent LLM interactions."""
        entry = {
            "timestamp": time.time(),
            "agent": agent_name,
            "prompt_preview": (prompt[:75] + "...") if len(prompt) > 75 else prompt,
            "response_preview": ((content[:75] + "...") if content else "[DETERMINISTIC_FALLBACK]"),
            "latency_ms": latency_ms,
            "status": status,
        }
        self.call_history.append(entry)
        if len(self.call_history) > 50:
            self.call_history.pop(0)

    # ==========================================================================
    # DOMAIN AGENT HELPER METHODS
    # ==========================================================================

    def generate_advisor_explanation(
        self,
        zone_id: str,
        tarp_level: str,
        citation: str,
        sop_text: str,
        incidents_count: int,
        telemetry: Dict[str, Any],
    ) -> Optional[str]:
        """
        Agent 5 (Advisor) synthesis: plain-language operational guidance grounded in RAG.
        """
        system = (
            "You are Agent 5 (Geotechnical Advisor) in the RockWatch multi-agent system. "
            "Your task is to synthesize clear, actionable, calm geotechnical advice for open-pit mine operators. "
            "You MUST explicitly cite the provided SOP passage ID and explain mandatory actions clearly."
        )
        prompt = (
            f"Zone: {zone_id}\n"
            f"Active TARP Level: {tarp_level}\n"
            f"Governing SOP Citation: {citation}\n"
            f"Mandatory SOP Text: {sop_text}\n"
            f"Precedent Case Count: {incidents_count}\n"
            f"Telemetry Summary: Velocity={telemetry.get('velocity_mm_hr', 0.0)} mm/h, "
            f"Rainfall={telemetry.get('rainfall_3h_mm', 0.0)} mm, "
            f"Vibration PPV={telemetry.get('vibration_ppv_mms', 0.0)} mm/s.\n\n"
            "Provide a concise (3-4 sentences) operational synthesis directing pit operations."
        )
        return self.query(prompt, system_prompt=system, agent_name="AGENT_ADVISOR", temperature=0.2, max_tokens=250)

    def generate_forecaster_narrative(
        self,
        zone_id: str,
        p_rockfall: float,
        risk_tier: str,
        uncertainty: float,
        telemetry: Dict[str, Any],
    ) -> Optional[str]:
        """
        Agent 4 (Forecaster) synthesis: explains cross-sensor correlation narrative.
        """
        system = (
            "You are Agent 4 (Calibrated Risk Forecaster) in the RockWatch multi-agent system. "
            "Explain the physical geotechnical dynamics driving the 6-hour risk horizon."
        )
        prompt = (
            f"Zone: {zone_id}\n"
            f"6-Hour Rockfall Probability: {p_rockfall:.1%}\n"
            f"Risk Tier: {risk_tier} (Uncertainty ±{uncertainty:.1%})\n"
            f"Sensor Inputs: Velocity={telemetry.get('effective_velocity_mm_hr', 0.0)} mm/h, "
            f"Rainfall={telemetry.get('rainfall_3h_mm', 0.0)} mm, "
            f"Vibration PPV={telemetry.get('vibration_ppv_mms', 0.0)} mm/s.\n\n"
            "Summarize the physical mechanism (e.g. rainwater pore pressure infiltration, blast vibration triggering creep) in 2 sentences."
        )
        return self.query(prompt, system_prompt=system, agent_name="AGENT_FORECASTER", temperature=0.25, max_tokens=180)

    def generate_arbiter_explanation(
        self,
        zone_id: str,
        tarp_level: str,
        forecast_tier: str,
        effective_level: str,
        discrepancy: bool,
    ) -> Optional[str]:
        """
        Agent 6 (Arbiter) synthesis: explains asymmetric caution and discrepancies.
        """
        system = (
            "You are Agent 6 (Decision Arbiter) in the RockWatch multi-agent safety system. "
            "Briefly justify the arbitrated safety level under the Asymmetric Caution Principle."
        )
        prompt = (
            f"Zone: {zone_id}\n"
            f"Deterministic TARP Rule: {tarp_level}\n"
            f"ML Forecaster Tier: {forecast_tier}\n"
            f"Arbitrated Effective Level: {effective_level}\n"
            f"Divergence Detected: {discrepancy}\n\n"
            "State why this safety level was selected in 1 clear sentence."
        )
        return self.query(prompt, system_prompt=system, agent_name="AGENT_ARBITER", temperature=0.2, max_tokens=120)

    def generate_governance_root_cause(
        self,
        zone_id: str,
        detached_volume_m3: float,
        telemetry: Dict[str, Any],
        sensor_health: Dict[str, Any],
    ) -> Optional[str]:
        """
        Agent 7 (Governance) synthesis: diagnoses un-alarmed detachment for geotech sign-off.
        """
        system = (
            "You are Agent 7 (Governance & Root Cause Analysis) in the RockWatch multi-agent system. "
            "Propose a plausible physical cause for an un-alarmed rockfall event for review by the Principal Geotechnical Engineer."
        )
        prompt = (
            f"Zone: {zone_id}\n"
            f"Detached Volume (LiDAR): {detached_volume_m3:.1f} m3\n"
            f"Preceding Velocity: {telemetry.get('effective_velocity_mm_hr', 0.0)} mm/h\n"
            f"Rainfall: {telemetry.get('rainfall_3h_mm', 0.0)} mm\n"
            f"Radar Health Status: {'ONLINE' if sensor_health.get('radar_online', True) else 'DEGRADED/BLIND'}\n\n"
            "Propose a diagnostic explanation (e.g. geological radar shadow, brittle planar joint failure) in 2 sentences."
        )
        return self.query(prompt, system_prompt=system, agent_name="AGENT_GOVERNANCE", temperature=0.2, max_tokens=180)


# Global singleton instance
_llm_router_instance: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """Returns the shared LLMRouter instance."""
    global _llm_router_instance
    if _llm_router_instance is None:
        _llm_router_instance = LLMRouter()
    return _llm_router_instance
