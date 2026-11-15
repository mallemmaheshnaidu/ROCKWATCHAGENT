"""
blackboard.py - Shared Mine State & Hash-Chained Audit Ledger for RockWatch.

Enforces:
- Partitioned sections with read/write versioning and writer identity tracking.
- Canonical JSON hash-chained append-only logging to logs/run_log.jsonl.
- Fail-closed sensitivity lookup with injectable clock and expiry audit.
- Inviolable TARP baseline safety floor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

# Inviolable constants
FORECASTER_BASELINE_SENSITIVITY: float = 1.0
TARP_SAFETY_FLOOR_VELOCITY_MM_HR: float = 15.0  # Level 4 Red trigger threshold


@dataclass
class DesensitizationProposal:
    """Proposal to temporarily adjust forecaster sensitivity."""
    proposal_id: str
    zone_id: str
    reason: str
    requested_by: str
    created_at: float
    expires_at: float
    approved_by: Optional[str] = None
    is_approved: bool = False
    adjusted_sensitivity: float = 0.85
    is_expired: bool = False

    def check_and_update_expiry(self, current_time: float) -> bool:
        """Returns True if expired, updating internal state."""
        if current_time >= self.expires_at and not self.is_expired:
            self.is_expired = True
            return True
        return self.is_expired

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MineBlackboard:
    """
    Central shared state object read and written by all 8 agents.
    Partitioned into named sections to prevent cross-agent collisions.
    """

    def __init__(
        self,
        log_path: str = "logs/run_log.jsonl",
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.log_path = log_path
        self.time_provider = time_provider or time.time
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

        # Partitioned state storage
        self.partitions: Dict[str, Dict[str, Any]] = {
            "raw_telemetry": {},
            "sensor_health": {},
            "clean_telemetry": {},
            "surface_changes": {},
            "tarp_status": {},
            "forecast": {},
            "advisor_notes": {},
            "arbitration": {},
            "proposals": {},       # zone_id -> DesensitizationProposal
            "golden_pool": {},     # pending missed events & benchmark sets
        }

        # Metadata tracking: section -> {"version": int, "last_writer": str, "updated_at": float}
        self.partition_meta: Dict[str, Dict[str, Any]] = {
            k: {"version": 0, "last_writer": "system", "updated_at": self.time_provider()}
            for k in self.partitions
        }

        # Serialises appends within this process (multi-process requires a lockfile).
        self._lock = threading.RLock()

        # Hash-chain state
        self.entry_index: int = 0
        self.last_hash: str = "0" * 64
        self.chain_status: Dict[str, Any] = {"valid": True, "entries": 0, "failures": []}
        self._init_log_anchor()

    # --- Audit ledger ---

    @staticmethod
    def verify_chain(log_path: str) -> Dict[str, Any]:
        """
        Recomputes every entry hash and link. Returns a report; never raises on
        corrupt content (a corrupt line is reported, not fatal).
        """
        prev_hash = "0" * 64
        expected_id = 1
        failures: List[Dict[str, Any]] = []
        count = 0

        if not os.path.exists(log_path):
            return {"valid": True, "entries": 0, "failures": [], "missing": True}

        with open(log_path, "r", encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as exc:
                    failures.append({"line": lineno, "reason": "UNPARSEABLE", "detail": str(exc)})
                    count += 1
                    continue

                body = {k: v for k, v in entry.items() if k != "hash"}
                expected_hash = hashlib.sha256(
                    json.dumps(body, sort_keys=True).encode("utf-8")
                ).hexdigest()

                if entry.get("hash") != expected_hash:
                    failures.append({"line": lineno, "reason": "HASH_MISMATCH",
                                     "entry_id": entry.get("entry_id")})
                if entry.get("prev_hash") != prev_hash:
                    failures.append({"line": lineno, "reason": "BROKEN_LINK",
                                     "entry_id": entry.get("entry_id")})
                if entry.get("entry_id") != expected_id:
                    failures.append({"line": lineno, "reason": "ENTRY_ID_GAP",
                                     "expected": expected_id, "found": entry.get("entry_id")})

                prev_hash = entry.get("hash", "")
                expected_id += 1
                count += 1

        return {"valid": not failures, "entries": count, "failures": failures}

    def _rotate_corrupt_log(self, report: Dict[str, Any]) -> None:
        """Archives a broken ledger before starting a new, verifiable segment."""
        stamp = int(self.time_provider())
        archive = f"{self.log_path}.corrupt-{stamp}"
        try:
            os.replace(self.log_path, archive)
        except OSError:
            # Fall back to truncation only if the file cannot be moved.
            open(self.log_path, "w", encoding="utf-8").close()
        self.chain_status = report

    def _init_log_anchor(self) -> None:
        """Initialises or resumes the hash chain, refusing to trust a broken ledger."""
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            report = self.verify_chain(self.log_path)
            if report["valid"]:
                with open(self.log_path, "r", encoding="utf-8") as handle:
                    last_line = ""
                    for line in handle:
                        if line.strip():
                            last_line = line
                if last_line:
                    data = json.loads(last_line)
                    self.entry_index = report["entries"]
                    self.last_hash = data["hash"]
                    self.chain_status = report
                    return
            else:
                self._rotate_corrupt_log(report)

        # Write genesis block if log is new, empty, or was rotated after corruption
        self.entry_index = 0
        self.last_hash = "0" * 64
        self._append_log_entry(
            agent_id="SYSTEM",
            event_type="GENESIS",
            payload={
                "message": "RockWatch hash-chained audit ledger initialized",
                "prior_chain_failures": self.chain_status.get("failures", [])[:5],
            },
        )

    def _canonical_hash(self, entry_dict: Dict[str, Any]) -> str:
        """DRY canonical SHA-256 hash using sorted keys."""
        canonical_str = json.dumps(entry_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def _append_log_entry(self, agent_id: str, event_type: str, payload: Dict[str, Any]) -> str:
        """Appends a new canonically hashed entry linked to prev_hash (atomic append)."""
        with self._lock:
            self.entry_index += 1
            entry = {
                "entry_id": self.entry_index,
                "prev_hash": self.last_hash,
                "timestamp": self.time_provider(),
                "agent": agent_id,
                "event_type": event_type,
                "payload": payload,
            }
            entry["hash"] = self._canonical_hash(entry)
            self.last_hash = entry["hash"]

            line = (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8")
            fd = os.open(self.log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)

            return entry["hash"]

    # --- Read & Write Contracts ---

    def read(self, section: str) -> Dict[str, Any]:
        """Reads a partition safely (deep copy: callers can never mutate live state)."""
        return copy.deepcopy(self.partitions.get(section, {}))

    def write(self, section: str, data: Dict[str, Any], writer_id: str) -> None:
        """
        Writes data into a partition, updates versioning metadata,
        and logs the transaction to the append-only ledger.
        """
        with self._lock:
            if section not in self.partitions:
                self.partitions[section] = {}

            self.partitions[section].update(data)

            # Update partition metadata
            meta = self.partition_meta.setdefault(
                section, {"version": 0, "last_writer": "", "updated_at": 0.0}
            )
            meta["version"] += 1
            meta["last_writer"] = writer_id
            meta["updated_at"] = self.time_provider()

            # Audit write
            self._append_log_entry(
                agent_id=writer_id,
                event_type=f"WRITE_{section.upper()}",
                payload={"section": section, "keys_updated": list(data.keys()), "version": meta["version"]},
            )

    # --- Fail-Closed Sensitivity Lookup ---

    def get_effective_forecaster_sensitivity(
        self, zone_id: str, current_time: Optional[float] = None
    ) -> float:
        """
        Returns active sensitivity for zone_id.
        Fails closed to FORECASTER_BASELINE_SENSITIVITY upon expiry, absence, or exception.
        Never affects or modifies TARP_SAFETY_FLOOR_VELOCITY_MM_HR.
        """
        now = current_time if current_time is not None else self.time_provider()
        try:
            proposals = self.partitions.get("proposals", {})
            proposal_dict = proposals.get(zone_id)
            if not proposal_dict:
                return FORECASTER_BASELINE_SENSITIVITY

            # Parse as model if dict
            if isinstance(proposal_dict, dict):
                proposal = DesensitizationProposal(**proposal_dict)
            else:
                proposal = proposal_dict

            if proposal.is_approved:
                if now < proposal.expires_at:
                    return proposal.adjusted_sensitivity
                # Proposal has expired: revert and log
                if not proposal.is_expired:
                    proposal.is_expired = True
                    self.partitions["proposals"][zone_id] = proposal.to_dict()
                    self._append_log_entry(
                        agent_id="SYSTEM",
                        event_type="PROPOSAL_EXPIRED",
                        payload={
                            "zone_id": zone_id,
                            "proposal_id": proposal.proposal_id,
                            "reverted_to": FORECASTER_BASELINE_SENSITIVITY,
                        },
                    )
        except Exception as exc:
            # Fail closed to baseline on any parsing or lookup failure
            self._append_log_entry(
                agent_id="SYSTEM",
                event_type="LOOKUP_ERROR_FAIL_CLOSED",
                payload={"zone_id": zone_id, "error": str(exc)},
            )
        return FORECASTER_BASELINE_SENSITIVITY

    def get_head_hash(self) -> str:
        """Returns the current head hash of the ledger."""
        return self.last_hash
