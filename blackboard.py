"""
blackboard.py - Shared Mine State & Hash-Chained Audit Ledger for RockWatch.

Enforces:
- Partitioned sections with read/write versioning and writer identity tracking.
- Canonical JSON hash-chained append-only logging to logs/run_log.jsonl.
- Fail-closed sensitivity lookup with injectable clock and expiry audit.
- Inviolable TARP baseline safety floor.
"""

from __future__ import annotations
import json
import hashlib
import os
import time
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

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

        # Hash-chain state
        self.entry_index: int = 0
        self.last_hash: str = "0" * 64
        self._init_log_anchor()

    def _init_log_anchor(self) -> None:
        """Initializes or resumes the hash chain from the log file."""
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            last_line = ""
            count = 0
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                        count += 1
            if last_line:
                try:
                    data = json.loads(last_line)
                    self.entry_index = count
                    self.last_hash = data.get("hash", "0" * 64)
                    return
                except Exception:
                    pass

        # Write genesis block if log is new or empty
        self.entry_index = 0
        self.last_hash = "0" * 64
        self._append_log_entry(
            agent_id="SYSTEM",
            event_type="GENESIS",
            payload={"message": "RockWatch hash-chained audit ledger initialized"},
        )

    def _canonical_hash(self, entry_dict: Dict[str, Any]) -> str:
        """DRY canonical SHA-256 hash using sorted keys."""
        canonical_str = json.dumps(entry_dict, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def _append_log_entry(self, agent_id: str, event_type: str, payload: Dict[str, Any]) -> str:
        """Appends a new canonically hashed entry linked to prev_hash."""
        self.entry_index += 1
        now = self.time_provider()

        entry = {
            "entry_id": self.entry_index,
            "prev_hash": self.last_hash,
            "timestamp": now,
            "agent": agent_id,
            "event_type": event_type,
            "payload": payload,
        }

        entry_hash = self._canonical_hash(entry)
        entry["hash"] = entry_hash
        self.last_hash = entry_hash

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

        return entry_hash

    # --- Read & Write Contracts ---

    def read(self, section: str) -> Dict[str, Any]:
        """Reads a partition safely."""
        return self.partitions.get(section, {}).copy()

    def write(self, section: str, data: Dict[str, Any], writer_id: str) -> None:
        """
        Writes data into a partition, updates versioning metadata,
        and logs the transaction to the append-only ledger.
        """
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
                else:
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
        except Exception as e:
            # Fail closed to baseline on any parsing or lookup failure
            self._append_log_entry(
                agent_id="SYSTEM",
                event_type="LOOKUP_ERROR_FAIL_CLOSED",
                payload={"zone_id": zone_id, "error": str(e)},
            )
        return FORECASTER_BASELINE_SENSITIVITY

    def get_head_hash(self) -> str:
        """Returns the current head hash of the ledger."""
        return self.last_hash
