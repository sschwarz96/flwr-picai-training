from __future__ import annotations

from pathlib import Path
import torch
import json
from opacus.accountants import RDPAccountant


class DPStateManager:
    def __init__(self, storage_dir: Path = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".flwr_dp_states"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, participant_id: int) -> Path:
        return self.storage_dir / f"client_{participant_id}.pt"

    def _gradnorm_log_path(self, participant_id: int) -> Path:
        return self.storage_dir / f"client_{participant_id}_gradnorms.jsonl"

    def save(self, participant_id: int, accountant: RDPAccountant) -> None:
        torch.save({
            "history": accountant.history,
        }, self._state_path(participant_id))

    def load(self, participant_id: int) -> RDPAccountant | None:
        path = self._state_path(participant_id)
        if not path.exists():
            return None
        state = torch.load(path)
        accountant = RDPAccountant()
        print(f"Loaded state from {path}")
        accountant.history = state["history"]
        return accountant

    def exists(self, participant_id: int) -> bool:
        return self._state_path(participant_id).exists()

    # ----- Adaptive clipping section -----
    def log_grad_norms(
            self,
            participant_id: int,
            round_idx: int,
            median_norm: float,
            p90_norm: float,
            mean_norm: float = None,
            min_norm: float = None,
            max_norm: float = None,
    ) -> None:
        """Append gradient norm stats for this round to a log file."""
        log_path = self._gradnorm_log_path(participant_id)
        entry = {
            "round": round_idx,
            "median": median_norm,
            "p90": p90_norm,
        }
        if mean_norm is not None:
            entry["mean"] = mean_norm
        if min_norm is not None:
            entry["min"] = min_norm
        if max_norm is not None:
            entry["max"] = max_norm

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_latest_grad_norms(self, participant_id: int) -> dict | None:
        """Load the most recent grad norm stats for this client."""
        log_path = self._gradnorm_log_path(participant_id)
        if not log_path.exists():
            return None
        with open(log_path, "r") as f:
            lines = f.readlines()
        if not lines:
            return None
        return json.loads(lines[-1])

    def get_adaptive_clip_value(self, participant_id: int, quantile: str = "median",
                                factor: float = 1.0) -> float | None:
        """Return the adaptive clip norm based on latest recorded stats.
        Args:
            quantile: "median", "p90", "mean", etc.
            factor: Multiply this value (e.g., for slack).
        """
        stats = self.get_latest_grad_norms(participant_id)
        if stats is None or quantile not in stats:
            return None
        if stats[quantile] <= 0.3:
            return float(stats["p90"]) * factor
        return float(stats[quantile]) * factor

    def reset_grad_norm_log(self, participant_id: int):
        """(Optional) Remove previous log if starting fresh."""
        log_path = self._gradnorm_log_path(participant_id)
        if log_path.exists():
            log_path.unlink()
