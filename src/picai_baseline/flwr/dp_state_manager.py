from __future__ import annotations

from pathlib import Path
import torch
from opacus.accountants import RDPAccountant

class DPStateManager:
    def __init__(self, storage_dir: Path = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".flwr_dp_states"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, participant_id: int) -> Path:
        return self.storage_dir / f"client_{participant_id}.pt"

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
