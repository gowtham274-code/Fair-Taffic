"""One federated communication round: select -> collect -> aggregate -> distribute -> validate.

Local training happens *before* a round (the trainer runs episodes in which every client
learns from its own experience). ``FederatedCoordinator.communication_round`` then
performs the exchange.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from common.config import FederatedConfig
from common.logging_setup import get_logger
from federated.aggregation import ClientUpdate, Weights, weights_nbytes
from federated.client import FederatedClient
from federated.server import FederatedServer, RoundRecord

log = get_logger(__name__)
Validator = Callable[[Weights], Dict[str, float]]


class FederatedCoordinator:
    def __init__(self, clients: Sequence[FederatedClient], cfg: FederatedConfig, seed: int = 0,
                 validator: Optional[Validator] = None):
        if not clients:
            raise ValueError("at least one client is required")
        self.clients: Dict[str, FederatedClient] = {c.client_id: c for c in clients}
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.validator = validator
        # shared initial model: client 0's initial weights become the global model v0
        first = clients[0]
        self.server = FederatedServer(first.agent.get_weights(), cfg.weight_by)
        for c in clients:
            self.server.register_client(c.client_id)
            c.receive_global(self.server.global_weights, 0)

    def select_participants(self) -> List[str]:
        ids = sorted(self.clients)
        k = max(1, int(round(self.cfg.participation_rate * len(ids))))
        if k >= len(ids):
            return ids
        return sorted(self.rng.choice(ids, size=k, replace=False).tolist())

    def communication_round(self, round_idx: int) -> RoundRecord:
        participants = self.select_participants()
        updates: List[ClientUpdate] = [self.clients[i].produce_update() for i in participants]
        rec = self.server.aggregate(round_idx, updates)
        if rec.status == "ok":
            targets = participants if self.cfg.distribute_to == "participants" else sorted(self.clients)
            for cid in targets:
                self.clients[cid].receive_global(self.server.global_weights, self.server.version)
            rec.bytes_down = len(targets) * weights_nbytes(self.server.global_weights)
        if (self.validator is not None and self.cfg.validate_every_rounds > 0
                and round_idx % self.cfg.validate_every_rounds == 0 and rec.status == "ok"):
            rec.validation = self.validator(self.server.global_weights)
        log.info("federated round %d: %s participants=%s v%d", round_idx, rec.status, rec.participants, rec.global_version)
        return rec
