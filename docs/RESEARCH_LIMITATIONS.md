# Research limitations
* **Simulation evidence only**; demand rates, network (2×2 single-lane grid), driver model are assumptions, not calibrated to real data.
* **SUMO integration is untested so far** in the development environment; mock-simulator output is a pipeline check, never a finding.
* Hyper-parameters and reward weights are **untuned**; MARL training can be unstable and sensitive to them.
* Independent DQN is not QMIX/MADDPG; no explicit handling of non-stationarity.
* **Federated learning is logical (one process)**: no network, no encryption, no secure aggregation, no differential privacy; updates can leak information; no privacy guarantee is claimed. FedAvg under non-IID traffic is a compromise.
* Few seeds → wide uncertainty; GGF weight family is our choice; Jain's index measures equality not efficiency.
* Fuel is indicative (default emission class). Pedestrians, cyclists, public transport, incidents, real sensor data: **not implemented** (future work).
* Dynamic fairness weighting exists only as an optional extension, off by default.
