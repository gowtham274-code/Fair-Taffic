# Algorithms

## Multi-agent RL: independent Double-DQN with neighbour-aware observation
* **Observation (18 floats)**: for N,E,S,W: queue/10, mean local wait/60, vehicles/10; one-hot current phase (4); time in phase/60; mean neighbour queue/10. Clipped to [0,3].
* **Action**: desired green phase (0…3, masked to existing phases). A switch first inserts yellow (3 s) derived from the signal's own states; min-green 10 s; decision interval 5 s.
* **Learning**: Double DQN, Huber loss, Adam (lr 1e-3), γ = 0.95, replay 20 000, batch 64, target sync every 200 updates, ε linear 1.0→0.05, gradient clip 5. Networks 18→64→64→4 (ReLU). *All defaults are untuned.*
* **Backends**: PyTorch (intended; not executed in the sandbox) or an equivalent NumPy implementation with manual back-propagation (executed and tested, incl. a finite-difference gradient check). Same weight layout, so checkpoints/FedAvg are backend-independent.
* **Not QMIX/MADDPG.** There is no centralised critic or mixing network; non-stationarity from simultaneous learners is not explicitly handled. Cooperation comes from the shared fairness term and the neighbour-queue feature.
* **Evaluation** uses ε = 0 and never calls the learner (tested: weights unchanged).

## Federated averaging
Per round: (1) clients train locally for `episodes_per_round` episodes; (2) participating clients (`participation_rate`) send weights, sample count, mean loss; (3) the server validates names/shapes/finiteness and rejects bad updates with a logged reason; (4) FedAvg; (5) global model versioned and checkpointed; (6) distributed to clients (and target nets); (7) round logged (participants, weights, bytes, update norm, optional validation episode). Replay buffers, optimiser state and exploration stay local.
Limitations: non-IID traffic across intersections → one averaged policy is a compromise; averaging Q-networks interacts with target networks and Adam state; no secure aggregation, encryption or differential privacy.
