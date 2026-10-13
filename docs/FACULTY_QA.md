# Likely faculty questions
**Is this really MARL?** Four agents learn concurrently in one environment, each with its own policy/network and a fairness-shaped reward; it is independent Double-DQN with neighbour information — not QMIX. Limitations are in ALGORITHMS.md.
**What is GGF and why not variance?** Order-weighted sum of sorted utilities with largest weight on the worst-off; variance ignores who is worst-off and is only reported as an extra metric.
**Is the federated learning real?** Real FedAvg on real model weights, but all clients run in one process (logical FL). Raw observations are not sent to the aggregator; weights may still leak information; no encryption/DP.
**Why not just centralised training?** It needs all experience in one place — we run it only as a separate baseline.
**Are results valid?** Only SUMO runs with multiple seeds count; report mean ± std; mock results are pipeline checks.
**Does fairness cost efficiency?** Compare avg wait/throughput next to Jain/GGF/max wait — that is the experiment; no conclusion is claimed before results exist.
**Slide says "encrypted updates"?** Not implemented; it would require TLS/secure aggregation in a real deployment.
**Was it deployed?** Not yet; Docker files are provided but unverified.
