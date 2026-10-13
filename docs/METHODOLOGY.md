# Methodology

## Problem
Efficiency-focused signal control (fixed-time, delay-minimising RL) can give some approaches much longer waits. We define **fairness** as an equitable waiting-time distribution among approaches (junction × compass side), while still keeping overall waiting low.

## Why these choices
* **Fixed-time** ignores demand, so queues build where demand is high. **MARL** lets interconnected intersections adapt to local and neighbouring traffic.
* An efficiency-only reward is blind to *who* waits; the mean can fall while the worst approaches get worse.
* **Federated learning**: agents train on local experience and share only model weights, so raw local observations are not centrally collected. This reduces, but does **not** eliminate, privacy risk: weight updates can leak information; no encryption/DP is implemented.

## Reward (fairness/reward.py)
For agent *i* at each decision step

r_i = w_wait·r_wait,i + w_queue·r_queue,i + w_thr·r_thr,i + w_fair·r_fair

* r_wait,i = −mean local wait on i's approaches / 60 s; r_queue,i = −halting vehicles / 20; r_thr,i = vehicles served / 5 (all clipped to ±5).
* "Local wait" = waiting accrued since the vehicle entered the approach, so no delay is counted twice.
* r_fair = GGF over utilities u_g = −(mean local wait of approach g)/60 for all approaches in the network (shared team signal).
* Defaults w = (1, 0.25, 0.1, 1.0) are **untuned engineering choices**. `w_fairness = 0` gives the conventional delay-only RL baseline. Components are logged separately.

## Generalized Gini Welfare Function
Sort utilities ascending, u₍₁₎ ≤ … ≤ u₍ₙ₎, and take GGF(u) = Σₖ wₖ u₍ₖ with w₁ ≥ … ≥ wₙ ≥ 0, Σwₖ = 1 (default wₖ ∝ 0.5^(k−1)). The worst-off group gets the largest weight. Equal weights give the mean. **GGF is not the variance.** Weight family is our decision (the sources do not fix it). Hand-checked example in `tests/unit/test_fairness.py`.

## Evaluation metrics
Average/maximum waiting time, waiting-time variance, **Jain's index** J = (Σx)²/(nΣx²) ∈ [1/n,1] over approach means (higher = more equal), GGF score (closer to 0 better), throughput = completed/hour, queue length, unfinished vehicles, fuel (SUMO only). Unfinished vehicles are reported so a controller cannot look good by leaving cars behind. Jain measures equality only, so it is always read together with mean wait.

## Training architectures (kept strictly separate)
| | Parameters | Experience | Federated? |
|---|---|---|---|
| independent (`marl`) | one net per agent | local | no |
| centralized | one shared net | pooled in one place | **no** |
| federated | one net per agent, FedAvg every k episodes | local, never leaves client | yes (logical) |

FedAvg: θ_global = Σₖ (nₖ/Σⱼnⱼ) θₖ with nₖ = local transitions in the round. All nets share one architecture and all agents have identical observation/action sizes, so parameters correspond.

## Hypotheses (to test, not claims)
H1 fairness-aware MARL has lower worst-case/approach-level wait disparity than efficiency-only RL at similar throughput; H2 federated training reaches performance comparable to independent training; H3 both beat fixed-time on mean wait. **No result is claimed here**; see EXPERIMENTS.md.

## Limitations of simulation evidence
See RESEARCH_LIMITATIONS.md.
