| backend | scenario | variant | runs | avg_wait_s | max_wait_s | jain_index | ggf_score | throughput_veh_per_h | unfinished_vehicles |
|---|---|---|---|---|---|---|---|---|---|
| sumo | medium | baseline | 1 | 53.77 ± 12.24 | 300.33 ± 69.21 | 0.61 ± 0.05 | -56.71 ± 16.36 | 1812.00 ± 27.28 | 104.33 ± 11.90 |
| sumo | medium | federated | 1 | 24.04 ± 1.65 | 114.67 ± 11.90 | 0.71 ± 0.06 | -20.79 ± 3.29 | 1898.67 ± 31.38 | 61.00 ± 9.27 |
| sumo | medium | marl | 1 | 21.45 ± 0.48 | 120.00 ± 19.60 | 0.79 ± 0.01 | -16.59 ± 0.31 | 1897.33 ± 45.21 | 61.67 ± 15.97 |
| sumo | uneven | baseline | 1 | 31.26 ± 6.69 | 207.50 ± 76.50 | 0.54 ± 0.21 | -46.43 ± 24.94 | 1632.00 ± 36.00 | 63.50 ± 9.50 |

Better: avg_wait_s lower; max_wait_s lower; jain_index higher; ggf_score higher (closer to 0); throughput_veh_per_h higher; unfinished_vehicles lower
