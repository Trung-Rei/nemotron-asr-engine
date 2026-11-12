| run | batch | chunks/s | finalize p95 | rounds/s | worker CPU % | loop CPU % | GPU enc / greedy / idle % | drain ms/round (blocked) | prepare ms/round (blocked) | decode ms/round (blocked) | finish ms/round (blocked) | flush ms/round (blocked) | submit ms/round (blocked) | idle ms/round (blocked) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c448_p20 | 26.6 | 1499 | 41 | 68.7 | 31 | 34 | 86 / 5 / 8 | 2.48 (0.03) | 0.91 (0.69) | 0.28 (0.03) | 0.40 (0.08) | 0.05 (0.02) | 0.73 (0.13) | 9.65 (9.07) |
| c512_p20 | 59.7 | 1796 | 516 | 36.4 | 37 | 43 | 82 / 9 / 10 | 7.69 (0.72) | 1.89 (1.44) | 0.72 (0.24) | 0.91 (0.31) | 0.17 (0.12) | 1.06 (0.46) | 15.21 (14.17) |
| c512_p100 | 63.8 | 1795 | 5508 | 24.2 | 36 | 18 | 87 / 7 / 6 | 11.20 (1.33) | 1.21 (0.55) | 0.95 (0.01) | 1.20 (0.02) | 0.32 (0.24) | 0.66 (0.04) | 25.58 (24.01) |
