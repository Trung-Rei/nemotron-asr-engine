## Open workload sweep (chunk 320 ms, float32, max batch 64)

| backend | target streams | mean live | chunks/s | mean batch | lag p50/p95/p99 ms | finalize p50/p95/p99 ms | e2e finalize p95 ms | GPU % | WER % | SLO |
|---|---|---|---|---|---|---|---|---|---|---|
| nemo | 64 | 79 | 246 | 16.9 | 77/137/1340 | 77/207/1550 | 491 | - | 19.3 | ✅ |
| nemo | 128 | 134 | 418 | 30.0 | 90/148/163 | 87/147/163 | 421 | - | 20.7 | ✅ |
| nemo | 192 | 194 | 607 | 45.6 | 97/156/169 | 95/154/171 | 421 | - | 24.6 | ✅ |
| nemo | 256 | 273 | 853 | 62.5 | 173/778/939 | 176/763/887 | 970 | - | 23.9 | ❌ |

Capacity (largest tested level meeting p95 finalize ≤ SLO, p95 lag ≤ chunk, no rejects): **nemo: 192 streams**

