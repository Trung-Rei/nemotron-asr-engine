# Benchmark

*[English](README.md)*

Cài đặt trước mọi thứ, kể cả NeMo, trên máy có GPU NVIDIA:

```bash
pip install -r research/requirements.txt
```

`bash bench/run_all.sh` chạy toàn bộ. Chọn các stage bằng `STAGES="test parity rtfx capacity ws cli profile"`. Kết quả ghi vào `$RESULTS_DIR` (mặc định `results/`). Kết quả và phương pháp nằm ở [docs/benchmarks.vi.md](../docs/benchmarks.vi.md).

| Thư mục | Script | Đo gì |
|---|---|---|
| `parity/` | `run_parity.sh` | Độ chính xác của bản port so với NeMo: offline và streaming, mức trùng khớp từng câu, WER. Có cả parity tensor từng lớp (stage `dump` / `parity`). |
| `rtfx/` | `bench_rtfx.sh` | Thông lượng offline (RTFx) bằng chính `transcribe_speech.py` của NeMo, so với bản port, cùng manifest và cùng cấu hình. |
| `streaming/` | `engine_loadtest.py` | Load test thời gian thực qua API asyncio của engine: luồng đến theo Poisson, gói 20 ms theo đồng hồ thật, `end_turn` ở cuối mỗi câu. Báo độ trễ finalize, chunk lag, thông lượng và WER. |
| `streaming/` | `loadtest.py`, `nemo_vs_lite.sh` | Cùng workload, do một scheduler đồng bộ điều khiển, cho mô hình streaming cache-aware của NeMo và cho bản port. Sức tải của NeMo được đo bằng cách này. |
| `streaming/` | `ws_loadtest.py`, `ws_bench.sh` | Cùng workload, chạy qua server WebSocket, client ở các process riêng, có theo dõi bộ nhớ GPU và RSS. |
| `streaming/` | `ws_smoke.sh` | Smoke test server: WER trên các luồng 8 kHz μ-law đồng thời, và mã đóng 1013 khi hết slot. |
| `profiling/` | `graph_breakdown.py`, `profile_engine.py`, `profile_server.py`, `profile_step.py`, `multi_engine_stress.sh` | Thời gian từng phần trong một bước, các pha của worker và thời gian chờ GIL, chi phí event loop của server, hai engine trên một GPU. |

**Workload** (`streaming/workload.py`):
- Mỗi session là một luồng 16 kHz liên tục.
- Câu nói và khoảng lặng lấy theo phân phối log-normal: câu nói p50 2,4 s / p90 12,3 s, khoảng lặng p50 6,5 s / p90 25 s, khoảng 30% thời gian là tiếng nói.
- Câu nói được lấp bằng audio VietSuperSpeech, khoảng lặng có nhiễu nhỏ.
- `--telephone` lọc audio xuống dải 8 kHz.
