# Benchmark

*[English](benchmarks.md)*

Mọi số liệu trên trang này lấy từ một lần chạy [`bench/run_all.sh`](../bench/run_all.sh) trên một **NVIDIA A100 40 GB** (thuê riêng, không chạy chung với việc khác): A100-SXM4-40GB, driver 610.57, PyTorch 2.7.1 + CUDA 12.8, trọng số fp32 với matmul TF32 trừ khi ghi khác, máy chủ 12 core CPU. Ngày đo: 2026-11-12. Kết quả thô (bản tóm tắt): [`docs/results/a100-2026-11-12/`](results/a100-2026-11-12).

## Đo cái gì, và vì sao

Thông lượng offline (RTFx: số giây audio xử lý được trong một giây tính toán) là benchmark ASR thông dụng. Nó không trả lời được câu hỏi của một dịch vụ trực tiếp:

> Một GPU phục vụ được bao nhiêu luồng thời gian thực đồng thời, mà transcript cuối của mỗi câu vẫn ra nhanh sau khi người nói dừng?

Trong một dịch vụ trực tiếp:

- audio đến thành từng gói nhỏ theo đồng hồ thật, bất kể server có theo kịp hay không;
- mỗi luồng mở trong nhiều phút, kể cả lúc im lặng;
- điều người dùng cảm nhận là độ trễ từ lúc hết câu tới lúc có text cuối.

### Workload

[`bench/streaming/workload.py`](../bench/streaming/workload.py):

- **Session** đến theo quá trình Poisson. Mỗi session là một luồng 16 kHz liên tục dài 40 s.
- **Câu nói và khoảng lặng** lấy theo phân phối log-normal: câu nói p50 2,4 s / p90 12,3 s, khoảng lặng p50 6,5 s / p90 25 s. Khoảng 30% thời gian là tiếng nói.
- **Audio của câu nói** lấy từ VietSuperSpeech (công khai, có transcript tham chiếu nên tính được WER). Khoảng lặng có nhiễu nhỏ.
- **Gói** 20 ms được đẩy theo đồng hồ thật. `end_turn()` được gọi đúng lúc câu kết thúc (một VAD lý tưởng).
- **Kích thước chunk** là 320 ms (`lookahead=3`).

### Chỉ số

| Chỉ số | Định nghĩa |
|---|---|
| **Finalize** | Từ `end_turn()` tới lúc client nhận sự kiện final. |
| **Chunk lag** | Từ lúc audio của một chunk đã đủ tới lúc token của nó ra. |
| **SLO** | Finalize p95 ≤ 300 ms, chunk lag p95 ≤ 320 ms, không từ chối luồng nào. Đo từ giây 40 tới giây 90 của mỗi mức tải. |

### Mốc so sánh NeMo

NeMo không có sẵn API stream kiểu asyncio. Nó được điều khiển bởi [`bench/streaming/loadtest.py`](../bench/streaming/loadtest.py), một scheduler đồng bộ:

- tính sẵn feature;
- gom các chunk đã sẵn sàng lâu nhất thành batch (tối đa 64);
- gọi bước streaming cache-aware của NeMo (`conformer_stream_step`).

Scheduler này không tốn gì trên event loop, nên đây là trường hợp thuận lợi nhất cho NeMo.

## Kết quả

### Sức tải: số luồng thời gian thực đồng thời trên một A100

| | Số luồng đạt SLO | Finalize p50 / p95 / p99 ở mức đó | Chunk lag p50 / p95 | Thông lượng ở mức đó |
|---|---|---|---|---|
| NeMo streaming cache-aware, scheduler đồng bộ | **192** | 95 / 154 / 171 ms ¹ | 97 / 156 ms | 607 chunk/s, batch 45,6 |
| **nemotron-engine, API asyncio** | **384** | **51 / 96 / 120 ms** | 48 / 86 ms | 1.278 chunk/s, batch 45,0 |
| nemotron-engine, server WebSocket (4 process client) | **256** | 26 / 65 / 206 ms | 19 / 30 ms | |

¹ "Finalize" của NeMo đo phía server: tính từ lúc chunk chứa mẫu cuối của câu đã đủ, không gồm thời gian chờ chunk đó đầy. Đo đầu-cuối từ lúc hết tiếng, p95 của NeMo ở 192 luồng là **421 ms**. Số của engine luôn là đầu-cuối: từ `end_turn()` tới lúc client nhận final, gồm cả việc đệm chunk đang dở. Vì vậy phép so sánh sức tải này đang nghiêng về phía có lợi cho NeMo.

Mọi mức tải:

**NeMo** (`bench/streaming/nemo_vs_lite.sh`, backend `nemo`)

| Số luồng mục tiêu | Chunk/s | Batch TB | Chunk lag p50/p95 ms | Finalize p50/p95/p99 ms | Finalize đầu-cuối p95 | WER % | SLO |
|---|---|---|---|---|---|---|---|
| 64 | 246 | 16,9 | 77 / 137 | 77 / 207 / 1340 | 491 | 19,3 | ✅ |
| 128 | 418 | 30,0 | 90 / 148 | 87 / 147 / 163 | 421 | 20,7 | ✅ |
| 192 | 607 | 45,6 | 97 / 156 | 95 / 154 / 171 | 421 | 24,6 | ✅ |
| 256 | 853 | 62,5 | 173 / 778 | 176 / 763 / 887 | 970 | 23,9 | ❌ |

**nemotron-engine, API asyncio** (`bench/streaming/engine_loadtest.py`)

| Số luồng mục tiêu | Chunk/s | Batch TB | Chunk lag p50/p95 ms | Finalize p50/p95/p99 ms | CPU của process | WER % | SLO |
|---|---|---|---|---|---|---|---|
| 256 | 868 | 11,8 | 18 / 26 | 19 / 30 / 37 | 38% | 25,0 | ✅ |
| 384 | 1.278 | 45,0 | 48 / 86 | 51 / 96 / 120 | 48% | 25,1 | ✅ |
| 448 | 1.501 | 63,7 | 1.457 / 2.723 | 1.426 / 2.732 / 2.772 | 57% | 23,2 | ❌ |
| 512 | 1.781 | 63,9 | 8.106 / 14.165 | 8.083 / 14.126 / 14.693 | 60% | 23,6 | ❌ (13 luồng bị từ chối: hết slot) |

Ở 448 luồng, batch trung bình chạm trần 64 và chunk bắt đầu xếp hàng. GPU bão hoà ở khoảng 1.500–1.800 chunk/s. Khi đo riêng, một graph cả bước cho 64 luồng mất 27,7 ms trên A100 này. Khi có tải còn cộng thêm các batch chunk đầu nhỏ, graph nối tiếp và khoảng trống giữa các bước (xem [architecture.vi.md](architecture.vi.md#một-bước-tốn-thời-gian-ở-đâu)).

Ranh giới giữa 384 và 448 không sắc nét. Ở một lần chạy thứ hai, ngắn hơn (`bench/profiling/profile_engine.py`, 70 s mỗi mức), 448 luồng đạt với finalize p95 41 ms, còn 512 luồng trượt ở p95 516 ms. Bảng trên giữ kết quả chặt hơn.

**Khi tắt CUDA graph** (`--no-cuda-graphs`), ngay 256 luồng đã trượt: finalize p95 973 ms, batch kẹt ở 63,6, CPU của process 102%. Bước nào cũng bị giới hạn bởi việc gọi kernel, khoảng 41 ms.

**`nemotron-engine bench`**, lệnh kiểm tra nhanh của CLI, dùng workload đơn giản hơn: mỗi luồng nói 3 s rồi lặng 5 s, bắt đầu ở một điểm ngẫu nhiên trong chu kỳ đó, và mọi luồng mở suốt mức tải (không có luồng mới đến, không có đợt chunk đầu dồn dập). Trên A100 này nó cho:

| Số luồng | Chunk/s | Batch TB | Finalize p50 / p95 / p99 |
|---|---|---|---|
| 128 | 416 | 3,7 | 22 / 27 / 28 ms |
| 256 | 829 | 10,0 | 31 / 39 / 41 ms |
| 448 | 1.445 | 35,5 | 73 / 138 / 204 ms |

Dùng lệnh này để định cỡ GPU của bạn cho nhanh. Khi cần số liệu để so sánh, hãy dùng load test đầy đủ.

**nemotron-engine, server WebSocket** (`bench/streaming/ws_bench.sh`, PCM16 16 kHz, frame 20 ms, client chạy trong 4 process trên cùng máy)

| Số luồng mục tiêu | Bị từ chối | Finalize p50/p95/p99 ms | Chunk lag p50/p95 ms | WER % | SLO |
|---|---|---|---|---|---|
| 256 | 0 | 26 / 65 / 206 | 19 / 30 | 25,0 | ✅ |
| 384 | 0 | 337 / 849 / 938 | 116 / 588 | 25,2 | ❌ |
| 448 | 58 | 1.862 / 2.703 / 3.473 | 1.613 / 2.255 | 23,2 | ❌ |
| 512 | 161 | 1.526 / 3.614 / 3.775 | 1.144 / 3.183 | 23,7 | ❌ |

Qua server thì sức tải thấp hơn gọi engine trực tiếp trong process. Một event loop Python phải nhận 50 frame WebSocket mỗi giây cho mỗi luồng, và tranh GIL với worker GPU. Frame lớn hơn (60–100 ms) hoặc tách process front-end sẽ lấy lại phần lớn chênh lệch; xem [optimization-journey.vi.md](optimization-journey.vi.md#hiện-trạng-và-bước-tiếp-theo).

### Bộ nhớ

Lần chạy qua WebSocket dùng `max_streams=704`. Bộ nhớ GPU được cấp lúc khởi động và đứng yên qua mọi mức tải và các lúc nghỉ:

| | Giá trị |
|---|---|
| PyTorch cấp phát | 11.657 MB |
| PyTorch giữ (reserved) | 12.206 MB |
| nvidia-smi | 12.989 MB |
| RSS của process server | 1.585 MB lúc đầu → 1.972 MB sau các mức quá tải; tăng chủ yếu khi server quá tải rồi đứng yên lúc nghỉ (chưa làm soak test ở một mức cố định) |

Mỗi slot tốn khoảng 11,8 MB, chủ yếu là K/V của 24 lớp × 56 frame. Phần cố định (trọng số, graph, buffer) khoảng 3,4 GB. Ước lượng nhanh: **VRAM ≈ 3,4 GB + 11,8 MB × max_streams**, cộng khoảng 1,3 GB cho context CUDA và phần dư của allocator.

### Độ chính xác khi có tải và với audio 8 kHz

Từ `tests/test_engine.py --strict`: 64 luồng đồng thời, strict fp32.

| Kiểm tra | Kết quả |
|---|---|
| Engine so với bản port tham chiếu, gói 20 ms | 64/64 luồng giống hệt, 0 trên 9.272 token khác |
| Gói 100 ms so với 20 ms | giống hệt |
| `greedy_steps=1` (169 lần replay graph nối tiếp) | giống hệt |
| 50 câu: 16 kHz gốc / 16 kHz dải điện thoại / 8 kHz μ-law | WER 26,4% / 29,5% / 28,9% |
| Stress: engine đầy, đóng giữa câu, `end_turn` khi không có audio, `aclose` khi còn luồng mở | đạt, không rò slot |

## Độ chính xác

Bản port tham chiếu (`research/nemotron_lite`) so với NeMo, 500 câu validation của VietSuperSpeech (seed 42), strict fp32 (`bench/parity/run_parity.sh`):

| Chunk | NeMo offline | Port offline | NeMo streaming | Port streaming |
|---|---|---|---|---|
| 1120 ms (`56,13`) | 23,91% | 23,91% | 23,94% | 23,91% |
| 320 ms (`56,3`) | 25,18% | 25,19% | 25,27% | 25,19% |

Mức trùng khớp từng câu của text đã chuẩn hoá (`$RESULTS_DIR/comparison.md`):

| Cặp | 1120 ms | 320 ms |
|---|---|---|
| Port offline so với NeMo offline | 99,2% giống hệt | 99,0% giống hệt |
| Port streaming so với port offline | 100% giống hệt (cả token) | 100% giống hệt (cả token) |
| NeMo streaming so với NeMo offline | 98,8% giống hệt | 83,0% giống hệt |

Vòng lặp streaming của chính NeMo (ví dụ `conformer_stream_step` của NeMo) trả về một tiền tố của transcript offline ở 6/500 câu với chunk 1120 ms, và 85/500 câu với chunk 320 ms: nó bỏ mất token ở cuối audio. Đường streaming của bản port có flush chunk cuối nên khớp offline tuyệt đối.

Token của engine giống hệt bản port (xem phía trên), nên engine có cùng độ chính xác với bản port.

## Chạy lại

```bash
pip install -r research/requirements.txt
STAGES="test parity capacity ws cli profile" RESULTS_DIR=results bash bench/run_all.sh
```

Hoặc trên GPU thuê: `vkong run --detach` với [`vkong.yaml`](../vkong.yaml) của repo.

Từng stage riêng:

```bash
python tests/test_engine.py --strict --n 64                                   # parity token, stress
bash bench/parity/run_parity.sh ref bench stream-ref stream-bench compare      # độ chính xác so với NeMo
VA_BACKENDS=nemo VA_PHASES=sweep bash bench/streaming/nemo_vs_lite.sh          # sức tải của NeMo
python bench/streaming/engine_loadtest.py --levels 256 448 512 --out results/engine   # sức tải của engine
bash bench/streaming/ws_bench.sh 256 384 448                                   # qua server WebSocket
nemotron-engine bench --streams 64 128 256 --wav speech.wav                    # kiểm tra nhanh trên GPU của bạn
```

Số đo ở sát ngưỡng sức tải dao động giữa các lần chạy. Ở mức cao nhất còn đạt, finalize p95 của cùng một cấu hình có thể lệch nhau vài chục mili giây giữa các lần. Chỉ nên so sánh số đo trên cùng GPU và trong cùng một lần chạy.
