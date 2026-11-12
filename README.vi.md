# nemotron-engine

*[English](README.md)*

**Phục vụ hàng trăm luồng nhận dạng giọng nói trực tiếp trên một GPU.**

nemotron-engine là engine ASR streaming thời gian thực, nhiều luồng, cho mô hình [Nemotron ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) của NVIDIA (FastConformer + RNN-T, khoảng 100 ngôn ngữ). Engine viết bằng PyTorch thuần, lúc chạy không cần NeMo. Bạn có thể:

- nhúng engine vào ứng dụng asyncio;
- chạy engine như một server WebSocket;
- thử ngay trên dòng lệnh hoặc trình duyệt.

Trên một A100 40 GB, engine phục vụ **384 luồng thời gian thực đồng thời**, transcript cuối có sau **96 ms** (p95) kể từ lúc hết câu. Chính NeMo (streaming cache-aware), được gom batch bằng cùng scheduler trên cùng GPU, chịu được 192.

- **Kết quả giống hệt.** Token giống từng bit so với bản port tham chiếu, bản này khớp NeMo từng lớp (strict fp32). WER trên 500 câu bằng NeMo.
- **Nhiều luồng trên một GPU.** Chunk của mọi luồng đang mở được gom thành một bước GPU. Cả bước (log-mel → encoder → greedy RNN-T) là một lần replay CUDA graph.
- **Final ra nhanh.** Khi VAD của bạn báo hết câu, transcript cuối có sau một bước GPU, thay vì phải chờ thêm tới 320 ms audio.
- **Dễ dùng.** API `async for event in stream`, server WebSocket kèm trang demo, đầu vào PCM16 / μ-law / float32 ở 8 hoặc 16 kHz, và tự từ chối luồng mới khi GPU đã bão hoà.

> Dự án không chính thức, không liên quan và không được NVIDIA bảo trợ. Trọng số mô hình tải từ Hugging Face và tuân theo license [OpenMDW-1.1](https://openmdw.ai/license/1-1/) của NVIDIA.

## Bắt đầu nhanh

Cần GPU NVIDIA có CUDA, Python ≥ 3.10 và PyTorch ≥ 2.4.

```bash
pip install "nemotron-engine[server] @ git+https://github.com/Trung-Rei/nemotron-engine"

# nhận dạng file (mỗi file là một luồng chạy song song); lần đầu tự tải mô hình (~2,5 GB)
nemotron-engine transcribe speech.wav --language vi-VN

# server WebSocket + trang demo thu mic tại http://localhost:8090
nemotron-engine serve --port 8090 --max-streams 256
```

Trang demo cần quyền dùng micro, mà trình duyệt chỉ cho phép trên `localhost` hoặc HTTPS. Nếu GPU ở máy khác, mở tunnel trước: `ssh -L 8090:localhost:8090 may-gpu`.

### Trong Python

```python
import asyncio
from nemotron_engine import AsyncNemotronEngine, EngineConfig

async def main():
    engine = await AsyncNemotronEngine.create(EngineConfig(max_streams=256, language="vi-VN"))
    stream = engine.open_stream(sample_rate=16000, encoding="pcm16")   # mỗi nguồn audio trực tiếp một stream

    async def read():
        async for ev in stream:                 # TranscriptEvent
            if ev.kind == "partial":
                print("…", ev.text)             # toàn bộ text của câu đang nói
            else:
                print("✔", ev.text)             # "final": câu đã xong

    reader = asyncio.create_task(read())
    for packet in microphone_packets():         # bytes, kích thước tuỳ ý, đến đâu đẩy đến đó
        await stream.push(packet)
        if vad_says_utterance_ended():
            stream.end_turn()                   # -> đúng một sự kiện "final" cho câu này
    await stream.close()                        # chốt câu đang dở, trả slot
    await reader
    await engine.aclose()

asyncio.run(main())
```

Các phiên bản chạy được nằm trong [`examples/`](examples):

| Ví dụ | Minh hoạ |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | một file, đẩy gói theo thời gian thực, partial trực tiếp |
| [`many_streams.py`](examples/many_streams.py) | hàng trăm luồng đồng thời trong một process |
| [`ws_client.py`](examples/ws_client.py) | client WebSocket, đo độ trễ phía client |

## Benchmark

Mọi dòng đo trên cùng một NVIDIA A100 40 GB, chunk 320 ms, gói 20 ms theo thời gian thực, audio VietSuperSpeech. SLO: finalize p95 ≤ 300 ms, chunk lag p95 ≤ 320 ms, không từ chối luồng nào.

| | Số luồng đồng thời đạt SLO | Finalize p50 / p95 ở mức đó | WER |
|---|---|---|---|
| NeMo streaming cache-aware (gom batch, scheduler đồng bộ) | 192 | 95 / 154 ms | bằng nhau |
| **nemotron-engine, API asyncio** | **384** | **51 / 96 ms** | bằng nhau |
| nemotron-engine, server WebSocket | 256 | 26 / 65 ms | bằng nhau |

- **Độ chính xác.** Bản port tham chiếu khớp NeMo: WER 23,91% so với 23,91% ở chunk 1120 ms, và 25,19% so với 25,18% ở chunk 320 ms, trên 500 câu. Token của engine giống hệt bản port (64/64 stream, 0 trên 9.272 token khác).
- **Phạm vi.** Engine được làm cho luồng trực tiếp. Với nhận dạng file offline theo lô, engine không nhanh hơn đường offline của NeMo: RTFx của bản port tham chiếu đo trên A100 chỉ lệch NeMo vài phần trăm.

Phương pháp, toàn bộ số liệu và cách chạy lại: [docs/benchmarks.vi.md](docs/benchmarks.vi.md).

## API

### Đầu vào

`engine.open_stream(language=None, sample_rate=16000, encoding="pcm16")`

| `encoding` | Dữ liệu `push()` | `sample_rate` |
|---|---|---|
| `pcm16` | bytes, 16-bit little-endian | 16000 hoặc 8000 |
| `mulaw` | bytes, G.711 μ-law | 8000 (hoặc 16000) |
| `f32` | bytes, float32 | 16000 hoặc 8000 |

- `push()` nhận cả mảng numpy (float32 hoặc int16).
- Gói có kích thước tuỳ ý. Kết quả không phụ thuộc cách cắt gói.
- Audio 8 kHz được nâng lên 16 kHz bằng bộ lọc polyphase 64 tap chạy theo luồng (trễ khoảng 2 ms).
- `language` nhận locale như `vi-VN`, `en-US`, `ja-JP`, hoặc `auto`. Mỗi stream có thể dùng ngôn ngữ khác nhau.

### Sự kiện

Các trường của `TranscriptEvent`: `kind` (`"partial"` / `"final"`), `turn` (số thứ tự câu), `text` (text của câu tới lúc đó), `delta` (phần mới thêm), `tokens`, `audio_ms`, và các mốc thời gian `t_ready` / `t_emit` (`time.perf_counter`).

- **Partial** đi kèm mỗi chunk 320 ms có thêm text. RNN-T chỉ nối thêm, nên partial không bao giờ bị sửa lại.
- **Final** có đúng một lần cho mỗi `end_turn()`.

### Kết thúc một câu nói

| Lời gọi | Cách hoạt động | Đánh đổi |
|---|---|---|
| `end_turn(pad=True)` (mặc định) | Đệm im lặng cho đủ chunk đang dở và decode ngay. Final có sau một bước GPU. | Dòng thời gian của stream bị chèn tối đa 320 ms im lặng. Token mô hình phát muộn, thường là dấu câu, có thể rơi vào đầu câu sau. |
| `end_turn(pad=False)` | Chờ audio thật làm đủ chunk. | Thêm 0–320 ms độ trễ. |

Engine không tự phát hiện hết câu: hãy dùng VAD hoặc endpointer của bạn. Trang demo của server có sẵn một VAD đơn giản theo năng lượng.

### Lỗi và chặn nhận luồng

| Exception | Khi nào |
|---|---|
| `EngineFull` | `open_stream` khi cả `max_streams` slot đều đang dùng. |
| `EngineOverloaded` (lớp con của `EngineFull`) | Chunk chờ GPU lâu nhất đã chờ quá `max_lag_ms` (mặc định 200 ms). Engine bão hoà từ chối luồng mới thay vì làm chậm mọi luồng. |
| `EngineClosed` | Dùng stream hoặc engine đã đóng. |

### Cấu hình (`EngineConfig`)

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `max_streams` | 512 | Số slot stream đồng thời. Cache của chúng được cấp sẵn từ đầu (xem phần định cỡ bên dưới). |
| `max_batch` | 64 | Số stream tối đa trong một bước GPU. |
| `language` | `"auto"` | Prompt ngôn ngữ mặc định. |
| `lookahead` | 3 | Chunk = (lookahead + 1) × 80 ms. Mô hình hỗ trợ 0, 3, 6, 13. |
| `max_lag_ms` | 200 | Ngưỡng chặn nhận luồng; `None` để tắt. |
| `matmul_precision` | `"high"` | Matmul TF32; `"highest"` cho strict fp32. |
| `cuda_graphs` | `True` | `False` chạy eager: chậm hơn, dùng để debug. |
| `model_dir` | `None` | Thư mục local chứa `config.json`, `processor_config.json`, `tokenizer.json` và `model.safetensors`, thay cho việc tải về. |

### Định cỡ

Bộ nhớ GPU được giữ từ lúc khởi động và không đổi sau đó. Phần lớn là cache theo slot (K/V của 24 lớp × 56 frame). Dòng 704 là số đo thực trên A100; các dòng còn lại ước tính từ đó.

| `max_streams` | PyTorch cấp phát | Tổng bộ nhớ GPU (≈ +1,3 GB context CUDA và allocator) |
|---|---|---|
| 128 | 4,9 GB | 6,2 GB |
| 256 | 6,4 GB | 7,7 GB |
| 512 | 9,4 GB | 10,7 GB |
| 704 | 11,66 GB (đo thực tế) | 12,99 GB (đo thực tế) |
| 1024 | 15,5 GB | 16,8 GB |

Quy tắc rút ra từ lần đo trên A100: **khoảng 3,4 GB + 11,8 MB cho mỗi slot**. Đặt `max_streams` theo bộ nhớ GPU. Nên đặt cao hơn mức tải GPU phục vụ được trong ngưỡng độ trễ, và để `max_lag_ms` từ chối luồng mới khi GPU đã bão hoà.

Thông lượng bị giới hạn bởi sức tính của GPU chứ không phải bộ nhớ: xem bảng benchmark để biết một GPU chịu được bao nhiêu luồng đang hoạt động.

## Server WebSocket

```bash
nemotron-engine serve --host 0.0.0.0 --port 8090 --max-streams 256 [--language vi-VN]
# tương đương: NEMOTRON_MAX_STREAMS=256 uvicorn nemotron_engine.server:app --port 8090   (một worker cho mỗi GPU)
```

**`WS /v1/stream?language=vi-VN&sample_rate=16000&encoding=pcm16`**

| Chiều | Thông điệp |
|---|---|
| client → server | Frame nhị phân: audio theo encoding của stream, kích thước tuỳ ý. |
| client → server | Text `{"type": "end_turn", "pad": true}`: kết thúc câu nói. |
| client → server | Text `{"type": "close"}`: chốt câu đang dở rồi đóng. |
| server → client | `{"type": "partial" \| "final", "turn", "text", "delta", "audio_ms", "latency_ms"}` |
| server → client | `{"type": "closed"}` sau sự kiện cuối cùng. |

Mã đóng socket: `1013` engine đầy hoặc quá tải (thử lại sau), `1003` tham số hoặc lệnh sai, `1011` engine lỗi.

- **`GET /`** trả về trang demo thu mic.
- **`GET /health`** trả về số liệu của engine (số stream đang mở, slot trống, chunk lag, thời gian mỗi bước) và bộ nhớ.

Chạy một process server cho mỗi GPU, ví dụ dùng `CUDA_VISIBLE_DEVICES`. Nếu có nhiều GPU, đặt một load balancer phía trước.

## Cách hoạt động

Đọc [docs/architecture.vi.md](docs/architecture.vi.md) để biết thiết kế, và [docs/optimization-journey.vi.md](docs/optimization-journey.vi.md) để biết quá trình đi tới đó: đã đo gì, đã sửa gì, và những gì không hiệu quả. Tóm tắt:

1. Một worker thread duy nhất giữ GPU. Event loop của bạn chỉ giải mã gói rồi thêm vào inbox.
2. Mỗi stream có một slot trong các tensor cache cấp sẵn. Một batch là gather → 24 lớp → scatter theo chỉ số slot.
3. Cả bước, từ mẫu thô đến nhãn RNN-T, là một CUDA graph cho mỗi bucket batch. Host chỉ gọi khoảng 6 lời gọi PyTorch mỗi bước, nhờ vậy GIL của Python không cản đường GPU.
4. Greedy RNN-T chạy trong graph, mỗi stream có con trỏ frame riêng: không rẽ nhánh và khớp tuyệt đối.
5. Batch kế được chọn ngay lúc GPU xong. Một chunk chỉ chờ tối đa một bước đang chạy.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| [`src/nemotron_engine/`](src/nemotron_engine) | Package engine: những gì `pip install` cài cho bạn. |
| [`examples/`](examples) | Các chương trình nhỏ chạy được ngay. |
| [`research/nemotron_lite/`](research/nemotron_lite) | Bản port tham chiếu: offline + streaming, code đơn giản, đã kiểm với NeMo. Dùng cho test parity và thử nghiệm. |
| [`bench/`](bench) | Parity độ chính xác với NeMo, RTFx offline, load test streaming (trong process và qua WebSocket), profiler. |
| [`tests/`](tests) | Unit test CPU (`pytest`) và test parity/stress trên GPU (`python tests/test_engine.py`). |
| [`docs/`](docs) | Kiến trúc, benchmark, hành trình tối ưu (tiếng Anh và tiếng Việt). |

## Chạy lại benchmark

```bash
pip install -r research/requirements.txt          # engine + phụ thuộc của benchmark + NeMo cho các lần chạy tham chiếu
STAGES="test parity capacity ws cli" bash bench/run_all.sh
```

Muốn chạy trên GPU thuê, [`vkong.yaml`](vkong.yaml) là recipe dùng ngay: `vkong run --detach`.

## Hạn chế

- Một process dùng một GPU. Có nhiều GPU thì chạy nhiều process.
- Chỉ có greedy decoding; chưa có beam search, chưa có timestamp theo từ.
- Kích thước chunk cố định cho mỗi engine (`lookahead`). 320 ms là mặc định và là cấu hình đã kiểm.
- Qua server WebSocket, một event loop Python xử lý mọi frame. Khi số luồng rất cao và gói 20 ms, event loop thành nút thắt trước cả GPU. Dùng gói lớn hơn (60–100 ms) hoặc nhiều process front-end sẽ đỡ hơn.
- Server không có xác thực. Hãy đặt nó sau gateway của bạn.

## Ghi nhận và license

- Mô hình của NVIDIA, phát hành theo [OpenMDW-1.1](https://openmdw.ai/license/1-1/). Phần cài đặt theo sát [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/Speech) (Apache-2.0).
- Benchmark dùng bộ dữ liệu công khai [VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech).
- Code trong repo theo license [Apache-2.0](LICENSE). Xem [NOTICE](NOTICE).

Nếu bạn dùng engine trong công việc, xem [CITATION.cff](CITATION.cff).
