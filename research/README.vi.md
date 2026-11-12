# nemotron_lite: bản port tham chiếu

*[English](README.md)*

Bản port chỉ dùng cho suy luận, viết bằng PyTorch thuần, của `nvidia/nemotron-3.5-asr-streaming-0.6b`:

- FastConformer cache-aware + RNN-T, có prompt ngôn ngữ;
- chạy offline và streaming;
- đọc thẳng `model.safetensors` của checkpoint;
- chỉ cần `torch`, `safetensors`, `tokenizers`, `numpy` và `soundfile`, không cần NeMo.

Đây là phiên bản dễ đọc của mô hình, và là thước đo để kiểm engine:

| Kiểm tra | Kết quả |
|---|---|
| So từng lớp với NeMo | khớp trong sai số 1e-6 (khi tắt subsampling Triton gộp của NeMo) |
| Offline, 500 câu | WER bằng NeMo, khoảng 99% câu giống hệt |
| Streaming so với offline | giống hệt |

`nemotron_engine` (trong `src/`) là đường chạy production và không import package này.

```python
import sys; sys.path.insert(0, "research")
from nemotron_lite import NemotronLite

model = NemotronLite.from_pretrained(device="cuda")
print(model.transcribe(["a.wav"], language="vi-VN", lookahead=13))

sess = model.stream(language="vi-VN", lookahead=3)   # lookahead 0/3/6/13 -> chunk 80/320/560/1120 ms
for piece in mic_chunks():                            # các mảnh float32 16 kHz, kích thước tuỳ ý
    partial = sess.accept_audio(piece)
final = sess.finish()
```

| File | Nội dung |
|---|---|
| `features.py` | front-end log-mel, kèm bản tính dần cho streaming cho ra đúng các frame đó |
| `encoder.py` | subsampling nhân quả, attention vị trí tương đối giới hạn theo chunk, cache `stream_step` |
| `rnnt.py` | mạng dự đoán, joint, greedy decode theo batch |
| `batch_stream.py` | `BatchedStreamEngine`: gom batch nhiều luồng đồng bộ, dùng CUDA graph; tiền thân của engine |

Chạy parity với NeMo: `bash bench/parity/run_parity.sh` (xem `bench/README.vi.md`). File `requirements.txt` ở đây cài mọi thứ benchmark cần, kể cả NeMo.
