# Kiến trúc

*[English](architecture.md)*

nemotron-engine chạy [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) cho nhiều luồng audio trực tiếp cùng lúc trên một GPU. Trang này giải thích cách một process biến hàng trăm stream độc lập thành vài bước GPU lớn, và lý do của từng lựa chọn thiết kế.

## Mô hình trong một đoạn

Mô hình gồm encoder FastConformer cache-aware và decoder RNN-T (0,6 tỷ tham số):

- Audio được chuyển thành log-mel 128 bin, bước 10 ms.
- Bộ subsampling tích chập ×8 gộp 8 mel frame thành một frame encoder (80 ms).
- 24 lớp Conformer xử lý các frame. Mỗi lớp gồm FFN, self-attention vị trí tương đối, tích chập, FFN.
- Attention chia theo chunk và giới hạn ngữ cảnh: mỗi frame thấy 56 frame bên trái cộng các frame trong chunk của nó.
- Prompt ngôn ngữ (one-hot trên khoảng 100 locale, hoặc `auto`) được trộn vào đầu ra của encoder.
- Mạng dự đoán và joint của RNN-T phát ra các word piece.

Attention và tích chập chỉ nhìn sang trái (cộng phần bên trong chunk), nên encoder chạy được từng chunk một với cache gồm:

- 56 frame K/V cuối của mỗi lớp;
- 8 frame trạng thái conv cuối của mỗi lớp;
- phần đuôi của bộ subsampling.

Chunk mặc định là 320 ms (4 frame encoder, `lookahead=3`). Mô hình còn hỗ trợ 80, 560 và 1120 ms.

## Vòng lặp streaming đơn giản tốn thời gian ở đâu

Chạy từng chunk một, hoặc kể cả gom batch nhưng chạy PyTorch eager, thì GPU vẫn rảnh phần lớn thời gian:

| Vấn đề | Vì sao chậm |
|---|---|
| Khoảng 1.500 lần gọi kernel mỗi bước | 24 lớp × (LayerNorm, GEMM, các mảnh attention, conv, GLU, ...) đều gọi từ Python. Với chunk nhỏ, GPU chạy xong mỗi kernel trước khi Python kịp gọi kernel kế: bước bị giới hạn bởi host (CPU), không phải GPU. |
| Conv 1×1 qua cuDNN | Conv pointwise trong conv module chậm hơn khoảng 10 lần so với cùng phép tính viết dạng GEMM. |
| Greedy RNN-T chạy trên host | Mỗi token phát ra là một lần `.item()` / đồng bộ: hàng chục lượt GPU ↔ CPU mỗi bước. |
| Một process, nhiều stream, một GIL | Mỗi op PyTorch nhả rồi xin lại GIL. Nếu thread điều khiển GPU phải giành lại GIL từ event loop đang phục vụ hàng trăm socket trước mỗi op, GPU sẽ phải chờ Python. |

Engine xử lý từng vấn đề một.

## Thiết kế

```
 event loop asyncio (ứng dụng của bạn / server)        worker thread "nemotron-gpu" (giữ GPU)
 ──────────────────────────────────────────────        ───────────────────────────────────────────────
 open_stream / push / end_turn / close                  1. GPU đang chạy bước N: hỏi CUDA event,
   giải mã PCM16 / mu-law / f32, 8→16 kHz ──inbox──▶       xen kẽ drain inbox (chép mẫu, đánh dấu chunk
                                          (một deque)      đã đủ, ranh giới câu nói)
 detokenize, tính delta     ◀── call_soon_threadsafe ── 2. GPU xong: collect bước N, gửi partial / final
 asyncio.Queue → `async for ev in stream`                3. drain nốt, chọn batch N+1 (chunk chờ lâu nhất
                                                            trước, cùng phase, ≤ max_batch), chép cửa sổ
                                                            mẫu vào một buffer staging pinned
                                                         4. submit: chép H2D + MỘT lần replay graph; không chờ
```

### 1. Mỗi bước là một CUDA graph

Mỗi bucket kích thước batch có một CUDA graph bao trọn cả bước:

1. preemphasis và log-mel (`torch.stft` không center: buffer của stream cung cấp ngữ cảnh);
2. subsampling ×8, cache của nó được gather từ pool slot rồi scatter trả lại;
3. gather K/V, trạng thái conv, độ dài cache và one-hot ngôn ngữ của các slot trong batch;
4. 24 lớp Conformer;
5. projector của prompt;
6. scatter cache trở lại pool;
7. `greedy_steps` (12) bước greedy RNN-T;
8. chép nhãn sang bộ nhớ host pinned và record một CUDA event.

Host chỉ chép buffer staging lên GPU, replay một graph, rồi đọc nhãn: khoảng sáu lời gọi PyTorch mỗi bước thay vì khoảng 1.500. Nhờ vậy bước không còn bị giới hạn bởi việc gọi kernel. Phần lớn các lần trao GIL cũng biến mất, mà chính chúng từng khiến thời gian mỗi bước phụ thuộc vào độ bận của event loop.

Chunk đầu của stream (25 mel frame, chưa có cache; bucket 1, 2, 4, …, 64) và các chunk sau (32 mel frame; bucket 1, 2, 4, 8, 16, 24, …, 64) có graph riêng, cộng thêm một graph nối tiếp (xem bên dưới): tổng cộng 19 graph ở `max_batch=64`. Các graph dùng chung một memory pool và được capture từ bucket lớn xuống nhỏ, để graph nhỏ dùng lại block của graph lớn. Batch được đệm lên tới bucket gần nhất. Hàng đệm trỏ vào một slot nháp, nên phần ghi cache của chúng không bao giờ đụng tới stream thật.

### 2. Pool cache theo slot

Mỗi stream giữ một slot suốt vòng đời của nó. Mỗi slot trong pool chứa:

- K/V của 24 lớp × 56 frame;
- trạng thái conv;
- cache của subsampling;
- trạng thái LSTM và đầu ra của mạng dự đoán;
- one-hot của prompt ngôn ngữ.

Pool được cấp một lần lúc khởi động (`max_streams` slot). Bộ nhớ đứng yên dù stream mở và đóng liên tục: không có cấp phát nào trên đường nóng. Gather các slot của batch và scatter trả lại chỉ là một phép chép theo chỉ số, nằm trong graph.

### 3. Attention là một lần gọi SDPA

Attention vị trí tương đối cần `q·k` cộng `q·p` với một số hạng vị trí đã shift. Engine:

- tính Q, K, V bằng một GEMM;
- tính sẵn khoá vị trí đã shift, một lần cho mỗi shape chunk;
- cộng chúng làm bias cho một lần gọi `scaled_dot_product_attention` (kernel memory-efficient) trên khoá cache + chunk;
- cuộn cache K/V ngay trong bước scatter, không cần chép thêm.

Conv pointwise chạy bằng `F.linear`, không qua cuDNN.

### 4. Greedy RNN-T trong graph, với con trỏ frame cho mỗi stream

Greedy RNN-T phát 0 hoặc nhiều token cho mỗi frame encoder. Số token khác nhau giữa các stream, và đó là lý do thường phải có vòng lặp phía host. Engine thay vào đó cho mỗi stream một con trỏ frame riêng. Ở mỗi bước decode, mỗi stream:

- hoặc phát một token và đứng yên ở frame đó,
- hoặc sang frame kế (khi gặp blank, hoặc đã đủ 10 symbol trên một frame).

Một stream cần T + (số token của nó) bước. Với từng stream, chuỗi tính joint và mạng dự đoán giống hệt vòng lặp theo frame, nên **token giống hệt từng bit** so với bản tham chiếu.

Mười hai bước như vậy chạy trong graph của bước. Khi còn stream chưa hết frame, `collect` replay thêm một graph "nối tiếp" nhỏ gồm 4 bước. Với chunk 320 ms, việc này hiếm khi xảy ra.

Một phương án khác đã bị loại: giới hạn số symbol cố định mỗi frame trong graph thì đơn giản hơn, nhưng giọng nói thật thường xuyên phát ≥ 3 token trên một frame, nên gần như batch nào cũng vượt giới hạn.

### 5. Worker chọn batch ngay khi GPU rảnh

Vòng lặp của worker thread:

1. **Trong lúc GPU chạy một bước,** hỏi CUDA event của bước đó, xen kẽ drain inbox. Drain gồm chép mẫu vào buffer của từng stream, đánh dấu chunk đã đủ, và ghi nhận ranh giới câu nói.
2. **Khi GPU xong,** collect nhãn và gửi toàn bộ kết quả của vòng về event loop bằng một lần `call_soon_threadsafe`.
3. **Chọn batch kế** trong số các stream có chunk kế đã đủ. Chunk chờ lâu nhất được ưu tiên, và một batch chỉ gồm toàn chunk đầu hoặc toàn chunk sau, tối đa `max_batch`.
4. **Chép cửa sổ mẫu của từng stream** thẳng vào buffer staging pinned, rồi submit.

Chọn batch đúng lúc GPU rảnh, thay vì chuẩn bị sớm hơn, nghĩa là một chunk chỉ phải chờ tối đa một bước đang chạy. Mọi thứ chạy trên một CUDA stream, nên một stream có mặt ở hai batch liền nhau vẫn thấy các cập nhật cache theo đúng thứ tự.

### 6. Event loop làm càng ít càng tốt

- `push` giải mã gói (mu-law / PCM16 / float32, bộ upsample polyphase 8→16 kHz chạy theo luồng) rồi thêm vào inbox chung. Không có lock cho từng stream, không quét qua mọi stream.
- `wake.set()` chỉ được gọi khi worker đang ngủ.
- Detokenize chạy trên event loop, không chiếm thread GPU.
- `open_stream` lấy một slot từ danh sách slot trống (có lock). Nó báo `EngineFull` ngay khi hết slot, và `EngineOverloaded` khi chunk chờ GPU lâu nhất đã chờ quá `max_lag_ms` (mặc định 200 ms). Engine bão hoà sẽ từ chối stream mới thay vì để độ trễ của mọi stream cùng tăng.

### 7. Kết thúc câu nói mà không phải chờ audio

`end_turn(pad=True)` đệm số 0 cho đủ chunk đang dở, nên final ra ngay sau bước GPU kế tiếp thay vì chờ thêm tới 320 ms audio thật. Stream giữ nguyên cache; câu nói sau nối tiếp sau phần đệm.

Đánh đổi đã biết: token nào mô hình phát muộn, sau ranh giới (hay gặp nhất là dấu câu), sẽ rơi vào câu nói sau.

## Các file

| File | Nội dung |
|---|---|
| [`engine.py`](../src/nemotron_engine/engine.py) | API asyncio (`AsyncNemotronEngine`, `AsrStream`, `TranscriptEvent`), vòng lặp worker, slot, ranh giới câu nói |
| [`batch.py`](../src/nemotron_engine/batch.py) | `BatchStepper`: pool slot, capture graph, greedy theo con trỏ frame, `submit` / `ready` / `collect` |
| [`model.py`](../src/nemotron_engine/model.py) | Các module streaming của mô hình, log-mel kèm preemphasis, load trọng số và tokenizer |
| [`stream_audio.py`](../src/nemotron_engine/stream_audio.py) | Buffer mẫu thô của từng stream, hình học chunk, cửa sổ mẫu |
| [`audio.py`](../src/nemotron_engine/audio.py) | Giải mã PCM16 / mu-law / f32, upsample 8→16 kHz theo luồng |
| [`server.py`](../src/nemotron_engine/server.py) | Server WebSocket FastAPI và trang demo |
| [`cli.py`](../src/nemotron_engine/cli.py) | `nemotron-engine transcribe / serve / bench` |

## Một bước tốn thời gian ở đâu

Đo trên A100 bằng [`bench/profiling/graph_breakdown.py`](../bench/profiling/graph_breakdown.py). Mỗi phần được capture thành một CUDA graph riêng trên đầu vào tĩnh của một batch chunk sau, replay và đo bằng CUDA event, đặt cạnh một lần replay graph thật của cả bước.

| Phần, batch 64 | ms | Tỷ lệ trong tổng các phần |
|---|---|---|
| FFN 1 + FFN 2 (24 lớp) | 5,55 | 27% |
| Gather + scatter cache theo slot | 4,49 | 22% |
| Conv module (24 lớp) | 2,72 | 13% |
| Attention: SDPA (24 lớp) | 2,37 | 11% |
| Greedy RNN-T, 12 bước | 2,36 | 11% |
| Attention: QKV + ghi cache, bias rel-pos, chiếu đầu ra | 1,99 | 10% |
| Subsampling + cache của nó | 1,09 | 5% |
| Log-mel, LayerNorm cuối, projector của prompt | 0,30 | 1% |
| **Tổng các phần** | **20,88** | |
| **Graph cả bước** | **27,67** | |
| Cùng bước, PyTorch eager | 41,24 | |

| Batch | Graph cả bước | Eager | Một graph nối tiếp (4 bước greedy) |
|---|---|---|---|
| 8 | 9,35 ms | 41,46 ms | 0,77 ms |
| 32 | 22,92 ms | 42,21 ms | 0,68 ms |
| 64 | 27,67 ms | 41,24 ms | 0,77 ms |

Bảng này cho thấy:

- **PyTorch eager tốn khoảng 41 ms ở mọi kích thước batch.** Đó là giới hạn do gọi kernel: GPU chờ Python. Graph làm batch nhỏ rẻ hẳn (9 ms ở batch 8) và batch lớn nhanh hơn 1,5 lần.
- **Ở batch 64, graph cả bước tốn hơn tổng các phần khoảng 7 ms.** Các phần dùng chung băng thông bộ nhớ, và khi đo riêng thì chạy với cache đã nóng. Tổng các phần chỉ là chỉ dẫn thời gian đi đâu, không phải phép chia chính xác.
- **Gather và scatter cache chiếm khoảng một phần năm mỗi bước.** Attention đọc K/V thẳng từ pool slot sẽ bỏ được phần lớn chỗ này. Đây là hướng tối ưu rõ ràng nhất tiếp theo.
- **Greedy tốn gần như nhau ở batch 8 và 64 (khoảng 2,3 ms).** 12 bước tuần tự của nó bị giới hạn bởi độ trễ, không phải sức tính.

Khi tắt CUDA graph (`EngineConfig(cuda_graphs=False)`), cùng load test đó trượt SLO ngay ở 256 luồng trên A100 này: finalize p95 973 ms, batch kẹt ở 63,6, một core CPU chạy hết công suất. Có graph, 256 luồng chạy ở p95 30 ms.

## Độ chính xác

Engine được kiểm với [`research/nemotron_lite`](../research/nemotron_lite), một bản port PyTorch viết thẳng, bản thân nó đã được kiểm từng lớp với NeMo:

- mọi lớp khớp NeMo trong sai số tương đối 1e-6 khi tắt subsampling Triton gộp của NeMo;
- WER trên 500 câu bằng NeMo.

[`tests/test_engine.py`](../tests/test_engine.py) chạy 64 stream đồng thời qua engine ở strict fp32 và đòi **token giống hệt** bản tham chiếu. Kết quả vẫn giống hệt:

- khi dùng gói 100 ms thay vì 20 ms;
- khi đặt `greedy_steps=1`, tức chunk nào cũng phải qua graph nối tiếp.

Số liệu ở [benchmarks.vi.md](benchmarks.vi.md).
