# Engine nhanh lên như thế nào: hành trình tối ưu

*[English](optimization-journey.md)*

Trang này kể lại quá trình theo đúng thứ tự đã diễn ra: đo cái gì, thấy gì, sửa gì, và những ý tưởng không thành. Mục đích là làm tài liệu thực hành cho ai muốn phục vụ một mô hình streaming cho nhiều người dùng đồng thời bằng Python. Số đo hiện tại nằm ở [benchmarks.vi.md](benchmarks.vi.md), thiết kế cuối cùng ở [architecture.vi.md](architecture.vi.md).

## 0. Bắt đầu từ một bản tham chiếu đáng tin

Trước khi tối ưu, mô hình được port từ NeMo sang PyTorch thuần ([`research/nemotron_lite`](../research/nemotron_lite)):

- đọc thẳng `model.safetensors` của checkpoint;
- lúc chạy không cần NeMo, Lightning, Hydra hay transformers.

Bản port được kiểm ở ba mức:

| Mức | Cách kiểm | Kết quả |
|---|---|---|
| Từng lớp | So tensor trung gian của NeMo, dump trên một batch có padding, với tensor của bản port | khớp trong sai số 1e-6 |
| Transcript | 500 câu, chế độ offline | WER bằng NeMo, khoảng 99% câu giống hệt |
| Streaming | So kết quả streaming với offline | giống hệt 100% |

Chút sai khác nhỏ còn lại chỉ đến từ một chỗ: kernel subsampling Triton gộp của NeMo. Tắt kernel đó đi thì mọi lớp đều khớp.

**Bài học:** có bản tham chiếu khớp tuyệt đối thì mỗi lần tối ưu sau đó chỉ còn là câu hỏi có/không: "token còn giống hệt không?". Không có nó, mỗi lần tăng tốc lại kéo theo một cuộc tranh luận xem độ chính xác có bị ảnh hưởng không.

## 1. Đo trước: mỗi bước bị giới hạn bởi Python, không phải GPU

Bước đầu tiên hiển nhiên là gom các stream thành batch. Profile một bước streaming theo batch cho một kết quả bất ngờ: bước tốn thời gian như nhau ở batch 1 và batch 32. Nút thắt không phải GPU.

| Phát hiện | Chi tiết |
|---|---|
| Số lần gọi kernel | Khoảng 1.750 lần mỗi bước. Python gọi kernel cho 24 lớp còn lâu hơn GPU chạy chúng. |
| Kernel chậm nhất | Conv 1×1 ("pointwise") trong conv module của Conformer chiếm phần lớn thời gian GPU. Nó chạy qua cuDNN, chậm hơn khoảng 10 lần so với cùng phép tính viết dạng nhân ma trận. |
| Đồng bộ với host | Greedy decode đồng bộ với host gần như mỗi token (`int(k[i])` trong vòng lặp). |

**Bài học:** đo trước khi chọn chỗ tối ưu. Nhìn cả thời gian CPU lẫn thời gian kernel GPU của mỗi bước. Hai con số lệch nhau nhiều thì biết ngay nút thắt nằm ở phía nào.

## 2. Ba chỗ sửa trong code mô hình

1. **Conv pointwise → `F.linear`.** Vẫn là trọng số đó, chỉ reshape lại. Đây là chỗ lợi nhất phía GPU, chỉ với một dòng code.
2. **Pool cache theo slot, và một CUDA graph cho mỗi bucket batch.**
   - Mỗi stream có một slot cố định trong các tensor cache cấp sẵn. Một batch là gather → tính → scatter theo chỉ số slot.
   - Shape cố định theo bucket nên có thể capture encoder thành CUDA graph và replay bằng một lần gọi.
3. **Greedy không đồng bộ theo từng token.** Nhãn nằm trên GPU, host chỉ đồng bộ một lần mỗi vòng symbol thay vì mỗi token.

Token vẫn giống hệt. Lần đầu tiên, mỗi bước bị giới hạn bởi GPU.

## 3. Gộp attention

Ở bản tham chiếu, attention vị trí tương đối là vài phép nhân ma trận cộng một mẹo "shift", và mỗi bước lại nối K/V từ cache với chunk. Bản gộp:

- tính Q, K, V bằng một GEMM;
- tính sẵn khoá vị trí đã shift, nên phần vị trí thành một bias cộng vào attention;
- gọi `scaled_dot_product_attention` của PyTorch (backend memory-efficient) đúng một lần trên khoá cache + chunk;
- ghi K/V mới vào một buffer cache được cuộn ngay trong lúc scatter, không cần thêm lần chép nào.

Parity ở strict fp32: 0 token khác. Với matmul TF32, vài token khác nhau ở vài stream trong hàng nghìn token. Đó là nhiễu số học thường thấy của TF32, không phải lỗi.

## 4. Từ vòng lặp benchmark thành engine: API asyncio và một worker thread giữ GPU

Vòng lặp benchmark đồng bộ là trường hợp lý tưởng: nó chiếm trọn process. Ứng dụng thật thì có một event loop asyncio nhận audio từ hàng trăm kết nối. Vì vậy engine được dựng thành:

- một **API asyncio**: `open_stream`, `push`, `end_turn`, `async for event in stream`;
- một **worker thread** duy nhất giữ GPU.

Bản đầu chạy được, nhưng ở cùng mức tải thì chậm hơn vòng lặp benchmark. CPU ở mức khoảng 100% một core.

**Chẩn đoán: GIL.** Mỗi lời gọi PyTorch nhả GIL rồi phải xin lại. Khi event loop đang bận xử lý gói audio, worker phải chờ GIL trước mỗi op, và GPU chờ theo worker. Thời gian của worker được đo bằng wall time trừ CPU time của thread, theo từng pha. Phần chờ nằm đúng trên đường găng của GPU: lúc submit bước và trong vòng greedy.

## 5. Pipeline cho worker

1. **Một inbox chung** (deque) cho mọi lệnh từ event loop. Worker chỉ xử lý những stream có lệnh mới, không quét mọi stream mỗi vòng.
2. **Buffer staging pinned.** Lệnh chép host → device không còn chặn.
3. **Chuẩn bị batch N+1 trong lúc GPU chạy batch N.**
4. **Đẩy việc khỏi host.** Log-mel và preemphasis chuyển lên GPU, tính theo lô. Mỗi gói audio giờ chỉ là một lần chép vào buffer tuyến tính của stream, và cửa sổ chunk được chép thẳng vào staging.
5. **Chặn nhận stream khi quá tải.** `open_stream` báo `EngineOverloaded` khi chunk chờ lâu nhất đã chờ quá `max_lag_ms`. Engine quá tải sẽ từ chối stream mới, thay vì để độ trễ của mọi stream cùng tăng vọt.

Thông lượng tăng lên, nhưng greedy vẫn do host điều khiển: nhiều op nhỏ, mỗi op là một lần trao GIL.

## 6. Cả bước trong một CUDA graph

Thay đổi quyết định là đưa **toàn bộ** vào graph của bước: preemphasis, log-mel, subsampling, gather cache, 24 lớp, projector, scatter cache và greedy decode.

Phần khó là greedy RNN-T, vì luồng điều khiển phụ thuộc dữ liệu: số token mỗi frame khác nhau, và khác nhau giữa các stream. Lời giải là **mỗi stream có con trỏ frame riêng**. Ở mỗi bước, stream hoặc phát một token, hoặc sang frame kế. Mười hai bước như vậy, không rẽ nhánh, chạy trong graph. Khi còn stream chưa hết frame thì chạy thêm một graph nối tiếp nhỏ. Với từng stream, chuỗi tính toán giống hệt vòng lặp theo frame, nên token vẫn giống hệt.

Thêm một chi tiết: hàng đệm trong batch trỏ vào một **slot nháp**, nên phần ghi của chúng không đè lên stream thật.

Sau thay đổi này, mỗi bước host chỉ còn khoảng 6 lời gọi PyTorch cộng một lần replay graph. Thời gian chờ GIL trên đường găng gần như biến mất, và gói 20 ms hay 100 ms cho cùng tốc độ bước. Tranh chấp GIL không còn quyết định thông lượng.

## 7. Chọn batch ngay khi GPU xong

Pipeline ở mục 5 chọn batch N+1 *trong lúc* batch N đang chạy. Vì thế chunk nào sẵn sàng trong lúc đó phải chờ thêm trọn một bước.

Vòng lặp worker cuối cùng làm thế này:

1. Trong lúc GPU chạy, hỏi CUDA event của bước, xen kẽ drain inbox.
2. Khi event báo xong thì collect.
3. Sau đó mới chọn batch kế, chuẩn bị rồi submit.

Một chunk giờ chỉ phải chờ bước đang chạy. Chunk lag trung vị giảm rõ, và CPU của process giảm khoảng một nửa, vì worker không còn quay vòng trong `cudaStreamSynchronize`.

## Những gì không hiệu quả

| Ý tưởng | Vì sao bỏ |
|---|---|
| Greedy với số symbol cố định mỗi frame trong graph | Đơn giản, nhưng giọng nói thật trung bình có vài token mỗi chunk, nên gần như batch nào cũng có frame vượt giới hạn. Cách con trỏ frame khớp tuyệt đối mà gần như rẻ bằng. |
| Drain inbox trên event loop, thêm lock cho mỗi stream | Đo ra ngang với drain trên worker trong lúc GPU chạy, nhưng hai thread phải chia sẻ trạng thái có thể bị sửa. Bỏ cho đơn giản. |
| Một thread riêng để drain | Thêm một thread nữa tranh cùng một GIL, cho phần việc vốn đã nằm ngoài đường găng. |
| Greedy và encoder trên hai CUDA stream | Greedy chỉ còn khoảng 2 ms trong graph nên chẳng còn gì để chồng lên. Replay graph song song trên GPU dùng chung cũng là nguồn lỗi đã biết. |
| Gắn py-spy vào process đang chạy | Làm container bị giết. Bọc các method của worker từ bên ngoài (wall time so với CPU time của thread) cộng với CUDA event là đủ để tìm ra chỗ chờ GIL. |

## Hiện trạng và bước tiếp theo

**Khi gọi engine trực tiếp,** nút thắt là GPU. 24 lớp Conformer chiếm phần lớn mỗi bước; bảng chia thời gian nằm ở [architecture.vi.md](architecture.vi.md#một-bước-tốn-thời-gian-ở-đâu).

**Khi chạy qua server WebSocket,** nút thắt là event loop. Nó phải xử lý mọi frame WebSocket (50 frame mỗi giây cho mỗi stream với gói 20 ms) và tranh GIL với worker. Gói lớn hơn giúp đáng kể.

Hướng tiếp theo:

1. Tách phần nhận WebSocket ra process riêng, gửi audio đã gộp mỗi khoảng 10 ms sang process engine.
2. Bỏ gather/scatter cache: attention đọc K/V thẳng từ pool slot (kiểu paged attention).
3. Viết kernel attention riêng cho trường hợp ít query. Với 4 query mỗi stream, SDPA tốn lâu hơn nhiều so với thời gian đọc K/V.
4. Thử encoder bf16 (phải đo lại WER) và `max_batch=128`.
5. Tránh batch chunk đầu quá nhỏ, và chép sẵn cửa sổ của những chunk đã đủ.
