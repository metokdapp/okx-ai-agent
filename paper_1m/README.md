# BTC AI Paper Agent — nến 1 phút

Bot mô phỏng BTC-USDT Spot với **1.000 USDT giả**, Gemini phân tích nến 1m đã đóng, có bối cảnh 5m/15m. Chỉ gọi API thị trường công khai OKX. Không có mã đặt lệnh thật, API key OKX, short hoặc đòn bẩy.

## Chạy trên Railway

- Dockerfile dùng Python 3.12, chỉ thư viện chuẩn, không cần pip.
- Chạy **một replica**, tắt sleep/serverless, không đặt cron.
- Gắn persistent volume tại `/data`, đặt `DATA_DIR=/data`. SQLite lưu tiền, vị thế, quyết định AI, lịch sử và hàng đợi Telegram. Không xóa volume khi redeploy.
- Nếu triển khai từ thư mục `paper_1m` trong repo: đặt Root Directory là `/paper_1m`, start command `python agent.py`, Dockerfile `Dockerfile`, healthcheck `/health` (cấu hình trực tiếp trong Railway).
- Healthcheck `/health` là kiểm tra tiến trình; `market_fresh` cho biết dữ liệu mới, không coi HTTP 200 là đã kết nối đầy đủ AI/Telegram.
- Nhập các biến bên dưới và deploy. Không commit key vào GitHub.

| Biến | Nội dung |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token bot riêng do bạn nhập |
| `GEMINI_API_KEY` | API key Gemini do bạn nhập |
| `GEMINI_MODELS` | `gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash`; danh sách dự phòng đã cấu hình sẵn |

Telegram chỉ cần `TELEGRAM_BOT_TOKEN`. Sau khi deploy, mở bot và nhắn riêng `/start`. **Tài khoản đầu tiên nhắn `/start` sẽ được liên kết**, vì vậy chính bạn phải nhắn trước; bot không thể xác minh đó là chủ token. Không cần Chat ID hay mã ghép nối. Bot lưu người nhận vào SQLite trên volume, giữ liên kết sau redeploy và không cho tài khoản khác ghi đè. Không chia sẻ tên bot trước khi liên kết. Dùng token riêng, không chạy bot Telegram khác với cùng token (polling sẽ xung đột). Nếu xóa DB/volume, việc liên kết đầu tiên sẽ bắt đầu lại.

Thiếu hoặc lỗi key AI: giữ HOLD, không dùng chỉ báo giả làm AI. Nhập key và redeploy để bắt đầu quyết định từ nến mới. Key Telegram không thay thế key AI. Chi phí Railway và API AI là chi phí thật dù vốn giao dịch là giả; một lần phân tích mỗi phút có thể đạt 1.440 lần/ngày. `AI_DAILY_LIMIT` giới hạn số lần gọi, không phải giới hạn chi tiêu tiền.

## Telegram

- Báo cáo mặc định mỗi 15 phút; thông báo mỗi lần mua/bán mô phỏng.
- `/status`, `/report`: vốn, giá, PnL, vị thế, quyết định AI, trạng thái dữ liệu.
- `/trades`: 5 lệnh đã đóng gần nhất.
- `/pause`: dừng quyết định AI; SL/TP vẫn hoạt động nếu có dữ liệu mới.
- `/resume`: phân tích từ nến 1m đóng tiếp theo.
- `/settings`: xem cấu hình; thay đổi qua Railway Variables và redeploy.
- `/whoami`: ID chat của bạn. `/help`: danh sách lệnh.

## Cấu hình mặc định

Tỷ lệ là số thập phân: `0.005` = 0,5%.

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| INITIAL_BALANCE | 1000 | Vốn giả lúc tạo DB lần đầu; không reset tài khoản cũ |
| RISK_PER_TRADE | 0.005 | Mục tiêu rủi ro 0,5% vốn tại SL, gồm phí và trượt giá dự kiến |
| MAX_ALLOCATION | 0.25 | Tối đa 25% tiền mặt mỗi vị thế |
| DAILY_LOSS_LIMIT | 0.03 | Giới hạn 3% vốn đầu ngày UTC; tính cả lỗ chưa chốt |
| FEE_RATE | 0.001 | Phí giả định 0,1% mỗi chiều, không phải phí xác minh của tài khoản OKX |
| SLIPPAGE_RATE | 0.0005 | Trượt giá bất lợi giả định 0,05% mỗi chiều |
| STOP_ATR_MULTIPLIER | 1.5 | Khoảng SL theo ATR(14), có khoảng tối thiểu theo chi phí |
| REWARD_RISK | 2 | Khoảng TP gấp 2 khoảng SL trước phí; không phải RR ròng |
| MIN_CONFIDENCE | 70 | Ngưỡng điểm chủ quan AI; không phải xác suất thắng |
| REPORT_SECONDS | 900 | Báo cáo định kỳ 15 phút |
| AI_DAILY_LIMIT | 1440 | Tối đa số lần gọi AI/ngày UTC, bao gồm lần lỗi |

Chỉ một vị thế. Sau thoát lệnh chờ ít nhất 60 giây. Chặn mua khi spread >0,2%, AI chưa đạt điểm, pause, chạm lỗ ngày hoặc dữ liệu cũ. Khi đạt lỗ ngày, đóng vị thế theo bid mới và chặn mua đến ngày UTC tiếp theo. Lỗ thực tế có thể vượt mục tiêu khi giá nhảy hoặc kết nối gián đoạn.

## Tính toán và độ tin cậy

Python tính EMA20/50, RSI14 Wilder, MACD(12,26,9), ATR14 Wilder và volume ratio. Chỉ nhận nến OKX `confirm=1`, kiểm tra trùng, khoảng trống và độ mới. Đánh dấu nến trong DB **trước** khi gọi AI, tránh lệnh lặp sau restart; nến bị lỗi có thể bị bỏ qua, không phát lại tín hiệu cũ.

AI chạy độc lập với vòng giá; mỗi model có timeout tối đa 12 giây, dùng chung hạn chót 50 giây sau nến đóng. Quyết định quá 50 giây sau thời điểm đóng nến bị loại. Giá phải mới trong 10 giây. Giá lấy tối đa khoảng mỗi giây khi API đáp ứng; đây không phải luồng tick đầy đủ. BUY khớp theo ask cộng trượt giá, SELL theo bid trừ trượt giá; tính phí hai chiều. SL kiểm tra bid, nếu gap thì khớp theo giá hiện tại, không giả vờ khớp đúng mức SL.

PnL đã chốt tính sau phí; equity và PnL tổng gồm vị thế đang mở theo giá bid và phí thoát dự kiến. Max drawdown đo trên equity quan sát được, không phải mọi tick. Không tái tạo chạm SL/TP trong thời gian bot mất mạng/tắt máy. Telegram có outbox bền vững, retry; nếu Telegram đã nhận tin nhưng kết nối mất trước ACK thì có thể trùng tin. Giữ tối đa 100 thông báo chờ để không tích tụ vô hạn.

Đây là thử nghiệm phần mềm, chưa chứng minh chiến lược có lợi nhuận. Không có backtest hiệu quả chiến lược trong bản này. Cần theo dõi mô phỏng và kiểm định độc lập. Các kiểm thử chỉ xác minh tính đúng của phần mềm.

## Chạy kiểm thử

```bash
python -m unittest discover -s tests -v
```

Chạy local: xuất biến môi trường cần thiết rồi `python agent.py`. `.env.example` là mẫu tham khảo; chương trình không tự đọc file `.env`.

## Nguồn API

- OKX: https://www.okx.com/docs-v5/en/ — market ticker/candles.
- Gemini: https://ai.google.dev/api/generate-content — JSON response schema.
- Telegram: https://core.telegram.org/bots/api — polling và sendMessage.
- Railway: https://docs.railway.com/volumes — lưu dữ liệu qua deploy.

## Gemini nhiều model

Một API key, thử tuần tự danh sách `GEMINI_MODELS`. Model trả kết quả thành công gần nhất được ưu tiên ở chu kỳ sau. HTTP 403/404: nghỉ model 6 giờ; lỗi yêu cầu 400: 1 giờ; lỗi tạm thời/JSON/timeout: 60 giây. Lỗi 429: chờ ít nhất 15 phút hoặc Retry-After/retryDelay nếu dài hơn, hạn mức ngày chờ bảo thủ 24 giờ. Chỉ chuyển model ngay khi quota error xác định rõ phạm vi riêng model; lỗi quota chung/không rõ phạm vi dừng cả danh sách. Key sai/hết hạn hoặc project bị khóa: nghỉ toàn bộ 1 giờ. Thời gian chờ được lưu qua restart.

Mỗi lần thử model, kể cả lỗi, đều tính vào `AI_DAILY_LIMIT` tổng chung. Không đổi API key/project để vượt quota. Hết model hoặc hết thời gian thì HOLD; SL/TP vẫn được vòng giá kiểm tra. Báo cáo Telegram hiển thị model thành công gần nhất, danh sách dự phòng và lỗi model gần nhất (lịch sử; có thể đã hồi phục).

Các model mặc định có Free Tier theo bảng giá Google khi cấu hình, nhưng miễn phí phụ thuộc tier của **project API**. Nếu project bật billing, cùng model có thể bị tính phí; bot không thể bảo đảm miễn phí hoặc tự xác định tier từ key. Dùng project Free Tier nếu chỉ muốn miễn phí. Hạn mức miễn phí có thể không đủ phân tích 1.440 nến/ngày; bot sẽ HOLD khi hết quota. Xem https://ai.google.dev/gemini-api/docs/pricing và https://ai.google.dev/gemini-api/docs/rate-limits.
