# Tái lập thực nghiệm W2VPCA về đề xuất LEZ tại TP.HCM

Gói này tái lập W2VPCA trên corpus gộp bài báo và bình luận, cùng hai phép đối chiếu cosine.

## Cài đặt

Yêu cầu Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy thực nghiệm

Chạy theo thứ tự:

```bash
python scripts/run_w2vpca_combined.py
python compare/run_cosine_baseline.py
python compare/run_procrustes_baseline.py
```

Kết quả W2VPCA được tạo tại `reanalysis_combined_timed/`; kết quả hai phép đối chiếu được tạo tại `compare/results/`.

Các lần chạy sử dụng năm hạt giống cố định 13, 27, 41, 59 và 83. Mô hình Word2Vec dùng CBOW, vector 100 chiều, `window=5`, `negative=5`, `sample=1e-3`, `epochs=20`, `min_count=1`, `workers=1` và `sorted_vocab=1`.

## Dữ liệu và mã nguồn

- `outputs/` chứa hai corpus đã tiền xử lý, là đầu vào của các thực nghiệm.
- `data/keyword_list.csv` chứa 42 từ khóa mục tiêu thuộc sáu nhóm chủ đề.
- `scripts/` chứa ba bước tiền xử lý và script W2VPCA.
- `compare/` chứa cosine baseline và Procrustes--cosine baseline.

Ba script tiền xử lý có thể được chạy tuần tự khi cần tạo lại corpus từ `articles.csv` và `comments.csv`; bước phân đoạn từ sử dụng VnCoreNLP thông qua `py-vncorenlp`.
