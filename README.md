# DATN - Invoice OCR GCN Demo

Project trich xuat thong tin hoa don theo pipeline:

`OCR -> Graph -> Node Classification -> Invoice JSON`

## Thu muc chinh

- `src/`: source code chinh.
- `src/pipeline/core/`: logic core (ocr, graph, gcn, postprocess, visualize, schema).
- `src/pipeline/services/`: service layer cho OCR, pretrained infer, GCN infer/train, eval, data convert.
- `src/pipeline_runner.py`: entrypoint CLI cho tat ca mode.
- `src/pretrained_invoice_baseline.py`: wrapper chay nhanh mode pretrained.
- `src/test_ocr_invoice.py`: wrapper chay nhanh mode GCN infer.
- `src/render_invoice_text.py`: render ket qua JSON sang TXT/MD.
- `src/download_data.py`: download/convert dataset HuggingFace generic -> GCN CSV.

## Cai dat

```powershell
cd <project_dir>\src
pip install -r requirements.txt
```

## Chay Frontend + Backend

### 1) Cai dat dependencies

```powershell
cd C:\Users\PC\Documents\datn_hao\DATN
conda activate datn_hao
pip install -r app\requirements-web.txt
pip install -r src\requirements.txt
```

### 2) Build Frontend (React -> static files)

```powershell
cd C:\Users\PC\Documents\datn_hao\DATN\app\web
npm install
npm run build
```

Sau khi build, file se duoc tao tai:
- `app/frontend/dist/`

### 3) Chay Backend API (FastAPI + serve frontend)

```powershell
cd C:\Users\PC\Documents\datn_hao\DATN
conda activate datn_hao
uvicorn app.backend.main:app --reload --host 127.0.0.1 --port 8080
```

### 4) Truy cap

- Giao dien: `http://127.0.0.1:8080/frontend/`
- API health: `http://127.0.0.1:8080/api/health`

Luu y:
- Moi lan sua code trong `app/web/src`, can chay lai `npm run build`.
- Backend co `--reload` nen se tu restart khi sua file Python.

## Cau hinh `.env`

Tao file `.env` trong `src`:

```env
MODEL_ID=nielsr/layoutlmv3-finetuned-funsd
```

Ban co the doi model khac bang cach sua `MODEL_ID`.

## Huong dan theo tung file

### 1) `pipeline_runner.py` (file chinh)
Day la command hub. Moi tac vu nen uu tien chay tu file nay.

```powershell
python .\pipeline_runner.py -h
```

Mode chinh:
- `gcn_infer`: OCR + Graph + GCN infer.
- `pretrained`: baseline LayoutLMv3.
- `train_gcn_stage_a`, `train_gcn_stage_b`, `train_gcn_full`.
- `test_gcn`: danh gia checkpoint GCN.
- `evaluate`: danh gia file predict vs ground-truth.
- `preprocess_gcn_dataset`: CSV -> JSON cho train GCN.
- `convert_hf_to_gcn_csv`: HF dataset generic -> CSV.

### 2) `test_ocr_invoice.py` (wrapper nhanh cho infer)
Khi can test nhanh 1 anh voi luong GCN infer:

```powershell
python .\test_ocr_invoice.py
```

### 3) `pretrained_invoice_baseline.py` (wrapper nhanh pretrained)
Khi can chay baseline doc `.env`:

```powershell
python .\pretrained_invoice_baseline.py
```

### 4) `render_invoice_text.py` (xuat ket qua de doc)
Doc `outputs/ocr_result.json` va xuat:
- `outputs/invoice_final.txt`
- `outputs/invoice_final.md`

```powershell
python .\render_invoice_text.py
```

### 5) `download_data.py` (download dataset tong quat)
Dung cho nhieu dataset HuggingFace (chi can map dung field):

```powershell
python .\download_data.py --dataset-id naver-clova-ix/cord-v2 --split train
```

Neu field dat ten khac:

```powershell
python .\download_data.py --dataset-id your_org/your_dataset --split train --doc-id-field document_id --text-field token_text --label-field tag --bbox-field box --score-field confidence
```

Neu can map nhan:

```powershell
python .\download_data.py --dataset-id your_org/your_dataset --label-map .\data\label_map.json
```

### 6) `prepare_ocr_labeling` (tao du lieu de gan nhan Stage B)
Dat anh hoa don tho vao:
- `src/data/stage_b_raw_images/`

Chay OCR hang loat va tao file gan nhan:

```powershell
python .\pipeline_runner.py prepare_ocr_labeling --input-dir .\data\stage_b_raw_images --output-dir .\data\labeling_stage_b --lang en
```

Output de gan nhan:
- `data/labeling_stage_b/nodes_to_label.csv` (ban dien cot `label`)
- `data/labeling_stage_b/ocr_json/*.json` (chi tiet OCR moi anh)
- `data/labeling_stage_b/debug_boxes/*.jpg` (anh ve bbox de kiem tra)

## Cach chay

### Cach chay nhanh

```powershell
cd <project_dir>\src

# GCN infer
python .\pipeline_runner.py gcn_infer --image .\data\anh_test.jpg

# Pretrained baseline (doc .env)
python .\pipeline_runner.py pretrained --project-dir .

# Evaluate
python .\pipeline_runner.py evaluate --pred-json .\outputs\ocr_result.json --gt-json path\to\gt.json

# Train GCN Stage A (du lieu chung: receipt/invoice)
python .\pipeline_runner.py train_gcn_stage_a --dataset-json path\to\stage_a_dataset.json --checkpoint outputs\checkpoints\gcn_stage_a.pt

# Train GCN Stage B (du lieu hoa don tieng Viet, fine-tune tu Stage A)
python .\pipeline_runner.py train_gcn_stage_b --dataset-json path\to\stage_b_vi_dataset.json --base-checkpoint outputs\checkpoints\gcn_stage_a.pt --checkpoint outputs\checkpoints\gcn_stage_b.pt

# Train OCR command wrapper
python .\pipeline_runner.py train_ocr --command "python your_ocr_train_script.py"
```

### 1) Pretrained baseline (khong fine-tune)

```powershell
python .\pretrained_invoice_baseline.py
```

Hoac chay truc tiep:

```powershell
python .\pipeline_runner.py pretrained --project-dir . --image .\data\anh_test.jpg --lang en --ocr-debug-image outputs/ocr_boxes_pretrained.jpg --output-json outputs/pretrained_invoice_result.json
```

Output:
- `outputs/pretrained_invoice_result.json`
- `outputs/ocr_boxes_pretrained.jpg`

### 2) GCN infer

```powershell
python .\test_ocr_invoice.py
```

Hoac:

```powershell
python .\pipeline_runner.py gcn_infer --image .\data\anh_test.jpg --lang en --ocr-debug-image outputs/ocr_boxes.jpg --output-json outputs/ocr_result.json
```

### 3) Train GCN (2 stage)

```powershell
# Stage A: train tren du lieu chung (SROIE/CORD/du lieu receipt khac)
python .\pipeline_runner.py train_gcn_stage_a --dataset-json .\data\stage_a_dataset.json --checkpoint outputs/checkpoints/gcn_stage_a.pt --epochs 30 --lr 1e-3

# Stage B: fine-tune tren du lieu hoa don tieng Viet
python .\pipeline_runner.py train_gcn_stage_b --dataset-json .\data\stage_b_vi_dataset.json --base-checkpoint outputs/checkpoints/gcn_stage_a.pt --checkpoint outputs/checkpoints/gcn_stage_b.pt --epochs 20 --lr 5e-4
```

### 3.1) Download data de train Stage A (CORD -> preprocess -> train)

Windows PowerShell:

```powershell
cd <project_dir>\src

# 1) Download CORD split train -> CSV node-level
python .\download_cord_data.py --dataset-id naver-clova-ix/cord-v2 --split train

# 2) CSV -> JSON cho GCN Stage A
python .\pipeline_runner.py preprocess_gcn_dataset --input-csv .\data\cord_train_nodes.csv --output-json .\data\stage_a_dataset.json

# 3) Train Stage A
python .\pipeline_runner.py train_gcn_stage_a --dataset-json .\data\stage_a_dataset.json --checkpoint .\outputs\checkpoints\gcn_stage_a.pt --epochs 30 --lr 1e-3
```

Linux/WSL:

```bash
cd /path/to/DATN/src

# 1) Download du lieu CORD cho train / validation / test
python .\download_data.py --dataset-id naver-clova-ix/cord-v2 --split train
python .\download_data.py --dataset-id naver-clova-ix/cord-v2 --split validation
python .\download_data.py --dataset-id naver-clova-ix/cord-v2 --split test

# 2) CSV -> JSON cho GCN Stage A
python .\pipeline_runner.py preprocess_gcn_dataset --input-csv .\data\train_nodes.csv --output-json .\data\stage_a_train.json
python .\pipeline_runner.py preprocess_gcn_dataset --input-csv .\data\validation_nodes.csv --output-json .\data\stage_a_val.json
python .\pipeline_runner.py preprocess_gcn_dataset --input-csv .\data\test_nodes.csv --output-json .\data\stage_a_test.json

# 3) Train Stage A voi validation
python .\pipeline_runner.py train_gcn_stage_a --dataset-json .\data\stage_a_train.json --val-dataset-json .\data\stage_a_val.json --checkpoint .\outputs\checkpoints\gcn_stage_a.pt --epochs 30 --lr 1e-3 --early-stop-patience 5

# 4) Test checkpoint tren tap test
python .\pipeline_runner.py test_gcn --dataset-json .\data\stage_a_test.json --checkpoint .\outputs\checkpoints\gcn_stage_a.pt --output-eval .\outputs\gcn_stage_a_test_report.json
```

### 3.2) Chay checkpoint Stage A vua train (tap test + 1 anh rieng)

```bash
cd /path/to/DATN/src

# 1) Danh gia tren tap test (co ground-truth)
python pipeline_runner.py test_gcn \
  --dataset-json ./data/stage_a_test.json \
  --checkpoint ./outputs/checkpoints/gcn_stage_a.pt \
  --output-eval ./outputs/gcn_stage_a_test_report.json

# 2) Infer tren 1 anh rieng bang dung model Stage A da train
python pipeline_runner.py gcn_infer \
  --image ./data/test_1.jpg \
  --lang vi \
  --checkpoint ./outputs/checkpoints/gcn_stage_a.pt \
  --ocr-debug-image ./outputs/ocr_boxes_test_1.jpg \
  --output-json ./outputs/gcn_infer_stage_a_test_1.json
```

Khi infer bang checkpoint, file output se co:
- `classifier_mode: "trained_gcn_checkpoint"`
- `checkpoint_path: "<duong_dan_checkpoint>"`

### 4) Evaluate

```powershell
python .\pipeline_runner.py evaluate --pred-json outputs/pretrained_invoice_result.json --gt-json .\data\ground_truth.json --output-eval outputs/eval_report.json
```

### 5) OCR training (custom command)

```powershell
python .\pipeline_runner.py train_ocr --command "your_ocr_training_command_here" --workdir .
```

## Luu y quan trong

- Neu gap loi `shm.dll` (Torch tren Windows), thu tu import da duoc dieu chinh trong `pipeline_runner.py` (pretrained mode import torch-side truoc OCR-side).
- Neu bi chan mang HuggingFace (`WinError 10013`), service pretrained se tu fallback sang cache local (`local_files_only=True`).
- Model pretrained FUNSD chi la baseline, thuong khong map tot ra field hoa don VN neu chua fine-tune.

## Huong phat trien tiep

1. Fine-tune model token classification tren dataset hoa don VN.
2. Train/so sanh GCN voi baseline pretrained.
3. Bo sung metric field-level (precision/recall/F1) cho `date`, `tax_code`, `total_amount`, `seller_name`.
