# DATN - Invoice OCR GCN Demo

Project trich xuat thong tin hoa don theo pipeline:

`OCR -> Graph -> Node Classification -> Invoice JSON`

## Thu muc chinh

- `invoice_ocr_gcn_demo/`: source code chinh.
- `invoice_ocr_gcn_demo/pipeline/core/`: logic core (ocr, graph, gcn, postprocess, visualize, schema).
- `invoice_ocr_gcn_demo/pipeline/services/`: service layer cho OCR, pretrained infer, GCN infer/train, eval.
- `invoice_ocr_gcn_demo/pipeline_runner.py`: entrypoint CLI cho cac mode.
- `invoice_ocr_gcn_demo/pretrained_invoice_baseline.py`: wrapper chay nhanh mode pretrained.
- `invoice_ocr_gcn_demo/test_ocr_invoice.py`: wrapper chay nhanh mode GCN infer.

## Cai dat

```powershell
cd <project_dir>\src
pip install -r requirements.txt
```

## Cau hinh `.env`

Tao file `.env` trong `invoice_ocr_gcn_demo`:

```env
MODEL_ID=nielsr/layoutlmv3-finetuned-funsd
```

Ban co the doi model khac bang cach sua `MODEL_ID`.

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
