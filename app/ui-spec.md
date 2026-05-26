# UI/Backend Plan and Contract

## Screens
- dashboard.html
- data_ingestion.html
- ocr_preview.html
- labeling_workspace.html
- dataset_builder.html
- training_center.html
- evaluation.html
- run_history.html

## API
- GET /api/health
- GET /api/features
- GET /api/jobs
- GET /api/jobs/{id}
- POST /api/jobs

## Job request schema
```json
{
  "mode": "prepare_ocr_labeling",
  "args": {
    "input_dir": "src/data/stage_b_raw_images",
    "output_dir": "src/data/labeling_stage_b",
    "lang": "en"
  }
}
```

## Mode mapping
- prepare_ocr_labeling: OCR batch + nodes_to_label.csv + ocr_json
- preprocess_gcn_dataset: CSV -> JSON
- train_gcn_stage_a/stage_b/train_gcn_full: training
- gcn_infer/pretrained: inference
- evaluate/test_gcn: evaluation

## End-to-end flow
1. Data Ingestion: submit `prepare_ocr_labeling`
2. Labeling Workspace: edit `nodes_to_label.csv`
3. Dataset Builder: submit `preprocess_gcn_dataset`
4. Training Center: submit `train_gcn_stage_a` and `train_gcn_stage_b` or `train_gcn_full`
5. Evaluation: submit `evaluate` or `test_gcn`
6. Run History: monitor all jobs
