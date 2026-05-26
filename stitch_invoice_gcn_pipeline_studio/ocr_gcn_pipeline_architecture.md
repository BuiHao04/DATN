# OCR & GCN Training Pipeline - Architecture & User Flow

## 1. Information Architecture (IA)

### Module A: Dashboard (Overview)
- KPI Cards: Total Images, OCR Success Rate, Labeling Progress, Datasets Created, Last Training Status.
- Recent Activity Feed.

### Module B: Data Ingestion
- Upload Zone (Drag-and-drop).
- Image Queue Table: Filename, Status (Pending/Processed/Failed), Size, Action.
- Batch Operations: "Run OCR Batch".

### Module C: OCR & Labeling Workspace (Unified Viewer)
- Image Viewer: Overlay Bounding Boxes (BBoxes).
- Data Table: Token list (Text, Score, Coordinates, Label).
- Annotation Tools: Dropdown labeling, Bulk actions, Shortcuts.
- Progress: Completion % per document.

### Module D: Dataset Builder
- Compilation: Merge labeled tokens into `nodes_to_label.csv`.
- Transformation: Export to GCN-ready JSON format.
- Statistics: Label distribution, Graph metrics.

### Module E: Training Center
- Configuration Form: Mode selection (Stage A/B/Full), Hyperparameters.
- Execution Monitor: Real-time logs, Loss/Accuracy charts.
- Artifact Management: Checkpoint download/save path.

### Module F: Evaluation & History
- Metric Visualizations: Confusion Matrix, F1-Score.
- Job History: Detailed logs of past runs.

---

## 2. User Flow (End-to-End)

1. **Ingestion**: User uploads receipt images -> Runs Batch OCR.
2. **Review/OCR**: User checks OCR results -> Fixes major errors if needed.
3. **Labeling**: Annotator assigns labels (DATE, TOTAL, etc.) to tokens -> Validates progress.
4. **Dataset Building**: ML Engineer compiles labeled data -> Generates JSON graphs.
5. **Training**: Engineer configures GCN model -> Monitors training progress in real-time.
6. **Evaluation**: System generates metrics -> Engineer exports report and best checkpoint.

---

## 3. Design System Strategy
- **Primary Color**: Deep Indigo/Blue (Professional, Trust).
- **Status Colors**: 
  - Success: Emerald Green.
  - Warning: Amber/Gold.
  - Error: Crimson Red.
  - Running: Electric Blue.
- **Typography**: Inter or Roboto (High readability for data tables).
- **Layout**: Sidebar navigation + Fluid main content area to accommodate large tables.