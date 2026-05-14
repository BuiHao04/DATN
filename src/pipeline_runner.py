from __future__ import annotations

import argparse
from loguru import logger


def cmd_gcn_infer(args: argparse.Namespace) -> None:
    from pipeline.services.ocr_service import OCRService
    from pipeline.services.gcn_pipeline_service import GCNPipelineService

    ocr_service = OCRService()
    gcn_service = GCNPipelineService()

    nodes = ocr_service.run(args.image, lang=args.lang)
    ocr_service.save_debug_image(args.image, nodes, args.ocr_debug_image)

    result = gcn_service.infer(nodes)
    gcn_service.save_result(result, args.output_json)

    logger.info("Saved GCN infer JSON: {}", args.output_json)


def cmd_pretrained(args: argparse.Namespace) -> None:
    from pipeline.services.pretrained_service import PretrainedInferenceService
    from pipeline.services.ocr_service import OCRService

    model_id = PretrainedInferenceService.load_model_id_from_env(args.project_dir)
    service = PretrainedInferenceService(model_id=model_id)
    ocr_service = OCRService()

    nodes = ocr_service.run(args.image, lang=args.lang)
    ocr_service.save_debug_image(args.image, nodes, args.ocr_debug_image)

    result = service.infer(args.image, nodes)
    result["ocr_boxes_image"] = args.ocr_debug_image
    service.save_result(result, args.output_json)

    logger.info("Saved pretrained baseline JSON: {}", args.output_json)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from pipeline.services.evaluation_service import EvaluationService

    service = EvaluationService()
    out = service.evaluate_from_files(args.pred_json, args.gt_json, args.output_eval)
    logger.info("Saved evaluation report: {}", out)


def cmd_train_gcn(args: argparse.Namespace) -> None:
    from pipeline.services.gcn_training_service import GCNTrainingService

    service = GCNTrainingService()
    out = service.train(
        dataset_json_path=args.dataset_json,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
    )
    logger.info("GCN training done: {}", out)


def cmd_train_gcn_stage_a(args: argparse.Namespace) -> None:
    from pipeline.services.gcn_training_service import GCNTrainingService

    service = GCNTrainingService()
    out = service.train_stage_a(
        dataset_json_path=args.dataset_json,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        init_checkpoint=args.init_checkpoint,
    )
    logger.info("GCN Stage A done: {}", out)


def cmd_train_gcn_stage_b(args: argparse.Namespace) -> None:
    from pipeline.services.gcn_training_service import GCNTrainingService

    service = GCNTrainingService()
    out = service.train_stage_b(
        dataset_json_path=args.dataset_json,
        checkpoint_path=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
    )
    logger.info("GCN Stage B done: {}", out)


def cmd_train_ocr(args: argparse.Namespace) -> None:
    from pipeline.services.ocr_training_service import OCRTrainingService

    service = OCRTrainingService()
    service.train(command=args.command, workdir=args.workdir)
    logger.info("OCR training command done")


def cmd_preprocess_gcn_dataset(args: argparse.Namespace) -> None:
    from pipeline.services.gcn_dataset_preprocess_service import GCNDatasetPreprocessService

    service = GCNDatasetPreprocessService()
    out = service.preprocess_csv(
        input_csv_path=args.input_csv,
        output_json_path=args.output_json,
        doc_id_col=args.doc_id_col,
        text_col=args.text_col,
        label_col=args.label_col,
        score_col=args.score_col,
        x1_col=args.x1_col,
        y1_col=args.y1_col,
        x2_col=args.x2_col,
        y2_col=args.y2_col,
        same_line_ratio=args.same_line_ratio,
        near_threshold=args.near_threshold,
        min_nodes_per_graph=args.min_nodes_per_graph,
    )
    logger.info("GCN dataset preprocess done: {}", out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Invoice OCR + GCN pipeline runner")
    sub = p.add_subparsers(dest="mode", required=True)

    gcn = sub.add_parser("gcn_infer")
    gcn.add_argument("--image", required=True)
    gcn.add_argument("--lang", default="en")
    gcn.add_argument("--ocr-debug-image", default="outputs/ocr_boxes.jpg")
    gcn.add_argument("--output-json", default="outputs/ocr_result.json")
    gcn.set_defaults(func=cmd_gcn_infer)

    pre = sub.add_parser("pretrained")
    pre.add_argument("--project-dir", default=".")
    pre.add_argument("--image", required=True)
    pre.add_argument("--lang", default="en")
    pre.add_argument("--ocr-debug-image", default="outputs/ocr_boxes_pretrained.jpg")
    pre.add_argument("--output-json", default="outputs/pretrained_invoice_result.json")
    pre.set_defaults(func=cmd_pretrained)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--pred-json", required=True)
    ev.add_argument("--gt-json", required=True)
    ev.add_argument("--output-eval", default="outputs/eval_report.json")
    ev.set_defaults(func=cmd_evaluate)

    tg = sub.add_parser("train_gcn")
    tg.add_argument("--dataset-json", required=True)
    tg.add_argument("--checkpoint", default="outputs/checkpoints/gcn_invoice.pt")
    tg.add_argument("--epochs", type=int, default=30)
    tg.add_argument("--lr", type=float, default=1e-3)
    tg.set_defaults(func=cmd_train_gcn)

    # Train Stage A only (generic receipt/invoice dataset).
    tga = sub.add_parser("train_gcn_stage_a")
    tga.add_argument("--dataset-json", required=True)
    tga.add_argument("--checkpoint", default="outputs/checkpoints/gcn_stage_a.pt")
    tga.add_argument("--epochs", type=int, default=30)
    tga.add_argument("--lr", type=float, default=1e-3)
    tga.add_argument("--init-checkpoint", default=None)
    tga.set_defaults(func=cmd_train_gcn_stage_a)

    # Train Stage B only (Vietnamese invoice dataset), starting from Stage A checkpoint.
    tgb = sub.add_parser("train_gcn_stage_b")
    tgb.add_argument("--dataset-json", required=True)
    tgb.add_argument("--base-checkpoint", required=True)
    tgb.add_argument("--checkpoint", default="outputs/checkpoints/gcn_stage_b.pt")
    tgb.add_argument("--epochs", type=int, default=20)
    tgb.add_argument("--lr", type=float, default=5e-4)
    tgb.set_defaults(func=cmd_train_gcn_stage_b)

    to = sub.add_parser("train_ocr")
    to.add_argument("--command", required=True)
    to.add_argument("--workdir", default=None)
    to.set_defaults(func=cmd_train_ocr)

    pg = sub.add_parser("preprocess_gcn_dataset")
    pg.add_argument("--input-csv", required=True)
    pg.add_argument("--output-json", required=True)
    pg.add_argument("--doc-id-col", default="doc_id")
    pg.add_argument("--text-col", default="text")
    pg.add_argument("--label-col", default="label")
    pg.add_argument("--score-col", default="score")
    pg.add_argument("--x1-col", default="x1")
    pg.add_argument("--y1-col", default="y1")
    pg.add_argument("--x2-col", default="x2")
    pg.add_argument("--y2-col", default="y2")
    pg.add_argument("--same-line-ratio", type=float, default=1.2)
    pg.add_argument("--near-threshold", type=float, default=250.0)
    pg.add_argument("--min-nodes-per-graph", type=int, default=1)
    pg.set_defaults(func=cmd_preprocess_gcn_dataset)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
