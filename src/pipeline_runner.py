from __future__ import annotations

import argparse
from loguru import logger


def cmd_gcn_infer(args: argparse.Namespace) -> None:
    from pipeline.services.gcn_pipeline_service import GCNPipelineService
    from pipeline.services.ocr_service import OCRService

    gcn_service = GCNPipelineService()
    ocr_service = OCRService()

    nodes = ocr_service.run(args.image, lang=args.lang, engine=args.ocr_engine)
    ocr_service.save_debug_image(args.image, nodes, args.ocr_debug_image)

    result = gcn_service.infer(nodes, checkpoint_path=args.checkpoint)
    result["image_path"] = args.image
    result["ocr_boxes_image"] = args.ocr_debug_image
    gcn_service.save_result(result, args.output_json)

    logger.info("Saved GCN infer JSON: {}", args.output_json)


def cmd_pretrained(args: argparse.Namespace) -> None:
    from pipeline.services.pretrained_service import PretrainedInferenceService
    from pipeline.services.ocr_service import OCRService

    model_id = PretrainedInferenceService.load_model_id_from_env(args.project_dir)
    service = PretrainedInferenceService(model_id=model_id)
    ocr_service = OCRService()

    nodes = ocr_service.run(args.image, lang=args.lang, engine=args.ocr_engine)
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
        val_dataset_json_path=args.val_dataset_json,
        early_stop_patience=args.early_stop_patience,
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
        val_dataset_json_path=args.val_dataset_json,
        early_stop_patience=args.early_stop_patience,
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
        val_dataset_json_path=args.val_dataset_json,
        early_stop_patience=args.early_stop_patience,
    )
    logger.info("GCN Stage B done: {}", out)


def cmd_train_ocr(args: argparse.Namespace) -> None:
    from pipeline.services.ocr_training_service import OCRTrainingService

    service = OCRTrainingService()
    service.train(command=args.command, workdir=args.workdir)
    logger.info("OCR training command done")


def cmd_test_gcn(args: argparse.Namespace) -> None:
    from pipeline.services.gcn_evaluation_service import GCNEvaluationService

    service = GCNEvaluationService()
    out = service.evaluate(
        dataset_json_path=args.dataset_json,
        checkpoint_path=args.checkpoint,
        output_eval_path=args.output_eval,
    )
    logger.info("GCN test/eval done: {}", out)


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


def cmd_convert_hf_cord_to_csv(args: argparse.Namespace) -> None:
    from pipeline.services.hf_cord_to_gcn_csv_service import HFCordToGcnCsvService

    service = HFCordToGcnCsvService()
    output_csv = args.output_csv or f"data/cord_{args.split}_nodes.csv"
    out = service.convert(
        dataset_id=args.dataset_id,
        split=args.split,
        output_csv_path=output_csv,
        limit=args.limit,
        streaming=bool(args.streaming),
    )
    logger.info("HF CORD -> CSV done: {}", out)


def cmd_convert_hf_to_gcn_csv(args: argparse.Namespace) -> None:
    from pipeline.services.hf_generic_to_gcn_csv_service import HFGenericToGcnCsvService

    service = HFGenericToGcnCsvService()
    output_csv = args.output_csv or f"data/{args.split}_nodes.csv"
    label_map = service.load_label_map(args.label_map)
    out = service.convert(
        dataset_id=args.dataset_id,
        split=args.split,
        output_csv_path=output_csv,
        doc_id_field=args.doc_id_field,
        text_field=args.text_field,
        label_field=args.label_field,
        bbox_field=args.bbox_field,
        score_field=args.score_field,
        label_map=label_map,
        limit=args.limit,
        streaming=bool(args.streaming),
    )
    logger.info("HF generic -> GCN CSV done: {}", out)


def cmd_prepare_ocr_labeling(args: argparse.Namespace) -> None:
    from pipeline.services.ocr_labeling_prep_service import OCRLabelingPrepService

    service = OCRLabelingPrepService()
    out = service.prepare(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        engine=args.ocr_engine,
        ocr_overrides={
            "det_db_thresh": args.det_db_thresh,
            "det_db_box_thresh": args.det_db_box_thresh,
            "det_db_unclip_ratio": args.det_db_unclip_ratio,
            "drop_score": args.drop_score,
            "use_dilation": args.use_dilation,
            "det_limit_side_len": args.det_limit_side_len,
            "upscale_factor": args.upscale_factor,
        },
        save_debug_images=bool(args.save_debug_images),
        copy_images=bool(args.copy_images),
        num_workers=args.num_workers,
        worker_index=args.worker_index,
        save_every_images=args.save_every_images,
    )
    logger.info("OCR labeling prep done: {}", out)


def cmd_train_gcn_full(args: argparse.Namespace) -> None:
    """One-command training flow:
    optional preprocess -> stage A train -> stage B train -> optional eval.
    """
    from pipeline.services.gcn_dataset_preprocess_service import GCNDatasetPreprocessService
    from pipeline.services.gcn_training_service import GCNTrainingService
    from pipeline.services.gcn_evaluation_service import GCNEvaluationService

    preprocess = GCNDatasetPreprocessService()
    trainer = GCNTrainingService()
    evaluator = GCNEvaluationService()

    stage_a_json = args.stage_a_json
    stage_b_json = args.stage_b_json

    # If CSV is provided, build JSON automatically.
    if args.stage_a_csv:
        stage_a_json = preprocess.preprocess_csv(
            input_csv_path=args.stage_a_csv,
            output_json_path=stage_a_json,
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

    if args.stage_b_csv:
        stage_b_json = preprocess.preprocess_csv(
            input_csv_path=args.stage_b_csv,
            output_json_path=stage_b_json,
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

    stage_a_ckpt = trainer.train_stage_a(
        dataset_json_path=stage_a_json,
        checkpoint_path=args.stage_a_ckpt,
        epochs=args.stage_a_epochs,
        lr=args.stage_a_lr,
        init_checkpoint=args.init_checkpoint,
    )

    stage_b_ckpt = trainer.train_stage_b(
        dataset_json_path=stage_b_json,
        checkpoint_path=args.stage_b_ckpt,
        base_checkpoint=stage_a_ckpt,
        epochs=args.stage_b_epochs,
        lr=args.stage_b_lr,
    )

    if args.eval_json:
        report_path = evaluator.evaluate(
            dataset_json_path=args.eval_json,
            checkpoint_path=stage_b_ckpt,
            output_eval_path=args.output_eval,
        )
        logger.info("GCN full flow done. Final ckpt: {} | Eval: {}", stage_b_ckpt, report_path)
    else:
        logger.info("GCN full flow done. Final ckpt: {}", stage_b_ckpt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Invoice OCR + GCN pipeline runner")
    sub = p.add_subparsers(dest="mode", required=True)

    gcn = sub.add_parser("gcn_infer")
    gcn.add_argument("--image", required=True)
    gcn.add_argument("--lang", default="vi")
    gcn.add_argument("--ocr-engine", default="paddle")
    gcn.add_argument("--checkpoint", default=None, help="Trained GCN checkpoint (.pt). If omitted, use rule-based fallback.")
    gcn.add_argument("--ocr-debug-image", default="outputs/ocr_boxes.jpg")
    gcn.add_argument("--output-json", default="outputs/ocr_result.json")
    gcn.set_defaults(func=cmd_gcn_infer)

    pre = sub.add_parser("pretrained")
    pre.add_argument("--project-dir", default=".")
    pre.add_argument("--image", required=True)
    pre.add_argument("--lang", default="vi")
    pre.add_argument("--ocr-engine", default="paddle")
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
    tg.add_argument("--val-dataset-json", default=None)
    tg.add_argument("--early-stop-patience", type=int, default=0)
    tg.set_defaults(func=cmd_train_gcn)

    # Train Stage A only (generic receipt/invoice dataset).
    tga = sub.add_parser("train_gcn_stage_a")
    tga.add_argument("--dataset-json", required=True)
    tga.add_argument("--checkpoint", default="outputs/checkpoints/gcn_stage_a.pt")
    tga.add_argument("--epochs", type=int, default=30)
    tga.add_argument("--lr", type=float, default=1e-3)
    tga.add_argument("--init-checkpoint", default=None)
    tga.add_argument("--val-dataset-json", default=None)
    tga.add_argument("--early-stop-patience", type=int, default=0)
    tga.set_defaults(func=cmd_train_gcn_stage_a)

    # Train Stage B only (Vietnamese invoice dataset), starting from Stage A checkpoint.
    tgb = sub.add_parser("train_gcn_stage_b")
    tgb.add_argument("--dataset-json", required=True)
    tgb.add_argument("--base-checkpoint", required=True)
    tgb.add_argument("--checkpoint", default="outputs/checkpoints/gcn_stage_b.pt")
    tgb.add_argument("--epochs", type=int, default=20)
    tgb.add_argument("--lr", type=float, default=5e-4)
    tgb.add_argument("--val-dataset-json", default=None)
    tgb.add_argument("--early-stop-patience", type=int, default=0)
    tgb.set_defaults(func=cmd_train_gcn_stage_b)

    to = sub.add_parser("train_ocr")
    to.add_argument("--command", required=True)
    to.add_argument("--workdir", default=None)
    to.set_defaults(func=cmd_train_ocr)

    te = sub.add_parser("test_gcn")
    te.add_argument("--dataset-json", required=True)
    te.add_argument("--checkpoint", required=True)
    te.add_argument("--output-eval", default="outputs/gcn_eval_report.json")
    te.set_defaults(func=cmd_test_gcn)

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

    ch = sub.add_parser("convert_hf_cord_to_csv")
    ch.add_argument("--dataset-id", default="naver-clova-ix/cord-v2")
    ch.add_argument("--split", default="train")
    ch.add_argument("--output-csv", default=None)
    ch.add_argument("--limit", type=int, default=None)
    ch.add_argument("--streaming", type=int, default=1, help="1=low RAM mode, 0=normal mode")
    ch.set_defaults(func=cmd_convert_hf_cord_to_csv)

    hg = sub.add_parser("convert_hf_to_gcn_csv")
    hg.add_argument("--dataset-id", required=True)
    hg.add_argument("--split", default="train")
    hg.add_argument("--output-csv", default=None)
    hg.add_argument("--doc-id-field", default="id")
    hg.add_argument("--text-field", default="text")
    hg.add_argument("--label-field", default="label")
    hg.add_argument("--bbox-field", default="bbox")
    hg.add_argument("--score-field", default=None)
    hg.add_argument(
        "--label-map",
        default=None,
        help="JSON string or path to JSON file, e.g. '{\"0\":\"OTHER\",\"1\":\"DATE\"}'",
    )
    hg.add_argument("--limit", type=int, default=None)
    hg.add_argument(
        "--streaming",
        type=int,
        default=1,
        help="1=streaming mode (low RAM, recommended), 0=normal mode",
    )
    hg.set_defaults(func=cmd_convert_hf_to_gcn_csv)

    po = sub.add_parser("prepare_ocr_labeling")
    po.add_argument("--input-dir", required=True, help="Folder containing raw invoice images")
    po.add_argument("--output-dir", default="data/labeling_stage_b")
    po.add_argument("--lang", default="vi")
    po.add_argument("--ocr-engine", default="paddle")
    po.add_argument("--det-db-thresh", type=float, default=0.25)
    po.add_argument("--det-db-box-thresh", type=float, default=0.58)
    po.add_argument("--det-db-unclip-ratio", type=float, default=1.25)
    po.add_argument("--drop-score", type=float, default=0.45)
    po.add_argument("--use-dilation", type=int, default=0)
    po.add_argument("--det-limit-side-len", type=int, default=1536)
    po.add_argument("--upscale-factor", type=float, default=1.6)
    po.add_argument("--save-debug-images", type=int, default=1)
    po.add_argument("--copy-images", type=int, default=1)
    po.add_argument("--num-workers", type=int, default=1, help="Total parallel workers")
    po.add_argument("--worker-index", type=int, default=0, help="This worker index [0..num_workers-1]")
    po.add_argument("--save-every-images", type=int, default=10, help="Flush CSV every N images")
    po.set_defaults(func=cmd_prepare_ocr_labeling)

    # Single command for full training flow (A -> B -> eval).
    tf = sub.add_parser("train_gcn_full")
    tf.add_argument("--stage-a-csv", default=None)
    tf.add_argument("--stage-b-csv", default=None)
    tf.add_argument("--stage-a-json", default="data/stage_a_dataset.json")
    tf.add_argument("--stage-b-json", default="data/stage_b_vi_dataset.json")
    tf.add_argument("--stage-a-ckpt", default="outputs/checkpoints/gcn_stage_a.pt")
    tf.add_argument("--stage-b-ckpt", default="outputs/checkpoints/gcn_stage_b.pt")
    tf.add_argument("--stage-a-epochs", type=int, default=30)
    tf.add_argument("--stage-b-epochs", type=int, default=20)
    tf.add_argument("--stage-a-lr", type=float, default=1e-3)
    tf.add_argument("--stage-b-lr", type=float, default=5e-4)
    tf.add_argument("--init-checkpoint", default=None)
    tf.add_argument("--eval-json", default=None)
    tf.add_argument("--output-eval", default="outputs/gcn_eval_report.json")
    # Preprocess options (used only when stage-a-csv or stage-b-csv is passed).
    tf.add_argument("--doc-id-col", default="doc_id")
    tf.add_argument("--text-col", default="text")
    tf.add_argument("--label-col", default="label")
    tf.add_argument("--score-col", default="score")
    tf.add_argument("--x1-col", default="x1")
    tf.add_argument("--y1-col", default="y1")
    tf.add_argument("--x2-col", default="x2")
    tf.add_argument("--y2-col", default="y2")
    tf.add_argument("--same-line-ratio", type=float, default=1.2)
    tf.add_argument("--near-threshold", type=float, default=250.0)
    tf.add_argument("--min-nodes-per-graph", type=int, default=1)
    tf.set_defaults(func=cmd_train_gcn_full)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
