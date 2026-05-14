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


def cmd_train_ocr(args: argparse.Namespace) -> None:
    from pipeline.services.ocr_training_service import OCRTrainingService

    service = OCRTrainingService()
    service.train(command=args.command, workdir=args.workdir)
    logger.info("OCR training command done")


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

    to = sub.add_parser("train_ocr")
    to.add_argument("--command", required=True)
    to.add_argument("--workdir", default=None)
    to.set_defaults(func=cmd_train_ocr)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
