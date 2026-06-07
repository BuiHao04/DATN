from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from pipeline.core.gcn_classifier import InvoiceGCN, build_edge_index


class GCNTrainingService:
    """
    Expect training data as list of graph samples:
    [
      {
        "x": [[...], [...]],
        "edge_index": [[src...], [dst...]],
        "y": [label_id, ...]
      }
    ]
    """

    def _load_samples(self, dataset_json_path: str) -> list[dict]:
        data = json.loads(Path(dataset_json_path).read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        if not samples:
            raise ValueError(f"Empty dataset: missing samples in {dataset_json_path}")
        return samples

    def _infer_shape(self, samples: list[dict]) -> tuple[int, int]:
        in_channels = len(samples[0]["x"][0])
        out_channels = max(max(s["y"]) for s in samples) + 1
        return in_channels, out_channels

    def _load_init_checkpoint_compatible(
        self,
        model: InvoiceGCN,
        init_checkpoint: str,
        stage_name: str,
    ) -> None:
        state = torch.load(init_checkpoint, map_location="cpu")
        model_state = model.state_dict()

        filtered_state = {}
        skipped_keys: list[str] = []
        for key, value in state.items():
            if key not in model_state:
                skipped_keys.append(key)
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                skipped_keys.append(key)
                continue
            filtered_state[key] = value

        load_result = model.load_state_dict(filtered_state, strict=False)
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)

        if skipped_keys:
            logger.warning(
                "[{}] Skip {} incompatible checkpoint tensors: {}",
                stage_name,
                len(skipped_keys),
                ", ".join(skipped_keys),
            )
        if missing_keys:
            logger.warning(
                "[{}] Model parameters not initialized from checkpoint: {}",
                stage_name,
                ", ".join(missing_keys),
            )
        if unexpected_keys:
            logger.warning(
                "[{}] Unexpected checkpoint tensors ignored: {}",
                stage_name,
                ", ".join(unexpected_keys),
            )

    def _run_epoch_train(self, model: InvoiceGCN, samples: list[dict], optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        for s in samples:
            x = torch.tensor(s["x"], dtype=torch.float32)
            if "edge_index" in s:
                edge_index = torch.tensor(s["edge_index"], dtype=torch.long)
            else:
                edge_index = build_edge_index(s.get("edges", []))
            y = torch.tensor(s["y"], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        return total_loss / max(len(samples), 1)

    def _run_eval(self, model: InvoiceGCN, samples: list[dict], out_channels: int) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        conf = [[0 for _ in range(out_channels)] for _ in range(out_channels)]

        with torch.no_grad():
            for s in samples:
                x = torch.tensor(s["x"], dtype=torch.float32)
                if "edge_index" in s:
                    edge_index = torch.tensor(s["edge_index"], dtype=torch.long)
                else:
                    edge_index = build_edge_index(s.get("edges", []))
                y = torch.tensor(s["y"], dtype=torch.long)

                logits = model(x, edge_index)
                loss = F.cross_entropy(logits, y)
                total_loss += float(loss.item())

                pred = torch.argmax(logits, dim=1)
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    if 0 <= yi < out_channels and 0 <= pi < out_channels:
                        conf[yi][pi] += 1

        correct = sum(conf[i][i] for i in range(out_channels))
        total = sum(sum(r) for r in conf)
        acc = (correct / total) if total else 0.0

        macro_f1 = 0.0
        for c in range(out_channels):
            tp = conf[c][c]
            fp = sum(conf[r][c] for r in range(out_channels) if r != c)
            fn = sum(conf[c][k] for k in range(out_channels) if k != c)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            macro_f1 += f1
        macro_f1 /= max(out_channels, 1)

        return {
            "loss": total_loss / max(len(samples), 1),
            "accuracy": acc,
            "f1_macro": macro_f1,
        }

    def _train_once(
        self,
        dataset_json_path: str,
        checkpoint_path: str,
        epochs: int = 30,
        lr: float = 1e-3,
        init_checkpoint: str | None = None,
        stage_name: str = "GCN",
        val_dataset_json_path: str | None = None,
        early_stop_patience: int = 0,
    ) -> str:
        samples = self._load_samples(dataset_json_path)
        val_samples = self._load_samples(val_dataset_json_path) if val_dataset_json_path else None
        in_channels, out_channels = self._infer_shape(samples)

        model = InvoiceGCN(in_channels=in_channels, hidden_channels=64, out_channels=out_channels)
        if init_checkpoint:
            logger.info("[{}] Load init checkpoint: {}", stage_name, init_checkpoint)
            self._load_init_checkpoint_compatible(model, init_checkpoint, stage_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt.with_name(f"{ckpt.stem}.best{ckpt.suffix or '.pt'}")
        last_ckpt = ckpt.with_name(f"{ckpt.stem}.last{ckpt.suffix or '.pt'}")
        best_score = float("-inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch_train(model, samples, optimizer)
            torch.save(model.state_dict(), last_ckpt)

            if val_samples:
                val_metrics = self._run_eval(model, val_samples, out_channels)
                val_score = val_metrics["f1_macro"]
                improved = val_score > best_score
                if improved:
                    best_score = val_score
                    no_improve = 0
                    torch.save(model.state_dict(), best_ckpt)
                else:
                    no_improve += 1
                logger.info(
                    "[{}] Epoch {}/{} train_loss={:.6f} val_loss={:.6f} val_acc={:.4f} val_f1={:.4f} best_f1={:.4f}",
                    stage_name,
                    epoch,
                    epochs,
                    train_loss,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["f1_macro"],
                    best_score,
                )
                if early_stop_patience > 0 and no_improve >= early_stop_patience:
                    logger.info(
                        "[{}] Early stopping at epoch {} (no improve {} epochs)",
                        stage_name,
                        epoch,
                        no_improve,
                    )
                    break
            else:
                logger.info("[{}] Epoch {}/{} train_loss={:.6f}", stage_name, epoch, epochs, train_loss)

        if val_samples and best_score > float("-inf"):
            logger.info("[{}] Saved best checkpoint: {}", stage_name, best_ckpt)
            return str(best_ckpt)

        torch.save(model.state_dict(), ckpt)
        logger.info("[{}] Saved checkpoint: {}", stage_name, ckpt)
        return str(ckpt)

    # Stage A: pretrain/fine-tune on generic receipt/invoice dataset.
    def train_stage_a(
        self,
        dataset_json_path: str,
        checkpoint_path: str,
        epochs: int = 30,
        lr: float = 1e-3,
        init_checkpoint: str | None = None,
        val_dataset_json_path: str | None = None,
        early_stop_patience: int = 0,
    ) -> str:
        return self._train_once(
            dataset_json_path=dataset_json_path,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            lr=lr,
            init_checkpoint=init_checkpoint,
            stage_name="Stage A",
            val_dataset_json_path=val_dataset_json_path,
            early_stop_patience=early_stop_patience,
        )

    # Stage B: fine-tune on Vietnamese invoice dataset (main stage for final quality).
    def train_stage_b(
        self,
        dataset_json_path: str,
        checkpoint_path: str,
        base_checkpoint: str,
        epochs: int = 30,
        lr: float = 1e-3,
        val_dataset_json_path: str | None = None,
        early_stop_patience: int = 0,
    ) -> str:
        return self._train_once(
            dataset_json_path=dataset_json_path,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            lr=lr,
            init_checkpoint=base_checkpoint,
            stage_name="Stage B",
            val_dataset_json_path=val_dataset_json_path,
            early_stop_patience=early_stop_patience,
        )

    # Backward-compatible single training entry.
    def train(
        self,
        dataset_json_path: str,
        checkpoint_path: str,
        epochs: int = 30,
        lr: float = 1e-3,
        val_dataset_json_path: str | None = None,
        early_stop_patience: int = 0,
    ) -> str:
        return self._train_once(
            dataset_json_path=dataset_json_path,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            lr=lr,
            stage_name="GCN",
            val_dataset_json_path=val_dataset_json_path,
            early_stop_patience=early_stop_patience,
        )
