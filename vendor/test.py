"""
EfficientDet inference for pathology detection.

- ``--gt_mode mediastinal`` (default): chest-style CSV with ``label``; predictions
  summarized as Mediastinal Shift vs normal (legacy).
- ``--gt_mode spine_osteophytes`` (or ``Osteophytes`` / ``osteophytes`` / ``spine``): same validation CSV as RF-DETR spine eval
  (``image_path``, ``label``, ``findings`` under ``--data_dir``). Image-level
  GT positive if osteophyte-related findings are present; prediction positive if
  any detection survives ``--conf_threshold``. Bbox CSV is still written; mAP
  is not computed here when GT has no boxes. Annotated images are saved under
  ``<output_dir>/{tp,tn,fp,fn}/`` (image-level confusion vs GT).

Example (spine validation):

  python test.py --checkpoint runs/.../last.ckpt \\
    --input_csv /root/SPINE_PATHOLOGIES/VALIDATION/NEW_CSV/spine_validation_sample_250abnormal_250normal_not_in_osteo_unique.csv \\
    --data_dir /root/Data_utils/DATA_SPINE/ \\
    --gt_mode Osteophytes \\
    --output_dir effdet_spine_val_infer

Uses model and transforms from ``train.py`` in this directory.
"""
import os
import sys
import argparse
import ast
import json
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Import from train in same directory (Bronchi EFF_code)
from train import EfficientDetModel, get_valid_transforms


def _path_column(df: pd.DataFrame) -> str:
    for col in ("path", "image_path", "study_path"):
        if col in df.columns:
            return col
    raise ValueError("CSV must contain one of: path, image_path, study_path")


def parse_findings_cell(raw) -> set:
    """Parse spine CSV ``findings`` cell into a set of strings."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    s = str(raw).strip()
    if not s or s.lower() == "normal":
        return set()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1].replace('""', '"')
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (set, frozenset, list, tuple)):
            return {str(x).strip() for x in val if str(x).strip()}
    except (ValueError, SyntaxError, TypeError):
        pass
    tokens = set(re.findall(r'"([^"]+)"', s))
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1]
        for part in inner.split(","):
            part = part.strip()
            if not part or (part.startswith('"') and part.endswith('"')):
                continue
            tokens.add(part)
    return tokens


_OSTEOPHYTE_FINDINGS = (
    "Osteophytes",
    "Marginal Osteophytes",
    "Bridging Osteophytes",
    "Detached Osteophyte",
)


def findings_gt_positive_osteophytes(findings: set) -> bool:
    """True if report findings include any osteophyte type (exact list or name contains 'osteophyte')."""
    if any(name in findings for name in _OSTEOPHYTE_FINDINGS):
        return True
    return any("osteophyte" in f.lower() for f in findings)


def spine_row_gt_positive_osteophytes(row: pd.Series) -> bool:
    """True if this row is image-level GT-positive for osteophytes (report / label)."""
    label = str(row.get("label", "") or "").strip().lower()
    if label == "normal":
        return False
    findings = parse_findings_cell(row.get("findings", ""))
    return findings_gt_positive_osteophytes(findings)


def load_spine_osteophyte_gt(df: pd.DataFrame) -> Dict[str, bool]:
    """Map basename -> True (GT osteophytes) / False."""
    path_col = _path_column(df)
    out: Dict[str, bool] = {}
    for _, row in df.iterrows():
        p = row[path_col]
        if pd.isna(p):
            continue
        out[os.path.basename(str(p).strip())] = spine_row_gt_positive_osteophytes(row)
    return out


def extract_single_prediction(outputs, idx: int, original_size: Tuple[int, int], model_img_dim: int):
    try:
        if isinstance(outputs, dict) and "detections" in outputs:
            detections = outputs["detections"]
            detection = (
                detections[idx].detach().cpu().numpy()
                if torch.is_tensor(detections[idx])
                else detections[idx]
            )
            if detection.shape[0] > 0 and detection.shape[1] >= 6:
                boxes = detection[:, :4].copy()
                scores = detection[:, 4].copy()
                labels = detection[:, 5].copy()
                orig_w, orig_h = original_size
                scale_x = orig_w / model_img_dim
                scale_y = orig_h / model_img_dim
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                return {
                    "boxes": boxes.astype(np.float32),
                    "scores": scores.astype(np.float32),
                    "labels": labels.astype(np.float32),
                }
    except Exception as e:
        print(f"[WARN] Failed to extract predictions for idx={idx}: {e}")
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros(0, dtype=np.float32),
        "labels": np.zeros(0, dtype=np.float32),
    }


@torch.no_grad()
def predict_batch(model, batch_imgs: torch.Tensor, original_sizes: List[Tuple[int, int]], conf_threshold: float):
    device = next(model.parameters()).device
    dummy_targets = {
        "bbox": [torch.zeros((1, 4), dtype=torch.float32, device=device)] * batch_imgs.shape[0],
        "cls": [torch.zeros(1, dtype=torch.long, device=device)] * batch_imgs.shape[0],
        "img_size": torch.stack([
            torch.tensor([model.image_dim, model.image_dim], dtype=torch.float32, device=device)
            for _ in range(batch_imgs.shape[0])
        ]),
        "img_scale": torch.stack([
            torch.tensor(1.0, dtype=torch.float32, device=device)
        ] * batch_imgs.shape[0]),
    }
    outputs = model(batch_imgs.to(device), dummy_targets)
    batch_preds = []
    for idx, orig_size in enumerate(original_sizes):
        preds = extract_single_prediction(outputs, idx, orig_size, model.image_dim)
        keep = preds["scores"] >= conf_threshold
        batch_preds.append({
            "boxes": preds["boxes"][keep],
            "scores": preds["scores"][keep],
            "labels": preds["labels"][keep],
        })
    return batch_preds


def draw_boxes(image_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, conf_threshold: float = 0.3):
    for box, score in zip(boxes, scores):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr, f"{score:.2f}", (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return image_bgr


def normalize_gt_mode(value: str) -> str:
    """Map CLI synonyms to internal mode names."""
    key = str(value).strip().lower().replace(" ", "_")
    if key in ("mediastinal", "chest"):
        return "mediastinal"
    if key in ("spine_osteophytes", "osteophytes", "spine"):
        return "spine_osteophytes"
    raise argparse.ArgumentTypeError(
        f"invalid gt_mode {value!r}; use mediastinal, spine_osteophytes, or Osteophytes (alias for spine CSV eval)"
    )


def parse_args():
    p = argparse.ArgumentParser(description="EfficientDet pathology inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt checkpoint")
    p.add_argument("--input_dir", type=str, default=None, help="Directory of images (recursively searched)")
    p.add_argument("--input_csv", type=str, default=None, help="CSV with path/image_path/study_path column")
    p.add_argument("--data_dir", type=str, default="/root/Data_utils/DATA_CHEST", help="Root dir for images when using --input_csv")
    p.add_argument("--gt_csv", type=str, default=None, help="Ground truth CSV (path/image_path, label); optional if --gt_mode spine_osteophytes")
    p.add_argument(
        "--gt_mode",
        type=normalize_gt_mode,
        default="mediastinal",
        help="mediastinal (or chest): GT from label column. spine_osteophytes, Osteophytes, osteophytes, or spine: "
        "GT from label+findings on --input_csv (spine osteophyte validation CSV).",
    )
    p.add_argument("--output_dir", type=str, default="infer_media_d5", help="Output directory")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--conf_threshold", type=float, default=0.3)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="auto: use CUDA if it initializes; cuda: require GPU; cpu: force CPU (e.g. driver/PyTorch CUDA mismatch)",
    )
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested (--device cuda) but torch.cuda.is_available() is False")
        try:
            torch.tensor([0.0], device="cuda")
        except RuntimeError as e:
            raise RuntimeError("CUDA requested but failed to initialize (driver / PyTorch CUDA mismatch?)") from e
        return torch.device("cuda")
    # auto
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.tensor([0.0], device="cuda")
        return torch.device("cuda")
    except RuntimeError:
        print("[WARN] CUDA looked available but init failed (often old driver vs new torch+cu); using CPU.", file=sys.stderr)
        return torch.device("cpu")


def gather_image_paths(args) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    if args.input_csv:
        if not args.data_dir:
            raise ValueError("--data_dir required when using --input_csv")
        df = pd.read_csv(args.input_csv)
        path_col = _path_column(df)
        paths = []
        for p in df[path_col].tolist():
            p = str(p).strip()
            if os.path.isabs(p):
                resolved = p
            else:
                candidates = [
                    os.path.join(args.data_dir, p.lstrip("/")),
                    os.path.join(args.data_dir, os.path.basename(p)),
                    os.path.join(args.data_dir, p.replace("/", "_")),
                ]
                found = next((c for c in candidates if os.path.isfile(c)), None)
                resolved = found if found else candidates[0]
            if os.path.isfile(resolved):
                paths.append(resolved)
            else:
                print(f"[WARN] Image not found, skip: {resolved}")
        return paths
    if args.input_dir:
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(args.input_dir)
            for f in files if f.lower().endswith(exts)
        ]
    raise ValueError("Provide --input_csv + --data_dir or --input_dir")


def load_gt_labels(gt_csv_path: str) -> dict:
    if not gt_csv_path or not os.path.exists(gt_csv_path):
        return {}
    df = pd.read_csv(gt_csv_path)
    path_col = _path_column(df)
    if "label" not in df.columns:
        return {}
    gt_dict = {}
    for _, row in df.iterrows():
        img_path = row[path_col]
        label = str(row["label"]).strip().lower()
        if label in ("Mediastinal Shift", "abnormal"):
            label = "Mediastinal Shift"
        elif label == "normal":
            label = "normal"
        else:
            label = label  # keep as-is for unknown
        gt_dict[os.path.basename(str(img_path))] = label
    return gt_dict


def confusion_bucket(gt_mode: str, gt_label: str, pred_label: str) -> str:
    """Image-level bucket: tp, tn, fp, or fn."""
    gt = str(gt_label).lower()
    pred = str(pred_label).lower()
    if gt_mode == "spine_osteophytes":
        gt_pos = gt == "positive"
        pred_pos = pred == "positive"
    else:
        gt_pos = gt in ("mediastinal shift", "abnormal")
        pred_pos = pred in ("mediastinal shift", "abnormal")
    if gt_pos and pred_pos:
        return "tp"
    if gt_pos and not pred_pos:
        return "fn"
    if not gt_pos and pred_pos:
        return "fp"
    return "tn"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("tp", "tn", "fp", "fn"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"gt_mode={args.gt_mode}")

    print("Loading model…")
    # map_location avoids restoring GPU checkpoints to CUDA during unpickle (breaks with CPU-only or bad drivers)
    model = EfficientDetModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    transforms = get_valid_transforms(model.image_dim)

    image_paths = gather_image_paths(args)
    print(f"Found {len(image_paths)} images for inference")

    gt_labels = {}
    spine_gt: Dict[str, bool] = {}
    if args.gt_mode == "spine_osteophytes":
        if not args.input_csv:
            raise ValueError("spine_osteophytes requires --input_csv with label and findings columns")
        df_gt = pd.read_csv(args.input_csv)
        if "label" not in df_gt.columns:
            raise ValueError("spine_osteophytes CSV must include a 'label' column")
        if "findings" not in df_gt.columns:
            raise ValueError("spine_osteophytes CSV must include a 'findings' column")
        spine_gt = load_spine_osteophyte_gt(df_gt)
        print(f"Loaded spine osteophyte image-level GT for {len(spine_gt)} rows from {args.input_csv}")
    else:
        gt_labels = load_gt_labels(args.gt_csv) if args.gt_csv else {}
        if gt_labels:
            print(f"Loaded {len(gt_labels)} ground truth labels from {args.gt_csv}")

    csv_records = []
    class_records = []
    batch_imgs, batch_orig_sizes, batch_file_paths = [], [], []

    def flush_batch():
        nonlocal batch_imgs, batch_orig_sizes, batch_file_paths
        if not batch_imgs:
            return
        batch_tensor = torch.stack(batch_imgs)
        all_preds = predict_batch(model, batch_tensor, batch_orig_sizes, conf_threshold=0.0)
        for img_path, all_pred in zip(batch_file_paths, all_preds):
            img_basename = os.path.basename(img_path)
            if args.gt_mode == "spine_osteophytes":
                gt_pos = spine_gt.get(img_basename, False)
                gt_label = "positive" if gt_pos else "negative"
            elif gt_labels and img_basename in gt_labels:
                gt_label = gt_labels[img_basename]
            else:
                gt_label = os.path.basename(os.path.dirname(img_path)).lower()
                if gt_label in ("Mediastinal Shift", "abnormal"):
                    gt_label = "Mediastinal Shift"
                elif gt_label in ("normal", "neg", "negative"):
                    gt_label = "normal"
                else:
                    gt_label = "unknown"

            max_conf = float(np.max(all_pred["scores"])) if all_pred["scores"].size else 0.0
            keep = all_pred["scores"] >= args.conf_threshold
            filtered_pred = {
                "boxes": all_pred["boxes"][keep],
                "scores": all_pred["scores"][keep],
                "labels": all_pred["labels"][keep],
            }
            if args.gt_mode == "spine_osteophytes":
                pred_label = "positive" if filtered_pred["boxes"].shape[0] > 0 else "negative"
            else:
                pred_label = "Mediastinal Shift" if filtered_pred["boxes"].shape[0] > 0 else "normal"

            bucket = confusion_bucket(args.gt_mode, gt_label, pred_label)

            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                draw_boxes(img_bgr, filtered_pred["boxes"], filtered_pred["scores"], args.conf_threshold)
                cv2.imwrite(os.path.join(args.output_dir, bucket, img_basename), img_bgr)

            for box, score in zip(filtered_pred["boxes"], filtered_pred["scores"]):
                x1, y1, x2, y2 = box.tolist()
                csv_records.append({
                    "image": img_basename, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(score),
                })
            class_records.append({
                "image": img_basename,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "max_confidence": max_conf,
                "bucket": bucket,
            })
        batch_imgs, batch_orig_sizes, batch_file_paths = [], [], []

    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {img_path}: {e}")
            continue
        orig_size = img.size
        img_np = np.array(img)
        transformed = transforms(image=img_np, bboxes=[], labels=[])
        batch_imgs.append(transformed["image"])
        batch_orig_sizes.append(orig_size)
        batch_file_paths.append(img_path)
        if len(batch_imgs) == args.batch_size:
            flush_batch()
    flush_batch()

    if csv_records:
        pd.DataFrame(csv_records).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    if class_records:
        pd.DataFrame(class_records).to_csv(os.path.join(args.output_dir, "classification.csv"), index=False)

    tp = fp = tn = fn = 0
    for rec in class_records:
        b = rec.get("bucket") or confusion_bucket(args.gt_mode, rec["gt_label"], rec["pred_label"])
        if b == "tp":
            tp += 1
        elif b == "fn":
            fn += 1
        elif b == "fp":
            fp += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    metrics = {
        "gt_mode": args.gt_mode,
        "checkpoint": args.checkpoint,
        "input_csv": args.input_csv,
        "data_dir": args.data_dir,
        "total_images": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total if total else 0,
        "precision": tp / (tp + fp) if (tp + fp) else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "f1_score": 0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "confidence_threshold": args.conf_threshold,
        "image_output_layout": "tp, tn, fp, fn subfolders under output_dir",
    }
    if metrics["precision"] + metrics["recall"]:
        metrics["f1_score"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics:", metrics)
    print("Inference completed!")


if __name__ == "__main__":
    main()
