#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Dell AmericanStories layout ONNX over a manifest (layout only)."
    )
    p.add_argument("--manifest", required=True, help="txt file with absolute image path per line")
    p.add_argument("--model", required=True, help="path to layout_model_new.onnx")
    p.add_argument("--label_map", required=True, help="path to label_map_layout.json")
    # Back-compat: older variants of this script imported helper funcs from an external repo.
    # This pipeline vendors the required preprocessing + NMS so repo_src is optional.
    p.add_argument(
        "--repo_src",
        default="",
        help="(optional) path to external repo src directory (no longer required)",
    )
    p.add_argument("--output_root", required=True)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--iou", type=float, default=0.10)
    p.add_argument("--max_det", type=int, default=2000)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--provider", choices=["auto", "cpu"], default="cpu")
    p.add_argument("--max_pages", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip pages with existing <slug>_dell_layout_boxes.json (and regenerate overlay if missing).",
    )
    return p.parse_args()


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 4) [cx, cy, w, h]
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # box1: (N, 4), box2: (M, 4) in xyxy
    # returns: (N, M)
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device)

    a1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    a2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = a1[:, None] + a2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    # Pure torch NMS to avoid requiring torchvision on Torch.
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    idxs = scores.argsort(descending=True)
    keep: list[int] = []
    while idxs.numel() > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = _box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_thres]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    agnostic: bool,
    nc: int,
) -> list[torch.Tensor]:
    """YOLO-style NMS.

    Accepts either:
    - (bs, n, 6) in xyxy + conf + cls, OR
    - (bs, n, 5 + nc) in xywh + obj + cls_probs.
    Some ONNX exports are (bs, 5+nc, n); we transpose when detected.
    """

    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim != 3:
        raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")

    # If shape is (bs, no, n) instead of (bs, n, no), transpose.
    if prediction.shape[1] < prediction.shape[2] and prediction.shape[1] in (6, 5 + nc):
        prediction = prediction.transpose(1, 2)

    bs, n, no = prediction.shape
    out: list[torch.Tensor] = []

    for bi in range(bs):
        x = prediction[bi]
        if no == 6:
            # xyxy + conf + cls
            boxes = x[:, 0:4]
            scores = x[:, 4]
            cls = x[:, 5].to(torch.float32)
            m = scores >= conf_thres
            boxes, scores, cls = boxes[m], scores[m], cls[m]
            if boxes.numel() == 0:
                out.append(torch.empty((0, 6), device=x.device))
                continue
        else:
            if no < 5:
                raise ValueError(f"Unexpected prediction width: {no}")
            obj = x[:, 4:5]
            cls_probs = x[:, 5:]
            if cls_probs.shape[1] != nc:
                # If the model exports a different class count, clamp to known label map.
                cls_probs = cls_probs[:, :nc]
            conf = obj * cls_probs  # (n, nc)
            scores, cls_i = conf.max(dim=1)
            m = scores >= conf_thres
            if m.sum() == 0:
                out.append(torch.empty((0, 6), device=x.device))
                continue
            boxes = _xywh2xyxy(x[:, 0:4])[m]
            scores = scores[m]
            cls = cls_i[m].to(torch.float32)

        if boxes.shape[0] > 50000:
            boxes = boxes[:50000]
            scores = scores[:50000]
            cls = cls[:50000]

        if not agnostic:
            # class-aware NMS via coordinate offset.
            max_wh = 4096.0
            boxes_for_nms = boxes + cls[:, None] * max_wh
        else:
            boxes_for_nms = boxes

        keep = _nms(boxes_for_nms, scores, iou_thres)
        keep = keep[:max_det]
        det = torch.cat([boxes[keep], scores[keep, None], cls[keep, None]], dim=1)
        out.append(det)

    return out


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize + pad image to fit `new_shape` (h,w), returning (img, ratio, (dw,dh))."""
    shape = img.shape[:2]  # (h, w)
    new_h, new_w = new_shape

    r = min(new_h / shape[0], new_w / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = (r, r)
    unpad_w, unpad_h = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_w - unpad_w, new_h - unpad_h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2

    if shape[1] != unpad_w or shape[0] != unpad_h:
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def clamp_box(box: list[float], w: int, h: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w), float(x1)))
    x2 = max(0.0, min(float(w), float(x2)))
    y1 = max(0.0, min(float(h), float(y1)))
    y2 = max(0.0, min(float(h), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def main() -> None:
    args = parse_args()
    # Back-compat only; no longer required. Keep it so older invocations don't break.
    if args.repo_src:
        sys.path.insert(0, args.repo_src)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map_raw = json.load(f)
    label_map = {int(k): v for k, v in label_map_raw.items()}
    num_classes = len(label_map)

    providers_available = ort.get_available_providers()
    if args.provider == "auto" and "CUDAExecutionProvider" in providers_available:
        providers_requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers_requested = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers_requested)
    providers_used = sess.get_providers()
    input_name = sess.get_inputs()[0].name

    report: dict[str, object] = {
        "model": args.model,
        "providers_available": providers_available,
        "providers_requested": providers_requested,
        "providers_used": providers_used,
        "torch_cuda_available": torch.cuda.is_available(),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "pages": [],
    }

    with open(args.manifest, "r", encoding="utf-8") as f:
        image_paths = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if args.max_pages and args.max_pages > 0:
        image_paths = image_paths[: args.max_pages]

    for img_path in image_paths:
        img_p = Path(img_path)
        slug = img_p.stem
        page_dir = output_root / slug
        page_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = page_dir / f"{slug}_dell_layout_overlay.png"
        boxes_path = page_dir / f"{slug}_dell_layout_boxes.json"

        if args.resume and boxes_path.exists():
            try:
                existing = json.loads(boxes_path.read_text(encoding="utf-8"))
                existing_boxes = existing.get("boxes") or []
                if isinstance(existing_boxes, list):
                    # Keep overlay around for convenient visual debug; regenerate if needed.
                    if not overlay_path.exists():
                        pil = Image.open(img_p).convert("RGB")
                        draw = ImageDraw.Draw(pil)
                        color_map = {
                            "article": (220, 40, 40),
                            "headline": (30, 80, 220),
                            "table": (220, 140, 20),
                            "photograph": (130, 40, 180),
                            "image_caption": (40, 130, 180),
                            "author": (20, 150, 120),
                            "cartoon_or_advertisement": (180, 60, 160),
                            "masthead": (20, 120, 220),
                            "newspaper_header": (20, 120, 220),
                            "page_number": (90, 90, 90),
                        }
                        for b in existing_boxes:
                            bb = b.get("bbox_xyxy")
                            if not bb or len(bb) != 4:
                                continue
                            x1, y1, x2, y2 = bb
                            lbl = str(b.get("label") or "")
                            score = b.get("score")
                            color = color_map.get(lbl, (120, 120, 120))
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                            if isinstance(score, (int, float)):
                                draw.text((x1 + 2, max(0, y1 - 14)), f"{lbl}:{float(score):.2f}", fill=color)
                            else:
                                draw.text((x1 + 2, max(0, y1 - 14)), lbl, fill=color)
                        pil.save(overlay_path)

                    report["pages"].append(
                        {
                            "slug": slug,
                            "image": str(img_p),
                            "status": "resume",
                            "box_count": len(existing_boxes),
                            "overlay": str(overlay_path),
                            "boxes_json": str(boxes_path),
                        }
                    )
                    continue
            except Exception:
                # Corrupt/partial files should be regenerated.
                pass

        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            report["pages"].append({"slug": slug, "image": str(img_p), "status": "read_fail"})
            continue
        h, w = img.shape[:2]

        padded, ratio, (dw, dh) = letterbox(img, (args.imgsz, args.imgsz), auto=False)
        x = padded.transpose((2, 0, 1))[::-1]
        x = np.expand_dims(np.ascontiguousarray(x), axis=0).astype(np.float32) / 255.0

        out = sess.run(None, {input_name: x})
        pred = torch.from_numpy(out[0])
        det = non_max_suppression(
            pred,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
            agnostic=True,
            nc=num_classes,
        )[0]

        boxes = []
        if det is not None and det.numel() > 0:
            for i, row in enumerate(det):
                x1, y1, x2, y2, score, cls_id = row[:6].tolist()
                x1 = (x1 - dw) / ratio[0]
                x2 = (x2 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                y2 = (y2 - dh) / ratio[1]
                bb = clamp_box([x1, y1, x2, y2], w, h)
                if bb is None:
                    continue
                cls_id_i = int(round(cls_id))
                boxes.append(
                    {
                        "id": i,
                        "label": label_map.get(cls_id_i, f"class_{cls_id_i}"),
                        "class_id": cls_id_i,
                        "score": float(score),
                        "bbox_xyxy": bb,
                    }
                )

        pil = Image.open(img_p).convert("RGB")
        draw = ImageDraw.Draw(pil)
        color_map = {
            "article": (220, 40, 40),
            "headline": (30, 80, 220),
            "table": (220, 140, 20),
            "photograph": (130, 40, 180),
            "image_caption": (40, 130, 180),
            "author": (20, 150, 120),
            "cartoon_or_advertisement": (180, 60, 160),
            "masthead": (20, 120, 220),
            "newspaper_header": (20, 120, 220),
            "page_number": (90, 90, 90),
        }
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            lbl = b["label"]
            score = b["score"]
            color = color_map.get(lbl, (120, 120, 120))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), f"{lbl}:{score:.2f}", fill=color)

        pil.save(overlay_path)

        with open(boxes_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "slug": slug,
                    "image": str(img_p),
                    "width": w,
                    "height": h,
                    "providers_used": providers_used,
                    "boxes": boxes,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        report["pages"].append(
            {
                "slug": slug,
                "image": str(img_p),
                "status": "ok",
                "box_count": len(boxes),
                "overlay": str(overlay_path),
                "boxes_json": str(boxes_path),
            }
        )

    with open(output_root / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
