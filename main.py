#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======

Single-file version: the whole panel-first + OCR core is embedded here, so
the program runs on its own (no ``import comic_ocr_v12``).

Pipeline:

1. **Walk the kind root folder** (``mangas`` or ``webtoons``) and find the
   folders that contain images (the chapters) — at any depth (e.g.
   ``mangas/one-piece/1``). Each chapter is processed separately.
2. Detect the panels and read the text inside each panel (RT-DETR-v2 +
   PP-OCRv5). Manga -> AI (YOLO26s-seg); webtoon -> gutter/whitespace.
3. Crop and save the image of every panel that has text.
4. Summarize in **two LLM stages** (local or opencode):
   a. 1st call: write the chapter **summary**;
   b. 2nd call: receives the summary plus the panel texts and picks **only
      the panels that show what the summary says**, in the order the events
      appear in the summary, quoting the summary excerpt each panel
      illustrates and the reason.
5. Build an HTML page (summary + gallery of the main panels) and **open it
   in the browser**.

The main panels are therefore **in line with the summary** (they show what
the summary says) and there is **no limit** on how many there are.

The ``--kind`` parameter selects which root folder to scan: ``manga`` ->
``mangas/`` and ``webtoon`` -> ``webtoons/``. Its default value is ``manga``.

The ``--language`` parameter selects the output language of the AI reply and
of the generated HTML page. It accepts ``pt-br``, ``english``, ``korean``,
``japanese`` or ``chinese`` and defaults to ``english``.

Output goes to the sibling folder of the chapter, with the ``-ai`` suffix
(e.g. ``mangas/one-piece/1`` -> ``mangas/one-piece/1-ai``).

LLM backends:

- ``--llm local``    -> local AI via ``llama-cpp-python`` (Qwen3, GGUF);
- ``--llm opencode`` -> via the **OpenCode CLI** (``opencode run``), using
  the model configured in OpenCode.

The GPU is used automatically when CUDA is available (for detection/OCR and,
if ``llama-cpp-python`` was built with CUDA, for the LLM too); otherwise it
falls back to CPU. Force a device with ``--device``.

With the ``opencode`` backend the program shows the **cost and tokens** of
the calls (it uses ``opencode session export`` and sums the session usage)
in the terminal, in ``summary.txt`` and in the HTML footer.

Usage::

    python main.py                              # scan mangas/ (whole tree)
    python main.py --kind webtoon               # scan webtoons/
    python main.py --language pt-br             # AI reply + HTML in pt-BR
    python main.py --llm opencode               # via opencode (default model)
    python main.py --llm opencode --opencode-model opencode-go/deepseek-v4.1-flash
    python main.py --model Qwen/Qwen3-8B-GGUF    # local backend
    python main.py --device cuda                 # force the GPU
    python main.py --max 0

Install::

    pip install torch transformers ultralytics onnxruntime \\
                opencv-python pillow numpy huggingface_hub
    # local backend:
    pip install llama-cpp-python
    # opencode backend: have the `opencode` binary on PATH.

Everything (identifiers, comments, docstrings, CLI help, terminal messages
and the LLM prompts) is written in English. Only the AI reply and the
generated HTML page follow the language chosen with ``--language``.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import urllib.request
import uuid
import webbrowser
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# --- PANELS: segmentation AI (YOLO26s-seg / ShadowB) ---
PANEL_REPO = "ShadowB/Manga109-panel-balloon-text-yolov26-segmentation"
PANEL_WEIGHTS = "best.pt"
PANEL_CLASS_NAMES = {"frame", "panel"}
PANEL_IMGSZ = 1280
PANEL_CONF = 0.25
PANEL_IOU = 0.7
PANEL_PAD = 8

# --- PANELS: gutter/whitespace (webtoon) ---
GUTTER_MIN_PX = 10
BG_THRESH = 0.98
BG_TOL = 14
MIN_PANEL_PX = 40
COL_SPLIT = True

# --- TEXT: RT-DETR-v2 ---
DETECTOR_REPO = "ogkalu/comic-text-and-bubble-detector"
CONFIDENCE_THRESHOLD = 0.3
TEXT_LABELS = {1, 2}
BUBBLE_LABELS = {0}
CLASS_NAMES = {0: "bubble", 1: "text_bubble", 2: "text_free"}

# --- OCR: PP-OCRv5 ---
PPOCR_BASE_URL = "https://huggingface.co/ogkalu/ppocr-v5-onnx/resolve/main/"
PPOCR_DET_FILE = "ch_PP-OCRv5_mobile_det.onnx"
PPOCR_REC_FILE = "ch_PP-OCRv5_rec_mobile_infer.onnx"
PPOCR_DICT_FILE = "ppocrv5_dict.txt"

MODELS_DIR = os.environ.get(
    "COMIC_OCR_MODELS_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "comic_ocr", "models"),
)

DB_THRESH = 0.3
DB_BOX_THRESH = 0.5
DB_UNCLIP_RATIO = 2.0
DB_MAX_CANDIDATES = 1000

# --- BATCH (folders) ---
MAX_IMAGES = 0  # 0 = every image in the chapter
OUTPUT_SUFFIX = "-ai"  # output: <chapter folder>-ai
# reading kind -> root folder that gets scanned
ROOTS = {"manga": "mangas", "webtoon": "webtoons"}
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (255, 165, 0),
    (186, 85, 211),
    (0, 206, 209),
    (219, 112, 147),
    (154, 205, 50),
    (70, 130, 180),
    (205, 133, 63),
]
TEXT_BOX_COLOR = (255, 0, 0)
TEXT_LABEL_COLOR = (255, 0, 255)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

# --- Local LLM (llama.cpp / GGUF) ---
MODEL_REPO = "Qwen/Qwen3-4B-GGUF"  # switch to Qwen/Qwen3-8B-GGUF if it fits
MODEL_FILE = None  # None = pick automatically (Q4_K_M)
N_CTX = 8192
N_GPU_LAYERS = -1  # -1 = offload as much as possible to the GPU
TEMPERATURE = 0.3
MAX_SUMMARY_TOKENS = 1024

# LLM backend: "local" (llama-cpp-python) or "opencode" (CLI)
LLM_BACKEND = "local"
OPENCODE_BIN = "opencode"
OPENCODE_MODEL = "opencode-go/deepseek-v4.1-flash"  # default opencode model
OPENCODE_AGENT = None  # e.g. the name of an opencode agent

# Output language: the value is the CLI option; the label is the language name
# handed to the LLM, and html_lang is the value used in the <html lang> tag.
SUPPORTED_LANGUAGES = {
    "pt-br": {"label": "Brazilian Portuguese", "html_lang": "pt-BR"},
    "english": {"label": "English", "html_lang": "en"},
    "korean": {"label": "Korean", "html_lang": "ko"},
    "japanese": {"label": "Japanese", "html_lang": "ja"},
    "chinese": {"label": "Chinese", "html_lang": "zh"},
}
DEFAULT_LANGUAGE = "english"

# UI strings of the generated HTML page, translated per output language.
HTML_STRINGS = {
    "pt-br": {
        "title": "Resumo",
        "generated": "gerado em",
        "panels_with_text": "paineis com texto",
        "main_panels": "paineis principais",
        "matching_summary": "(de acordo com o resumo)",
        "click_to_zoom": "clique nas imagens para ampliar",
        "zoom_nav_hint": "(no zoom, use as setas ← → para navegar)",
        "main_heading": "Principais paineis (de acordo com o resumo)",
        "no_summary": "(sem resumo)",
        "no_panel": "(nenhum painel selecionado)",
        "why_panel": "Por que este painel:",
        "in_summary": "No resumo:",
        "panel": "painel",
        "zoom_tip": "setas ← → para navegar · clique fora ou Esc para fechar",
        "prev_panel": "painel anterior",
        "next_panel": "proximo painel",
    },
    "english": {
        "title": "Summary",
        "generated": "generated at",
        "panels_with_text": "panels with text",
        "main_panels": "main panels",
        "matching_summary": "(matching the summary)",
        "click_to_zoom": "click the images to zoom",
        "zoom_nav_hint": "(while zoomed, use ← → to navigate)",
        "main_heading": "Main panels (matching the summary)",
        "no_summary": "(no summary)",
        "no_panel": "(no panel selected)",
        "why_panel": "Why this panel:",
        "in_summary": "In the summary:",
        "panel": "panel",
        "zoom_tip": "← → to navigate · click outside or press Esc to close",
        "prev_panel": "previous panel",
        "next_panel": "next panel",
    },
    "korean": {
        "title": "요약",
        "generated": "생성 시각",
        "panels_with_text": "텍스트가 있는 패널",
        "main_panels": "주요 패널",
        "matching_summary": "(요약과 일치)",
        "click_to_zoom": "이미지를 클릭하여 확대",
        "zoom_nav_hint": "(확대 상태에서 ← → 로 이동)",
        "main_heading": "주요 패널 (요약과 일치)",
        "no_summary": "(요약 없음)",
        "no_panel": "(선택된 패널 없음)",
        "why_panel": "이 패널을 선택한 이유:",
        "in_summary": "요약에서:",
        "panel": "패널",
        "zoom_tip": "← → 로 이동 · 바깥을 클릭하거나 Esc로 닫기",
        "prev_panel": "이전 패널",
        "next_panel": "다음 패널",
    },
    "japanese": {
        "title": "あらすじ",
        "generated": "生成日時",
        "panels_with_text": "テキストのあるコマ",
        "main_panels": "主要なコマ",
        "matching_summary": "(あらすじに一致)",
        "click_to_zoom": "画像をクリックして拡大",
        "zoom_nav_hint": "(拡大時は ← → で移動)",
        "main_heading": "主要なコマ (あらすじに一致)",
        "no_summary": "(あらすじなし)",
        "no_panel": "(選択されたコマなし)",
        "why_panel": "このコマを選んだ理由:",
        "in_summary": "あらすじより:",
        "panel": "コマ",
        "zoom_tip": "← → で移動 · 外側をクリックまたは Esc で閉じる",
        "prev_panel": "前のコマ",
        "next_panel": "次のコマ",
    },
    "chinese": {
        "title": "摘要",
        "generated": "生成于",
        "panels_with_text": "含文字的格子",
        "main_panels": "主要格子",
        "matching_summary": "(与摘要一致)",
        "click_to_zoom": "点击图片放大",
        "zoom_nav_hint": "(放大时用 ← → 导航)",
        "main_heading": "主要格子 (与摘要一致)",
        "no_summary": "(无摘要)",
        "no_panel": "(未选择格子)",
        "why_panel": "选择此格子的原因：",
        "in_summary": "摘要中：",
        "panel": "格子",
        "zoom_tip": "← → 导航 · 点击外部或按 Esc 关闭",
        "prev_panel": "上一个格子",
        "next_panel": "下一个格子",
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _panel_color(index: int) -> Tuple[int, int, int]:
    """Return a deterministic color for the given panel index."""
    return PALETTE[index % len(PALETTE)]


def _download(url: str, dest: str) -> str:
    """Download ``url`` to ``dest`` once, skipping it when already present."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[download] {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def _hf_download(repo_id: str, filename: str) -> str:
    """Download a single file from a Hugging Face repository."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=os.path.join(MODELS_DIR, "hf"),
    )


def _ort_providers() -> List[str]:
    """Return the ONNX Runtime execution providers, preferring CUDA."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _session_options():
    """Build the shared ONNX Runtime session options."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    return opts


def _order_points(points: np.ndarray) -> np.ndarray:
    """Sort four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    xs = points[np.argsort(points[:, 0])]
    left, right = xs[:2], xs[2:]
    top_left, bottom_left = left[np.argsort(left[:, 1])]
    top_right, bottom_right = right[np.argsort(right[:, 1])]
    return np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )


def _quad_axis_box(quad: np.ndarray) -> Tuple[int, int, int, int]:
    """Return the axis-aligned bounding box of a quadrilateral."""
    points = np.asarray(quad).reshape(-1, 2)
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _is_mostly_contained(outer, inner, threshold: float = 0.5) -> bool:
    """Return True when ``inner`` is mostly inside ``outer``."""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inner_area <= 0:
        return False
    intersection = max(0, min(ix2, ox2) - max(ix1, ox1)) * max(
        0, min(iy2, oy2) - max(iy1, oy1)
    )
    return intersection / inner_area >= threshold


def _iou(box_a, box_b) -> float:
    """Return the intersection-over-union of two axis-aligned boxes."""
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _area(box) -> int:
    """Return the area of an axis-aligned box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _merge_overlapping_items(
    items: list,
    containment_threshold: float = 0.3,
    overlap_threshold: float = 0.5,
) -> list:
    """Merge boxes that contain each other and drop heavily overlapping ones."""
    accepted: list = []
    for index, item in enumerate(items):
        merged = list(item[:4])
        best = item
        for other_index, other in enumerate(items):
            if index == other_index:
                continue
            if _is_mostly_contained(
                merged, other[:4], containment_threshold
            ) or _is_mostly_contained(
                other[:4], merged, containment_threshold
            ):
                merged = [
                    min(merged[0], other[0]),
                    min(merged[1], other[1]),
                    max(merged[2], other[2]),
                    max(merged[3], other[3]),
                ]
                if other[4] > best[4]:
                    best = other
        conflict = False
        for current in accepted:
            if (
                merged == current[:4]
                or _iou(merged, current[:4]) >= overlap_threshold
            ):
                conflict = True
                break
        if conflict:
            continue
        accepted = [
            current
            for current in accepted
            if not (
                merged == current[:4]
                or _iou(merged, current[:4]) >= overlap_threshold
            )
        ]
        accepted.append(
            (merged[0], merged[1], merged[2], merged[3], best[4], best[5])
        )
    return accepted


def _load_font(size: int):
    """Load a TrueType font when available, else PIL's default font."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _natural_key(name: str):
    """Return a key that sorts names with embedded numbers naturally."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def list_images(folder: str) -> List[str]:
    """List the image files of ``folder`` in natural order."""
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(EXTENSIONS):
            files.append(path)
    files.sort(key=lambda path: _natural_key(os.path.basename(path)))
    return files


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------


def sort_reading_order(boxes: list, kind: str) -> List[int]:
    """Return the indices of ``boxes`` in the reading order of ``kind``.

    - manga   : rows top to bottom; inside a row, right to left.
    - webtoon : rows top to bottom; inside a row, left to right.
    """
    if not boxes:
        return []
    heights = [box[3] - box[1] for box in boxes]
    median = float(np.median(heights)) if heights else 1.0
    tolerance = max(10.0, median * 0.5)

    items = sorted(
        range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2.0
    )
    rows: List[dict] = []
    for index in items:
        center_y = (boxes[index][1] + boxes[index][3]) / 2.0
        if rows and abs(center_y - rows[-1]["cy"]) <= tolerance:
            rows[-1]["indices"].append(index)
            rows[-1]["cy"] = float(
                np.mean(
                    [
                        (boxes[k][1] + boxes[k][3]) / 2.0
                        for k in rows[-1]["indices"]
                    ]
                )
            )
        else:
            rows.append({"cy": center_y, "indices": [index]})

    # both webtoon and manga go top to bottom; they differ only in the
    # horizontal order inside each row.
    order: List[int] = []
    for row in rows:
        right_to_left = kind == "manga"  # manga: right -> left
        row["indices"].sort(
            key=lambda i: (boxes[i][0] + boxes[i][2]) / 2.0,
            reverse=right_to_left,
        )
        order.extend(row["indices"])
    return order


# ---------------------------------------------------------------------------
# PANELS: gutter/whitespace (webtoon)
# ---------------------------------------------------------------------------


def _estimate_background(img_bgr: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Estimate the dominant background color of an image."""
    step = 256 // n_bins
    quantized = (img_bgr // step).astype(np.int32)
    codes = (
        quantized[:, :, 0] * n_bins + quantized[:, :, 1]
    ) * n_bins + quantized[:, :, 2]
    counts = np.bincount(codes.ravel(), minlength=n_bins**3)
    code = int(counts.argmax())
    red = (code // (n_bins * n_bins)) % n_bins
    green = (code // n_bins) % n_bins
    blue = code % n_bins
    return np.array(
        [
            red * step + step // 2,
            green * step + step // 2,
            blue * step + step // 2,
        ],
        dtype=np.int32,
    )


def _mask_runs(mask_1d: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Return the ``(start, end)`` runs of True values at least ``min_len`` long."""
    runs, start = [], None
    for index, value in enumerate(mask_1d):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_len:
                runs.append((start, index))
            start = None
    if start is not None and len(mask_1d) - start >= min_len:
        runs.append((start, len(mask_1d)))
    return runs


def _segments(
    total: int, runs: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return the complementary segments between the given runs."""
    segments, previous = [], 0
    for start, end in runs:
        if start > previous:
            segments.append((previous, start))
        previous = end
    if previous < total:
        segments.append((previous, total))
    return segments


def panels_by_gutter(
    img_bgr: np.ndarray,
    min_gutter: int = GUTTER_MIN_PX,
    bg_thresh: float = BG_THRESH,
    tol: int = BG_TOL,
    min_panel: int = MIN_PANEL_PX,
    col_split: bool = COL_SPLIT,
) -> list:
    """Detect panels of a webtoon image through whitespace (gutter) splitting."""
    height, width = img_bgr.shape[:2]
    background = _estimate_background(img_bgr)
    difference = np.abs(
        img_bgr.astype(np.int32) - background[None, None, :]
    ).max(axis=2)
    is_background = difference <= tol

    rows = _segments(
        height, _mask_runs(is_background.mean(axis=1) >= bg_thresh, min_gutter)
    )
    panels = []
    for y0, y1 in rows:
        if y1 - y0 < min_panel:
            continue
        if not col_split:
            panels.append((0, y0, width, y1))
            continue
        column_gutter = is_background[y0:y1].mean(axis=0) >= bg_thresh
        columns = [
            (x0, x1)
            for (x0, x1) in _segments(
                width, _mask_runs(column_gutter, min_gutter)
            )
            if x1 - x0 >= min_panel
        ]
        for x0, x1 in columns or [(0, width)]:
            panels.append((x0, y0, x1, y1))

    if not panels:
        panels = [(0, 0, width, height)]
    return [
        {"box": tuple(int(v) for v in box), "score": 1.0, "mask": None}
        for box in panels
    ]


# ---------------------------------------------------------------------------
# PANELS: segmentation AI (YOLO26s-seg / ShadowB) (manga)
# ---------------------------------------------------------------------------


class MangaSegmenter:
    """Panel segmentation model for manga pages (YOLO26s-seg, ShadowB)."""

    def __init__(
        self,
        device: Optional[str] = None,
        conf: float = PANEL_CONF,
        iou: float = PANEL_IOU,
        imgsz: int = PANEL_IMGSZ,
    ):
        """Load the segmentation weights and choose the inference device."""
        self.model = YOLO(_hf_download(PANEL_REPO, PANEL_WEIGHTS))
        self.names = dict(self.model.names)
        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device, self.conf, self.iou, self.imgsz = device, conf, iou, imgsz
        print(
            f"[panels] {PANEL_REPO} ({self.device}) | classes={self.names}",
            file=sys.stderr,
        )

    def segment(self, img_bgr: np.ndarray) -> list:
        """Return the detected regions (with optional masks) of an image."""
        results = self.model.predict(
            source=img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            retina_masks=True,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()
        regions = []
        if result.boxes is not None:
            for index, box in enumerate(result.boxes):
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                mask = None
                if masks is not None and index < len(masks):
                    mask = (masks[index] > 0.5).astype(np.uint8)
                regions.append(
                    {
                        "kind": self.names.get(class_id, str(class_id)),
                        "box": (x1, y1, x2, y2),
                        "score": round(float(box.conf[0].item()), 3),
                        "mask": mask,
                    }
                )
        return regions


def panels_by_ai(img_bgr: np.ndarray, segmenter: MangaSegmenter) -> list:
    """Detect manga panels with the segmentation model, dropping duplicates."""
    regions = segmenter.segment(img_bgr)
    panels = [
        r for r in regions if str(r["kind"]).lower() in PANEL_CLASS_NAMES
    ]
    kept = []
    for panel in sorted(panels, key=lambda r: -_area(r["box"])):
        if all(_iou(panel["box"], other["box"]) < 0.6 for other in kept):
            kept.append(panel)
    if not kept:
        height, width = img_bgr.shape[:2]
        return [{"box": (0, 0, width, height), "score": 1.0, "mask": None}]
    return [
        {"box": p["box"], "score": p["score"], "mask": p.get("mask")}
        for p in kept
    ]


# ---------------------------------------------------------------------------
# TEXT: RT-DETR-v2
# ---------------------------------------------------------------------------


class BubbleTextDetector:
    """Comic text/bubble detector based on RT-DETR-v2."""

    def __init__(self, device: Optional[str] = None):
        """Load the detector and move it to the chosen device."""
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[text] {DETECTOR_REPO} ({device})...", file=sys.stderr)
        self.processor = RTDetrImageProcessor.from_pretrained(
            DETECTOR_REPO, size={"width": 640, "height": 640}
        )
        self.model = (
            RTDetrV2ForObjectDetection.from_pretrained(DETECTOR_REPO)
            .to(device)
            .eval()
        )

    def detect(self, img_bgr: np.ndarray):
        """Return the text boxes and the bubble boxes of an image."""
        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {
            key: (value.to(self.device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self._torch.tensor([pil.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
        )[0]
        text_boxes, bubble_boxes = [], []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            if x2 <= x1 or y2 <= y1:
                continue
            item = (x1, y1, x2, y2, float(score.item()), int(label.item()))
            if int(label.item()) in BUBBLE_LABELS:
                bubble_boxes.append(item)
            elif int(label.item()) in TEXT_LABELS:
                text_boxes.append(item)
        return text_boxes, bubble_boxes


# ---------------------------------------------------------------------------
# OCR: PP-OCRv5
# ---------------------------------------------------------------------------


class CTCLabelDecoder:
    """CTC greedy decoder for the PP-OCRv5 recognition output."""

    def __init__(self, dict_path: str):
        """Load the character dictionary from ``dict_path``."""
        with open(dict_path, "r", encoding="utf-8") as handle:
            self.chars = [line.rstrip("\n") for line in handle]

    def __call__(self, logits: np.ndarray, prob_threshold: float = 0.0):
        """Decode a batch of logits into ``(texts, scores)``."""
        if logits.ndim == 2:
            logits = logits[None, ...]
        num_classes = logits.shape[-1]
        dict_len = len(self.chars)
        if num_classes == dict_len + 2:
            vocab = [""] + self.chars + [" "]
        elif num_classes == dict_len + 1:
            vocab = [""] + self.chars
        elif num_classes == dict_len:
            pad = np.zeros((*logits.shape[:-1], 1), dtype=logits.dtype)
            logits = np.concatenate([pad, logits], axis=-1)
            vocab = [""] + self.chars
        else:
            vocab = [""] + self.chars[: max(0, num_classes - 1)]
        if np.max(logits) > 1.0 or np.min(logits) < 0.0:
            exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exponent / exponent.sum(axis=-1, keepdims=True)
        else:
            probs = logits
        texts, blank = [], 0
        for sequence in probs:
            indices = sequence.argmax(axis=-1)
            last, decoded = -1, []
            for step, index in enumerate(indices):
                if index not in (blank, last):
                    prob = float(sequence[step, int(index)])
                    if prob < prob_threshold or index >= len(vocab):
                        last = int(index)
                        continue
                    char = vocab[int(index)]
                    if char and not any(ord(c) < 32 for c in char):
                        decoded.append(char)
                last = int(index)
            texts.append("".join(decoded))
        return texts, []


class PPOCRv5:
    """PP-OCRv5 text detection and recognition through ONNX Runtime."""

    def __init__(self, device: Optional[str] = None):
        """Download the models and create the ONNX Runtime sessions.

        ``device`` is accepted for interface symmetry; the execution provider
        is chosen automatically by :func:`_ort_providers`.
        """
        self.device = device
        det_path = _download(
            PPOCR_BASE_URL + PPOCR_DET_FILE,
            os.path.join(MODELS_DIR, PPOCR_DET_FILE),
        )
        rec_path = _download(
            PPOCR_BASE_URL + PPOCR_REC_FILE,
            os.path.join(MODELS_DIR, PPOCR_REC_FILE),
        )
        dict_path = _download(
            PPOCR_BASE_URL + PPOCR_DICT_FILE,
            os.path.join(MODELS_DIR, PPOCR_DICT_FILE),
        )
        providers = _ort_providers()
        print(f"[ocr] PP-OCRv5 ({providers[0]})...", file=sys.stderr)
        opts = _session_options()
        self.det_sess = ort.InferenceSession(
            det_path, sess_options=opts, providers=providers
        )
        self.rec_sess = ort.InferenceSession(
            rec_path, sess_options=opts, providers=providers
        )
        self.decoder = CTCLabelDecoder(dict_path)
        self.rec_img_shape = (3, 48, 320)

    @staticmethod
    def _resize_keep_stride(img, limit_side_len=960, limit_type="min"):
        """Resize keeping the side multiple of 32, as PP-OCR expects."""
        height, width = img.shape[:2]
        if limit_type == "max":
            longest = max(height, width)
            ratio = (
                limit_side_len / longest if longest > limit_side_len else 1.0
            )
        else:
            shortest = min(height, width)
            ratio = (
                limit_side_len / shortest if shortest < limit_side_len else 1.0
            )
        new_height = max(int(round(height * ratio / 32) * 32), 32)
        new_width = max(int(round(width * ratio / 32) * 32), 32)
        if new_height == height and new_width == width:
            return img
        return cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    def _det_preprocess(self, img_bgr):
        """Normalize an image into the detection model input tensor."""
        resized = self._resize_keep_stride(img_bgr, 960, "min")
        data = resized.astype(np.float32) / 255.0
        data = (data - 0.5) / 0.5
        return np.expand_dims(data.transpose(2, 0, 1), 0).astype(np.float32)

    def _rec_resize_norm(self, crop_bgr):
        """Normalize a cropped text line into the recognition input tensor."""
        _, rec_height, rec_width = self.rec_img_shape
        height, width = crop_bgr.shape[:2]
        ratio = width / float(max(1, height))
        target_width = max(rec_width, int(np.ceil(rec_height * ratio)))
        resized_width = min(target_width, int(np.ceil(rec_height * ratio)))
        resized = cv2.resize(
            crop_bgr,
            (max(1, resized_width), rec_height),
            interpolation=cv2.INTER_LINEAR,
        )
        data = resized.astype(np.float32) / 255.0
        data = (data - 0.5) / 0.5
        output = np.zeros((3, rec_height, target_width), dtype=np.float32)
        output[:, :, :resized_width] = data.transpose(2, 0, 1)
        return output

    @staticmethod
    def _box_score_fast(prob, box):
        """Return the mean probability inside a quadrilateral box."""
        height, width = prob.shape
        xs, ys = box[:, 0], box[:, 1]
        xmin = int(np.clip(np.floor(xs.min()), 0, width - 1))
        xmax = int(np.clip(np.ceil(xs.max()), 0, width - 1))
        ymin = int(np.clip(np.floor(ys.min()), 0, height - 1))
        ymax = int(np.clip(np.ceil(ys.max()), 0, height - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        points = box.copy()
        points[:, 0] -= xmin
        points[:, 1] -= ymin
        cv2.fillPoly(mask, [points.astype(np.int32).reshape(-1, 1, 2)], 1)
        region = prob[ymin : ymax + 1, xmin : xmax + 1]
        total = int(mask.sum())
        return float((region * mask).sum() / total) if total else 0.0

    def _db_postprocess(self, prediction, original_height, original_width):
        """Turn the DB probability map into a list of quadrilateral boxes."""
        prob = prediction[0, 0]
        mask = cv2.dilate(
            (prob > DB_THRESH).astype(np.uint8), np.ones((2, 2), np.uint8)
        )
        contours, _ = cv2.findContours(
            mask * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = mask.shape
        boxes, scores = [], []
        for contour in contours[:DB_MAX_CANDIDATES]:
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < 3:
                continue
            box = _order_points(cv2.boxPoints(rect))
            score = self._box_score_fast(prob, box)
            if score < DB_BOX_THRESH:
                continue
            area = float(rect[1][0] * rect[1][1])
            perimeter = float(2 * (rect[1][0] + rect[1][1]))
            if area <= 0 or perimeter <= 0:
                continue
            delta = area * DB_UNCLIP_RATIO / perimeter
            expanded = (
                rect[0],
                (rect[1][0] + 2 * delta, rect[1][1] + 2 * delta),
                rect[2],
            )
            if min(expanded[1]) < 5:
                continue
            box = _order_points(cv2.boxPoints(expanded))
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * original_width), 0, original_width
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * original_height),
                0,
                original_height,
            )
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return boxes, scores

    def detect_lines(self, img_bgr):
        """Return the text-line quadrilaterals detected by the DB model."""
        inputs = self._det_preprocess(img_bgr)
        input_name = self.det_sess.get_inputs()[0].name
        output_name = self.det_sess.get_outputs()[0].name
        prediction = self.det_sess.run([output_name], {input_name: inputs})[0]
        boxes, _ = self._db_postprocess(
            prediction, img_bgr.shape[0], img_bgr.shape[1]
        )
        return boxes

    @staticmethod
    def _crop_quad(img_bgr, quad):
        """Perspective-crop a quadrilateral, rotating tall crops by 90 degrees."""
        points = _order_points(quad)
        width = int(
            round(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3]),
                )
            )
        )
        height = int(
            round(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]),
                )
            )
        )
        if width < 1 or height < 1:
            return None
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        try:
            matrix = cv2.getPerspectiveTransform(
                points.astype(np.float32), destination
            )
            crop = cv2.warpPerspective(img_bgr, matrix, (width, height))
        except cv2.error:
            return None
        if height / float(width) >= 1.5:
            crop = np.rot90(crop)
        return crop

    def recognize_quads(self, img_bgr, quads):
        """Recognize the text inside each quadrilateral of ``quads``."""
        recognized = []
        for quad in quads:
            crop = self._crop_quad(img_bgr, quad)
            if crop is None or crop.size == 0:
                continue
            data = self._rec_resize_norm(crop)[None, ...].astype(np.float32)
            input_name = self.rec_sess.get_inputs()[0].name
            output_name = self.rec_sess.get_outputs()[0].name
            logits = self.rec_sess.run([output_name], {input_name: data})[0]
            if logits.ndim == 3 and logits.shape[1] > logits.shape[2]:
                logits = np.transpose(logits, (0, 2, 1))
            texts, _ = self.decoder(logits)
            text = (texts[0] if texts else "").strip()
            if text:
                recognized.append((np.asarray(quad, dtype=np.int32), text))
        return recognized

    def read_full(self, img_bgr):
        """Detect and recognize every text line of an image."""
        return self.recognize_quads(img_bgr, self.detect_lines(img_bgr))


def detect_texts(img_bgr, detector, ocr):
    """Match detector text boxes with OCR output and return the panel texts."""
    text_boxes, _ = detector.detect(img_bgr)
    text_boxes = _merge_overlapping_items(text_boxes)
    lines = ocr.read_full(img_bgr)
    texts = []
    for x1, y1, x2, y2, score, label in text_boxes:
        box = (x1, y1, x2, y2)
        matched = [
            (quad, text)
            for quad, text in lines
            if _is_mostly_contained(box, _quad_axis_box(quad), 0.5)
            or _is_mostly_contained(_quad_axis_box(quad), box, 0.9)
        ]
        matched.sort(
            key=lambda item: (
                _quad_axis_box(item[0])[1],
                _quad_axis_box(item[0])[0],
            )
        )
        text = " ".join(value for _, value in matched).strip()
        if not text:
            read = ocr.recognize_quads(
                img_bgr,
                [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)],
            )
            text = read[0][1] if read else ""
        texts.append(
            {
                "box": box,
                "text": text,
                "score": round(score, 3),
                "label": CLASS_NAMES.get(label, "text"),
            }
        )
    return texts


# ---------------------------------------------------------------------------
# PANEL-FIRST flow + reading order
# ---------------------------------------------------------------------------


def _detect_panels(img_bgr, kind, segmenter, gutter_kwargs, panel_order):
    """Detect the panels of an image using AI or the gutter method."""
    use_ai = (panel_order == "ia") or (
        panel_order == "auto" and kind == "manga"
    )
    if use_ai and segmenter is not None:
        return panels_by_ai(img_bgr, segmenter)
    return panels_by_gutter(img_bgr, **(gutter_kwargs or {}))


def _read_panel(img_bgr, panel, index, kind, detector, ocr, pad):
    """Read the text inside one panel and return its absolute-order result."""
    height, width = img_bgr.shape[:2]
    px1, py1, px2, py2 = panel["box"]
    x0, y0 = max(0, px1 - pad), max(0, py1 - pad)
    x1, y1 = min(width, px2 + pad), min(height, py2 + pad)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return {"panel": index, "texts": []}
    texts = detect_texts(crop, detector, ocr)
    for text in texts:
        tx1, ty1, tx2, ty2 = text["box"]
        text["box"] = (tx1 + x0, ty1 + y0, tx2 + x0, ty2 + y0)
    order = sort_reading_order([t["box"] for t in texts], kind)
    return {"panel": index, "texts": [texts[i] for i in order]}


def process_one(
    img_bgr,
    kind,
    detector,
    ocr,
    segmenter=None,
    pad=PANEL_PAD,
    gutter_kwargs=None,
    panel_order="auto",
):
    """Detect/order panels, crop them and read the text inside each panel."""
    panels = _detect_panels(
        img_bgr, kind, segmenter, gutter_kwargs, panel_order
    )
    order = sort_reading_order([panel["box"] for panel in panels], kind)
    panels = [panels[i] for i in order]
    results = []
    for index, panel in enumerate(panels):
        results.append(
            _read_panel(img_bgr, panel, index, kind, detector, ocr, pad)
        )
    return panels, results


def _draw_label(draw, font, x, y, text, color):
    """Draw a small filled label with white text at ``(x, y)``."""
    try:
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    except (ValueError, OSError):
        return
    draw.rectangle([left - 2, top - 2, right + 2, bottom + 2], fill=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _blend_masks(base, panels):
    """Blend the panel segmentation masks onto the base image."""
    height, width = base.shape[:2]
    overlay = base.copy()
    has_mask = False
    for index, panel in enumerate(panels):
        mask = panel.get("mask")
        if mask is None:
            continue
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        overlay[mask > 0] = _panel_color(index)
        has_mask = True
    if has_mask:
        return cv2.addWeighted(overlay, 0.35, base, 0.65, 0)
    return base


def _draw_boxes(
    draw, font, line_width, panels, results, height, kind, show_text
):
    """Draw the panel outlines and the text boxes with their labels."""
    for index, panel in enumerate(panels):
        x1, y1, x2, y2 = panel["box"]
        color = _panel_color(index)
        draw.rectangle(
            [x1, y1, x2, y2], outline=color, width=max(line_width, 3)
        )
        _draw_label(draw, font, x1, max(0, y1 - 15), f"P{index}", color)

    for result in results:
        panel_index = result["panel"]
        for text_index, text in enumerate(result["texts"]):
            x1, y1, x2, y2 = text["box"]
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=TEXT_BOX_COLOR,
                width=max(1, line_width - 1),
            )
            _draw_label(
                draw,
                font,
                x1,
                min(height - 12, y2 + 2),
                f"P{panel_index}.T{text_index}",
                TEXT_LABEL_COLOR,
            )
            if show_text and text["text"]:
                _draw_label(
                    draw,
                    font,
                    x1,
                    min(height - 12, y2 + 16),
                    text["text"],
                    TEXT_LABEL_COLOR,
                )

    text_count = sum(len(result["texts"]) for result in results)
    _draw_label(
        draw,
        font,
        4,
        4,
        f"[{kind}] {len(panels)} panels | {text_count} texts",
        (40, 40, 40),
    )


def draw_annotations(img_bgr, panels, results, kind, show_text=True):
    """Return an annotated copy of the image with panels and texts drawn."""
    base = _blend_masks(img_bgr, panels)
    height, width = base.shape[:2]
    pil = Image.fromarray(np.ascontiguousarray(base[:, :, ::-1]))
    draw = ImageDraw.Draw(pil)
    line_width = max(round((height + width) / 2 * 0.003), 2)
    font = _load_font(max(12, round(min(height, width) * 0.018)))
    _draw_boxes(
        draw, font, line_width, panels, results, height, kind, show_text
    )
    return pil


# ---------------------------------------------------------------------------
# Local GGUF model (llama-cpp-python)
# ---------------------------------------------------------------------------


def download_gguf(repo_id: str, filename: Optional[str] = None) -> str:
    """Download a GGUF file from Hugging Face (Q4_K_M by default)."""
    if filename is None:
        files = [
            name
            for name in list_repo_files(repo_id)
            if name.lower().endswith(".gguf")
        ]
        if not files:
            raise SystemExit(f"No .gguf file found in {repo_id}")
        preferred = [name for name in files if "q4_k_m" in name.lower()]
        filename = (preferred or files)[0]
        print(f"[llm] chosen file: {filename}", file=sys.stderr)

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=os.path.join(MODELS_DIR, "gguf"),
    )


def cuda_available() -> bool:
    """Return True when CUDA (torch) is available."""
    return bool(torch.cuda.is_available())


def load_llm(
    repo_id: str,
    filename: Optional[str] = None,
    n_gpu_layers: int = N_GPU_LAYERS,
):
    """Load a local GGUF model through llama-cpp-python."""
    try:
        from llama_cpp import Llama  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is missing. Run:\n"
            "  pip install llama-cpp-python\n"
            f"({exc})"
        ) from exc
    path = download_gguf(repo_id, filename)
    where = "GPU" if n_gpu_layers != 0 else "CPU"
    print(
        f"[llm] loading {os.path.basename(path)} (n_ctx={N_CTX}, {where})...",
        file=sys.stderr,
    )
    return Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Summary (2 stages: summary -> panels matching the summary)
# ---------------------------------------------------------------------------


def _extract_json(text: str):
    """Extract the first JSON object that looks like a summary/panel reply."""
    # non-greedy candidates first (avoids matching two objects at once)
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("summary" in obj or "panels" in obj):
            return obj
    greedy = re.search(r"\{.*\}", text, re.DOTALL)  # greedy fallback
    if not greedy:
        return None
    try:
        return json.loads(greedy.group(0))
    except json.JSONDecodeError:
        return None


def _texts_body(
    chapter_text: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Return the prompt excerpt that presents the texts (inline or attached)."""
    if chapter_text is not None:
        return (
            "Below are the texts extracted from a chapter, one per panel, "
            "in the format [id] text:\n\n" + chapter_text + "\n\n"
        )
    return (
        "The texts extracted from the chapter are in the attached file "
        f'"{filename}" (one per panel, in the format [id] text).\n\n'
    )


def build_summary_prompt(
    language: str = DEFAULT_LANGUAGE,
    chapter_text: Optional[str] = None,
    filename: Optional[str] = None,
    no_think: bool = False,
) -> str:
    """Build the 1st-stage prompt: only the chapter summary."""
    language_label = SUPPORTED_LANGUAGES[language]["label"]
    prompt = (
        "You summarize comics.\n\n"
        + _texts_body(chapter_text, filename)
        + "Task:\n"
        "Write a summary of the main events of the chapter.\n"
        f"Write the summary in {language_label}.\n\n"
        "Reply ONLY with valid JSON, no extra text, in the format:\n"
        '{"summary": "..."}'
    )
    if no_think:
        prompt += "\n/no_think"  # Qwen3: disable the "thinking" mode
    return prompt


def build_panels_prompt(
    summary: str,
    language: str = DEFAULT_LANGUAGE,
    chapter_text: Optional[str] = None,
    filename: Optional[str] = None,
    no_think: bool = False,
) -> str:
    """Build the 2nd-stage prompt: choose the panels that show the summary."""
    language_label = SUPPORTED_LANGUAGES[language]["label"]
    prompt = (
        "You pick comic panels that illustrate a summary.\n\n"
        "SUMMARY of the chapter:\n"
        + summary.strip()
        + "\n\n"
        + _texts_body(chapter_text, filename)
        + "Task:\n"
        "Pick ONLY the panels whose content shows what the summary says "
        "(the panels that illustrate the events of the summary).\n"
        "- Use only ids that appear in the text list above.\n"
        "- Order the panels in the SAME order the events appear in the summary.\n"
        "- For each panel, quote the summary excerpt it illustrates "
        '("excerpt") and explain in 1 sentence what it shows ("reason").\n'
        f"Write the excerpt and the reason in {language_label}.\n"
        "- There is no limit on the amount: include every panel that truly "
        "matches the summary.\n\n"
        "Reply ONLY with valid JSON, no extra text, in the format:\n"
        '{"panels": [{"id": "id1", "excerpt": "...", "reason": "..."}]}'
    )
    if no_think:
        prompt += "\n/no_think"  # Qwen3: disable the "thinking" mode
    return prompt


def parse_summary(reply: str) -> str:
    """Extract the summary from the LLM JSON reply (with fallback)."""
    data = _extract_json(reply or "")
    if data and isinstance(data, dict):
        summary = str(data.get("summary", "")).strip()
        if summary:
            return summary
    return (reply or "").strip()


def parse_panels(reply: str, events: list) -> List[dict]:
    """Extract the chosen panels from the LLM JSON reply (with fallback).

    Returns a list of dicts ``{"id", "excerpt", "reason"}``: ``excerpt`` is
    the part of the summary the panel illustrates and ``reason`` explains
    why it was chosen. There is no limit on the number of panels.
    """
    data = _extract_json(reply or "")
    valid_ids = {event["id"] for event in events}
    panels: List[dict] = []
    if data and isinstance(data, dict):
        for item in data.get("panels", []) or []:
            if isinstance(item, dict):
                panel_id = str(item.get("id", "")).strip()
                excerpt = str(item.get("excerpt", "")).strip()
                reason = str(item.get("reason", "")).strip()
            else:  # legacy format: a list of ids (strings)
                panel_id, excerpt, reason = str(item).strip(), "", ""
            if panel_id in valid_ids:
                panels.append(
                    {"id": panel_id, "excerpt": excerpt, "reason": reason}
                )

    if not panels:  # fallback: every panel with text, in reading order
        panels = [
            {
                "id": event["id"],
                "excerpt": "",
                "reason": "automatic fallback: panel with text",
            }
            for event in events
        ]

    # remove duplicates keeping the order (no limit on the amount)
    seen = set()
    unique = []
    for panel in panels:
        if panel["id"] not in seen:
            seen.add(panel["id"])
            unique.append(panel)
    return unique


def summarize_local(llm, prompt: str) -> str:
    """Summarize through llama-cpp-python (local GGUF model)."""
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_SUMMARY_TOKENS,
        temperature=TEMPERATURE,
    )
    return output["choices"][0]["message"]["content"]


def summarize_local_chapter(
    llm, chapter_text: str, events: list, language: str = DEFAULT_LANGUAGE
):
    """Local backend: 1) summary; 2) panels that match the summary."""
    summary = parse_summary(
        summarize_local(
            llm,
            build_summary_prompt(
                language, chapter_text=chapter_text, no_think=True
            ),
        )
    )
    panels_reply = summarize_local(
        llm,
        build_panels_prompt(
            summary, language, chapter_text=chapter_text, no_think=True
        ),
    )
    return summary, parse_panels(panels_reply, events)


def _opencode_run(
    binary: str,
    prompt: str,
    cwd: Optional[str],
    model: Optional[str] = None,
    agent: Optional[str] = None,
    title: Optional[str] = None,
    session_id: Optional[str] = None,
    attachment: Optional[str] = None,
):
    """Run ``opencode run --format json`` and return ``(reply, session_id)``."""
    command = [binary, "run", "--format", "json"]
    if session_id:  # continue the same session (cost/tokens accumulate)
        command += ["-s", session_id]
    if title:
        command += ["--title", title]
    if model:
        command += ["-m", model]
    if agent:
        command += ["--agent", agent]
    if attachment:
        command += ["-f", attachment]
    command.append(prompt)

    process = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    if process.returncode != 0:
        raise SystemExit(
            f"`{binary} run` failed (code {process.returncode}).\n"
            f"stderr:\n{process.stderr}"
        )

    # the output is NDJSON (one event per line): collect sessionID and texts
    session = session_id
    texts = []
    for line in process.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("sessionID"):
            session = event["sessionID"]
        if event.get("type") == "text":
            text = (event.get("part") or {}).get("text")
            if text:
                texts.append(text)

    reply = "\n".join(texts).strip()
    return (reply or process.stdout), session


def summarize_opencode(
    text_file: str,
    model: Optional[str] = OPENCODE_MODEL,
    binary: str = OPENCODE_BIN,
    agent: Optional[str] = OPENCODE_AGENT,
    cwd: Optional[str] = None,
    language: str = DEFAULT_LANGUAGE,
):
    """OpenCode backend: 1) summary; 2) panels that match the summary.

    Both calls run in the **same** opencode session, so the cost/tokens from
    ``session export`` already cover both of them.

    Returns ``(summary, panels_reply, session_id)``.
    """
    name = os.path.basename(text_file)
    # title = uuid -> avoids the automatic title generation (and the extra
    # call to the "small model", e.g. gpt-6-luna) in opencode.
    title = str(uuid.uuid4())

    print(
        f"[llm] {binary} run (summary) --format json --title {title} ...",
        file=sys.stderr,
    )
    reply, session_id = _opencode_run(
        binary,
        build_summary_prompt(language, filename=name),
        cwd,
        model=model,
        agent=agent,
        title=title,
        attachment=text_file,
    )
    summary = parse_summary(reply)

    print(f"[llm] {binary} run (panels) -s {session_id} ...", file=sys.stderr)
    panels_reply, session_id = _opencode_run(
        binary,
        build_panels_prompt(summary, language, filename=name),
        cwd,
        model=model,
        agent=agent,
        session_id=session_id,
        attachment=text_file,
    )
    return summary, panels_reply, session_id


def opencode_session_cost(
    session_id: str, binary: str = OPENCODE_BIN, cwd: Optional[str] = None
):
    """Return the cost/tokens of an opencode session, or None."""
    if not session_id:
        return None
    process = subprocess.run(
        [binary, "session", "export", session_id],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    if process.returncode != 0:
        return None
    try:
        data = json.loads(process.stdout)
    except json.JSONDecodeError:
        return None
    info = data.get("info", {}) if isinstance(data, dict) else {}
    return {"cost": info.get("cost", 0), "tokens": info.get("tokens", {})}


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _clip(box, width, height, pad):
    """Return ``box`` expanded by ``pad`` and clamped to the image size."""
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def find_chapters(base_dir: str, max_depth: int = 8) -> List[str]:
    """Walk ``base_dir`` recursively and return the folders with images.

    A folder that contains images is treated as a **chapter** (the walk stops
    there). Output folders (``-ai`` suffix) and hidden folders are skipped so
    already generated work is not reprocessed.
    """
    found: List[str] = []

    def walk(directory: str, depth: int) -> None:
        """Recurse into ``directory`` collecting folders that contain images."""
        if depth > max_depth:
            return
        try:
            if list_images(directory):
                found.append(directory)  # reached the image folder
                return
            names = sorted(os.listdir(directory))
        except OSError:
            return
        for name in names:
            if name.startswith(".") or name.endswith(OUTPUT_SUFFIX):
                continue
            subdir = os.path.join(directory, name)
            if os.path.isdir(subdir):
                walk(subdir, depth + 1)

    if os.path.isdir(base_dir):
        walk(base_dir, 0)
    return found


def _build_extractors(kind, device):
    """Create the segmentation/text/OCR models once for a chapter."""
    segmenter = MangaSegmenter(device=device) if kind == "manga" else None
    detector = BubbleTextDetector(device=device)
    ocr = PPOCRv5(device=device)
    gutter_kwargs = {
        "min_gutter": GUTTER_MIN_PX,
        "bg_thresh": BG_THRESH,
        "tol": BG_TOL,
        "min_panel": MIN_PANEL_PX,
    }
    return segmenter, detector, ocr, gutter_kwargs


def _events_for_image(image, panels, results, image_name, stem, out_dir, pad):
    """Crop the panels with text and return ``(events, lines)`` for one image."""
    height, width = image.shape[:2]
    events, lines = [], []
    for result in results:
        panel_index = result["panel"]
        texts = [text["text"] for text in result["texts"] if text["text"]]
        if not texts:
            continue
        x0, y0, x1, y1 = _clip(panels[panel_index]["box"], width, height, pad)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        panel_id = f"{stem}#P{panel_index}"
        crop_rel = f"panels/{panel_id.replace('#', '_')}.jpg"
        cv2.imwrite(os.path.join(out_dir, crop_rel), crop)
        text = " ".join(texts).strip()
        events.append(
            {
                "id": panel_id,
                "img": image_name,
                "panel": panel_index,
                "box": panels[panel_index]["box"],
                "text": text,
                "crop": crop_rel,
            }
        )
        lines.append(f"[{panel_id}] {text}")
    return events, lines


def extract_chapter(
    in_dir: str,
    name: str,
    kind: str,
    device=None,
    pad=PANEL_PAD,
    max_images: int = MAX_IMAGES,
    show_text: bool = True,
):
    """Run the panel-first flow on every image and collect the panel texts.

    ``in_dir`` is the chapter folder (the one with the images); the output
    goes to ``<in_dir>-ai``. Returns ``(events, chapter_text, out_dir)``.
    """
    if not os.path.isdir(in_dir):
        raise SystemExit(f"Folder not found: {in_dir}")
    out_dir = in_dir.rstrip("/\\") + OUTPUT_SUFFIX
    os.makedirs(os.path.join(out_dir, "panels"), exist_ok=True)

    images = list_images(in_dir)
    if max_images and max_images > 0:
        images = images[:max_images]
    print(
        f"[extract] {name} [{kind}] | {len(images)} image(s) | -> {out_dir}",
        file=sys.stderr,
    )

    segmenter, detector, ocr, gutter_kwargs = _build_extractors(kind, device)

    events: List[dict] = []
    lines: List[str] = []
    for index, image_path in enumerate(images, 1):
        image_name = os.path.basename(image_path)
        stem = os.path.splitext(image_name)[0]
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [{index}] FAILED to open {image_name}", file=sys.stderr)
            continue
        panels, results = process_one(
            image,
            kind,
            detector,
            ocr,
            segmenter=segmenter,
            pad=pad,
            gutter_kwargs=gutter_kwargs,
            panel_order="auto",
        )

        # annotated image (optional, helps inspecting the result)
        annotated = draw_annotations(
            image, panels, results, kind, show_text=show_text
        )
        annotated.save(os.path.join(out_dir, image_name))

        image_events, image_lines = _events_for_image(
            image, panels, results, image_name, stem, out_dir, pad
        )
        events.extend(image_events)
        lines.extend(image_lines)
        print(
            f"  [{index}/{len(images)}] {image_name}: {len(panels)} panels | "
            f"{sum(len(result['texts']) for result in results)} texts",
            file=sys.stderr,
        )

    return events, "\n".join(lines), out_dir


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------


def generate_html(
    manga: str,
    kind: str,
    summary: str,
    panels: List[dict],
    events: list,
    out_dir: str,
    language: str = DEFAULT_LANGUAGE,
    info_extra: str = "",
) -> str:
    """Write ``summary.html`` with the summary and the main-panel gallery."""
    strings = HTML_STRINGS[language]
    html_lang = SUPPORTED_LANGUAGES[language]["html_lang"]
    by_id = {event["id"]: event for event in events}
    summary_html = html.escape(summary).replace("\n", "<br>")

    cards = []
    for panel in panels:
        event = by_id.get(panel["id"])
        if not event:
            continue
        reason = (panel.get("reason") or "").strip()
        excerpt = (panel.get("excerpt") or "").strip()
        reason_html = (
            f'<div class="reason"><b>{html.escape(strings["why_panel"])}</b> '
            f"{html.escape(reason)}</div>"
            if reason
            else ""
        )
        excerpt_html = (
            f'<div class="excerpt"><b>{html.escape(strings["in_summary"])}</b> '
            f"&ldquo;{html.escape(excerpt)}&rdquo;</div>"
            if excerpt
            else ""
        )
        cards.append(
            '<figure class="card">'
            f'<img class="zoom" src="{html.escape(event["crop"])}" loading="lazy" '
            f'alt="{html.escape(event["id"])}" '
            f'title="{html.escape(strings["click_to_zoom"])}">'
            f"<figcaption><b>{html.escape(event['id'])}</b> "
            f'<span class="meta">({html.escape(event["img"])} · '
            f'{html.escape(strings["panel"])} {event["panel"]})</span>'
            f"{excerpt_html}"
            f"{reason_html}"
            f'<div class="txt">{html.escape(event["text"])}</div></figcaption>'
            "</figure>"
        )

    document = f"""<!doctype html>
<html lang="{html.escape(html_lang)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(strings["title"])} - {html.escape(manga)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, Segoe UI, Roboto, Arial, sans-serif;
         margin: 0; padding: 24px; background: #f5f5f7; color: #111; }}
  h1 {{ margin: 0 0 4px; }}
  .sub {{ color: #666; margin-bottom: 20px; }}
  .summary {{ background: #fff; border-radius: 12px; padding: 18px 22px;
             line-height: 1.55; box-shadow: 0 1px 4px rgba(0,0,0,.08);
             width: 100%; box-sizing: border-box; }}
  h2 {{ margin: 28px 0 12px; }}
  .grid {{ display: grid; gap: 16px;
           grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }}
  .card {{ margin: 0; background: #fff; border-radius: 12px; overflow: hidden;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); display: flex; flex-direction: column; }}
  .card img {{ width: 100%; height: auto; display: block; background: #ddd;
              cursor: zoom-in; }}
  figcaption {{ padding: 10px 12px; font-size: 13px; }}
  .meta {{ color: #888; font-weight: normal; }}
  .excerpt {{ margin-top: 8px; padding: 8px 10px; border-left: 3px solid #f0a500;
             background: rgba(240,165,0,.10); border-radius: 6px;
             color: #333; line-height: 1.4; font-style: italic; }}
  .reason {{ margin-top: 8px; padding: 8px 10px; border-left: 3px solid #6b8afd;
             background: rgba(107,138,253,.10); border-radius: 6px;
             color: #333; line-height: 1.4; }}
  .txt {{ margin-top: 6px; color: #333; }}
  .footer {{ margin-top: 28px; color: #666; font-size: 13px;
             border-top: 1px solid rgba(128,128,128,.3); padding-top: 12px; }}
  #overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,.92);
             display: none; align-items: center; justify-content: center;
             z-index: 99; cursor: zoom-out; }}
  #overlay.show {{ display: flex; }}
  #overlay img {{ max-width: 98vw; max-height: 96vh; box-shadow: 0 0 40px #000; }}
  #overlay .tip {{ position: absolute; top: 12px; left: 50%;
                   transform: translateX(-50%); color: #eee; font-size: 13px;
                   background: rgba(0,0,0,.5); padding: 4px 10px; border-radius: 8px; }}
  #overlay .nav {{ position: absolute; top: 50%; transform: translateY(-50%);
                  border: 0; background: rgba(0,0,0,.45); color: #fff;
                  font-size: 34px; line-height: 1; padding: 14px 20px;
                  border-radius: 50%; cursor: pointer; user-select: none;
                  font-family: inherit; }}
  #overlay .nav:hover {{ background: rgba(255,255,255,.22); }}
  #overlay .nav.prev {{ left: 18px; }}
  #overlay .nav.next {{ right: 18px; }}
  #overlay .counter {{ position: absolute; bottom: 14px; left: 50%;
                       transform: translateX(-50%); color: #eee; font-size: 13px;
                       background: rgba(0,0,0,.5); padding: 4px 10px;
                       border-radius: 8px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #14161a; color: #eee; }}
    .summary, .card {{ background: #1e2126; box-shadow: none; }}
    .txt {{ color: #cfd3d8; }}
    .reason {{ color: #cfd3d8; }}
    .excerpt {{ color: #cfd3d8; }}
  }}
</style>
</head>
<body>
  <h1>{html.escape(strings["title"])}: {html.escape(manga)}</h1>
  <div class="sub">[{html.escape(kind)}] · {html.escape(strings["generated"])}
    {datetime.now():%Y-%m-%d %H:%M}
    · {len(events)} {html.escape(strings["panels_with_text"])}
    · {len(panels)} {html.escape(strings["main_panels"])}
    {html.escape(strings["matching_summary"])} · {html.escape(strings["click_to_zoom"])}
    {html.escape(strings["zoom_nav_hint"])}</div>
  <div class="summary">{summary_html or f'<i>{html.escape(strings["no_summary"])}</i>'}</div>
  <h2>{html.escape(strings["main_heading"])}</h2>
  <div class="grid">
    {''.join(cards) or f'<i>{html.escape(strings["no_panel"])}</i>'}
  </div>
  {f'<div class="footer">{info_extra}</div>' if info_extra else ''}
  <div id="overlay">
    <span class="tip">{html.escape(strings["zoom_tip"])}</span>
    <button class="nav prev" type="button"
            aria-label="{html.escape(strings["prev_panel"])}">&#10094;</button>
    <img alt="">
    <button class="nav next" type="button"
            aria-label="{html.escape(strings["next_panel"])}">&#10095;</button>
    <span class="counter"></span>
  </div>
  <script>
    const ov = document.getElementById('overlay');
    const ovImg = ov.querySelector('img');
    const counter = ov.querySelector('.counter');
    const imgs = Array.from(document.querySelectorAll('img.zoom'));
    let current = -1;

    function show(i) {{
      if (!imgs.length) return;
      current = (i + imgs.length) % imgs.length;
      ovImg.src = imgs[current].src;
      counter.textContent = (current + 1) + ' / ' + imgs.length;
      ov.classList.add('show');
    }}
    function closeOverlay() {{ ov.classList.remove('show'); }}

    imgs.forEach(function (img, i) {{
      img.addEventListener('click', function () {{ show(i); }});
    }});
    ov.querySelector('.prev').addEventListener('click', function (e) {{
      e.stopPropagation();
      show(current - 1);
    }});
    ov.querySelector('.next').addEventListener('click', function (e) {{
      e.stopPropagation();
      show(current + 1);
    }});
    ov.addEventListener('click', function (e) {{
      if (e.target === ov || e.target === ovImg) closeOverlay();
    }});
    document.addEventListener('keydown', function (e) {{
      if (e.key === 'Escape') {{ closeOverlay(); return; }}
      if (!ov.classList.contains('show')) return;
      if (e.key === 'ArrowLeft') {{ e.preventDefault(); show(current - 1); }}
      if (e.key === 'ArrowRight') {{ e.preventDefault(); show(current + 1); }}
    }});
  </script>
</body>
</html>
"""
    path = os.path.join(out_dir, "summary.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return path


# ---------------------------------------------------------------------------
# Chapter processing
# ---------------------------------------------------------------------------


def summarize_chapter(args, llm, events, chapter_text, text_file, name):
    """Summarize one chapter (2 stages) and return ``(summary, panels, session)``."""
    if args.llm == "opencode":
        print(
            f"[summary] {name} | {len(events)} panels with text "
            f"| backend=opencode "
            f"| model={args.opencode_model or '(opencode default)'}",
            file=sys.stderr,
        )
        summary, panels_reply, session_id = summarize_opencode(
            text_file,
            model=args.opencode_model,
            binary=args.opencode_bin,
            agent=args.opencode_agent,
            cwd=SCRIPT_DIR,
            language=args.language,
        )
        return summary, parse_panels(panels_reply, events), session_id

    print(
        f"[summary] {name} | {len(events)} panels with text "
        f"| backend=local | model={args.model}",
        file=sys.stderr,
    )
    summary, panels = summarize_local_chapter(
        llm, chapter_text, events, args.language
    )
    return summary, panels, None


def opencode_cost_text(args, session_id) -> str:
    """Return the formatted opencode cost/token footer (or an empty string)."""
    if args.llm != "opencode":
        return ""
    info = opencode_session_cost(
        session_id, binary=args.opencode_bin, cwd=SCRIPT_DIR
    )
    if not info:
        print(
            "\n[cost] could not obtain the session cost.",
            file=sys.stderr,
        )
        return ""
    tokens = info.get("tokens") or {}
    cache = tokens.get("cache") or {}
    text = (
        f"Call cost (opencode): US$ {float(info.get('cost', 0)):.6f} · "
        f"tokens: in={tokens.get('input', 0)} out={tokens.get('output', 0)} "
        f"reasoning={tokens.get('reasoning', 0)} "
        f"cache_read={cache.get('read', 0)} cache_write={cache.get('write', 0)} "
        f"· session {session_id}"
    )
    print("\n" + text)
    return text


def write_summary_txt(out_dir, summary, panels, info_extra):
    """Write ``summary.txt`` with the summary and the main-panel list."""
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(summary + "\n\nMain panels (matching the summary):\n")
        for panel in panels:
            excerpt = (panel.get("excerpt") or "").strip()
            reason = (panel.get("reason") or "").strip()
            extra = f' [summary: "{excerpt}"]' if excerpt else ""
            if reason:
                extra += f" :: {reason}"
            handle.write(f"- {panel['id']}{extra}\n")
        if info_extra:
            handle.write("\n" + info_extra + "\n")
    return path


def print_result(name, summary, panels, html_path, out_dir):
    """Print the summary and the main-panel list for one chapter."""
    print(f"\n===== SUMMARY ({name}) =====\n")
    print(summary)
    print("\nMain panels (matching the summary):")
    for panel in panels:
        excerpt = (panel.get("excerpt") or "").strip()
        reason = (panel.get("reason") or "").strip()
        extra = f' [summary: "{excerpt}"]' if excerpt else ""
        if reason:
            extra += f" :: {reason}"
        print(f"  - {panel['id']}{extra}")
    print(f"\n# HTML   : {html_path}")
    print(f"# folder : {out_dir}")


def process_chapter(chapter_dir, args, llm, device):
    """Run the whole pipeline for one chapter; return ``(name, html, out_dir)``."""
    name = os.path.relpath(chapter_dir, SCRIPT_DIR)
    kind = args.kind
    events, chapter_text, out_dir = extract_chapter(
        chapter_dir,
        name,
        kind,
        device=device,
        pad=args.pad,
        max_images=args.max_images,
    )
    if not events:
        print(f"[{name}] no text extracted; skipping.", file=sys.stderr)
        return None

    # save the chapter text (used as an attachment by the opencode backend)
    text_file = os.path.join(out_dir, "_chapter_texts.txt")
    with open(text_file, "w", encoding="utf-8") as handle:
        handle.write(chapter_text + "\n")

    summary, panels, session_id = summarize_chapter(
        args, llm, events, chapter_text, text_file, name
    )
    info_extra = opencode_cost_text(args, session_id)

    html_path = generate_html(
        name,
        kind,
        summary,
        panels,
        events,
        out_dir,
        language=args.language,
        info_extra=info_extra,
    )
    write_summary_txt(out_dir, summary, panels, info_extra)
    print_result(name, summary, panels, html_path, out_dir)
    return name, html_path, out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Scan the folder of the chosen kind, find the folders with "
        "images (the chapters), extract the texts, summarize through a LLM "
        "(local or opencode) and open the result."
    )
    parser.add_argument(
        "--kind",
        dest="kind",
        default="manga",
        choices=["manga", "webtoon"],
        help="reading kind; defines which folder is scanned "
        "(manga -> mangas/, webtoon -> webtoons/)",
    )
    parser.add_argument(
        "--language",
        dest="language",
        default=DEFAULT_LANGUAGE,
        choices=list(SUPPORTED_LANGUAGES),
        help="language of the AI reply and of the generated HTML page "
        "(default: english)",
    )
    parser.add_argument(
        "--llm",
        dest="llm",
        default=LLM_BACKEND,
        choices=["local", "opencode"],
        help="LLM backend: 'local' (llama-cpp) or 'opencode' (CLI)",
    )
    # --- local (llama-cpp-python) ---
    parser.add_argument(
        "--model",
        dest="model",
        default=MODEL_REPO,
        help="GGUF repo on HuggingFace (local backend)",
    )
    parser.add_argument(
        "--model-file",
        dest="model_file",
        default=MODEL_FILE,
        help="specific .gguf file (local backend)",
    )
    # --- opencode CLI ---
    parser.add_argument(
        "--opencode-model",
        dest="opencode_model",
        default=OPENCODE_MODEL,
        help="opencode model: provider/model[#variant]",
    )
    parser.add_argument(
        "--opencode-bin",
        dest="opencode_bin",
        default=OPENCODE_BIN,
        help="opencode binary",
    )
    parser.add_argument(
        "--opencode-agent",
        dest="opencode_agent",
        default=OPENCODE_AGENT,
        help="opencode agent",
    )
    # --- general ---
    parser.add_argument(
        "--device",
        dest="device",
        default=None,
        help="cpu or cuda; otherwise it is detected automatically",
    )
    parser.add_argument(
        "--pad",
        dest="pad",
        type=int,
        default=PANEL_PAD,
        help="panel padding (px)",
    )
    parser.add_argument(
        "--max",
        dest="max_images",
        type=int,
        default=MAX_IMAGES,
        help="0 = every image; otherwise at most N",
    )
    return parser


def resolve_device(args) -> Tuple[Optional[str], int]:
    """Resolve the torch/OCR device and the number of GPU layers from ``args``."""
    device = args.device
    if device == "cpu":
        n_gpu_layers = 0
    elif device == "cuda":
        n_gpu_layers = -1
    else:  # auto: offload to the GPU only when CUDA is really available
        n_gpu_layers = -1 if cuda_available() else 0
    return device, n_gpu_layers


def discover_chapters(args) -> List[str]:
    """List the chapter folders to process and print the scan report."""
    rel_base = ROOTS[args.kind]
    base_dir = os.path.abspath(os.path.join(SCRIPT_DIR, rel_base))
    if not os.path.isdir(base_dir):
        raise SystemExit(f"Folder not found: {base_dir}")

    chapters = find_chapters(base_dir)
    if not chapters:
        raise SystemExit(
            f"No folder with images found in {rel_base} (extensions: {EXTENSIONS})"
        )
    print(
        f"[scan] {len(chapters)} chapter(s) with images in {rel_base}:",
        file=sys.stderr,
    )
    for chapter in chapters:
        print(f"  - {os.path.relpath(chapter, SCRIPT_DIR)}", file=sys.stderr)
    return chapters


def main(argv=None) -> int:
    """Entry point: parse the args, process every chapter and open the result."""
    args = build_arg_parser().parse_args(argv)
    device, n_gpu_layers = resolve_device(args)
    chapters = discover_chapters(args)

    # local backend: load the model ONCE and reuse it across chapters
    llm = None
    if args.llm == "local":
        print(
            f"[summary] backend=local | model={args.model} | "
            f"language={args.language}",
            file=sys.stderr,
        )
        llm = load_llm(args.model, args.model_file, n_gpu_layers=n_gpu_layers)

    generated: List[Tuple[str, str, str]] = []  # (name, html, out_dir)
    for chapter_dir in chapters:
        result = process_chapter(chapter_dir, args, llm, device)
        if result:
            generated.append(result)

    if not generated:
        print("Nothing was generated.", file=sys.stderr)
        return 1

    print("\n===== GENERATED FILES =====")
    for name, html_path, _ in generated:
        print(f"- {name}: {html_path}")

    for _, html_path, _ in generated:
        webbrowser.open("file://" + os.path.abspath(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
