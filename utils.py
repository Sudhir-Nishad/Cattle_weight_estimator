"""
utils.py
==============================================================================
Model loading, YOLO+KeypointRCNN inference, trait/weight computation, and
annotation drawing for the Cattle Weight Estimation Streamlit app.

Only the SIDE-VIEW is used:
  - Segmentation model (best.pt)      -> locates cow + calibration sticker
  - Side KeypointRCNN (best_model_side.pth) -> locates 15 side-view keypoints

Weight formula (Schaeffer's girth-length formula, metric):
    weight_kg = (heart_girth_cm ** 2 * body_length_cm) / 10840

------------------------------------------------------------------------------
CHANGES FROM THE ORIGINAL VERSION (see comments tagged "# CHANGED:")
------------------------------------------------------------------------------
1. Google Drive / gdown downloading has been removed from the default path.
   Google Drive is not a reliable production model host (no resumable
   downloads, virus-scan interstitials on large files, rate limits) and,
   combined with Render's ephemeral filesystem, it was causing the model to
   re-download on every container restart -> slow boot -> Render proxy
   502s -> restart loop.

2. Models are now expected to be baked directly into the Docker image at
   MODEL_DIR (default "models/"). This means zero network calls at
   startup, the fastest possible cold start, and nothing to re-download
   ever, because the files are already on disk the instant the container
   boots.

3. An OPTIONAL fallback remains for teams that don't want to rebuild the
   Docker image every time the model changes: if the baked-in files are
   missing, and MODEL_CACHE_DIR + HF_MODEL_REPO env vars are set, models
   are pulled once from a Hugging Face Hub model repo (resumable,
   versioned, no interstitials) into MODEL_CACHE_DIR. For this fallback to
   actually avoid repeated downloads on Render, MODEL_CACHE_DIR MUST point
   at a mounted Render Persistent Disk (e.g. /var/data/models) — otherwise
   you are back to the exact same ephemeral-storage problem, just with a
   nicer download client.

4. load_models() / build_keypoint_rcnn() signatures are UNCHANGED, so
   app.py's `load_models(progress_callback=progress)` call keeps working
   with no edits required there.
==============================================================================
"""

import os
import math

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────
# Model file locations
# ─────────────────────────────────────────────────────────────────────────
# CHANGED: MODEL_DIR is where the app looks FIRST — this is meant to be the
# path the Docker image bakes the checkpoints into. Override via env var
# only if you truly need to (e.g. local dev with a different layout).
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# CHANGED: MODEL_CACHE_DIR is the fallback download target. Point this at a
# mounted Render Persistent Disk (e.g. "/var/data/models") if you use the
# Hugging Face Hub fallback below. If unset, it defaults to MODEL_DIR, which
# is fine ONLY if MODEL_DIR itself is a persistent/baked-in location.
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", MODEL_DIR)

# CHANGED: optional Hugging Face Hub repo id, e.g. "your-username/cattle-weight-models"
# Only used if the baked-in files are not found. Leave unset if you bake
# models into the image (recommended).
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_SEG_FILENAME = os.environ.get("HF_SEG_FILENAME", "best.pt")
HF_SIDE_FILENAME = os.environ.get("HF_SIDE_FILENAME", "best_model_side.pth")

SEG_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
SIDE_RESNET_PATH = os.path.join(MODEL_DIR, "best_model_side.pth")

STICKER_CM_DEFAULT = 15.0

# ─────────────────────────────────────────────────────────────────────────
# Side-view keypoint schema (must match training order exactly)
# ─────────────────────────────────────────────────────────────────────────
SIDE_KP_NAMES = [
    "wither", "foot", "chest_top", "chest_bottom", "body_top", "body_bottom",
    "hip_bone", "pin_bone", "shoulder_bone", "stifle_thigh", "hock",
    "pastern", "hoof_tip", "heel_bulb", "pastern_hoof_junction",
]
SKP = {n: i for i, n in enumerate(SIDE_KP_NAMES)}


# ==============================================================================
# Model resolution: baked-in first, optional HF Hub fallback second
# ==============================================================================
def _hf_hub_fallback_download(filename, progress_callback=None):
    """
    Only invoked if a required model file is missing from MODEL_DIR.
    Downloads once from Hugging Face Hub into MODEL_CACHE_DIR (which should
    be a persistent disk mount on Render) and returns the local path.
    Uses huggingface_hub's built-in local caching + resumable download, so
    re-running this after the first successful download is a fast no-op
    (it checks ETags rather than re-downloading the whole file).
    """
    if not HF_MODEL_REPO:
        raise FileNotFoundError(
            f"Model file '{filename}' was not found in '{MODEL_DIR}', and no "
            f"HF_MODEL_REPO fallback is configured. Bake the model into the "
            f"Docker image at that path, or set HF_MODEL_REPO (and ideally "
            f"MODEL_CACHE_DIR pointed at a Render Persistent Disk)."
        )
    from huggingface_hub import hf_hub_download

    if progress_callback:
        progress_callback(f"Downloading {filename} from Hugging Face Hub (first run only)...")

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=filename,
        local_dir=MODEL_CACHE_DIR,
    )
    return local_path


def ensure_models_downloaded(progress_callback=None):
    """
    Resolves paths to the segmentation and keypoint model files.

    Order of resolution:
      1. Baked-in / already-cached file at MODEL_DIR — no network call.
      2. Hugging Face Hub fallback (only if configured) — downloads once
         into MODEL_CACHE_DIR, which should be a persistent disk.

    Google Drive downloading has been intentionally removed (see module
    docstring for why).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    seg_path = SEG_MODEL_PATH
    if not os.path.exists(seg_path):
        seg_path = _hf_hub_fallback_download(HF_SEG_FILENAME, progress_callback)

    side_path = SIDE_RESNET_PATH
    if not os.path.exists(side_path):
        side_path = _hf_hub_fallback_download(HF_SIDE_FILENAME, progress_callback)

    return seg_path, side_path


def build_keypoint_rcnn(num_keypoints, min_size, max_size, weights_path, device):
    from torchvision.models.detection import KeypointRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    backbone = resnet_fpn_backbone("resnet101", weights=None)
    model = KeypointRCNN(
        backbone, num_classes=2, num_keypoints=num_keypoints,
        min_size=min_size, max_size=max_size,
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_models(progress_callback=None):
    from ultralytics import YOLO

    seg_path, resnet_path = ensure_models_downloaded(progress_callback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if progress_callback:
        progress_callback("Loading segmentation model...")
    yolo_model = YOLO(seg_path)

    if progress_callback:
        progress_callback("Loading keypoint model...")
    resnet_model = build_keypoint_rcnn(len(SIDE_KP_NAMES), 640, 1024, resnet_path, device)

    return yolo_model, resnet_model, device


# ==============================================================================
# Sticker calibration geometry
# ==============================================================================
def _order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _contour_to_quad(contour):
    peri = cv2.arcLength(contour, True)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return _order_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(contour)
    return _order_points(cv2.boxPoints(rect))


def _sticker_px_from_corners(corners):
    sides = [
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[1] - corners[2]),
        np.linalg.norm(corners[2] - corners[3]),
        np.linalg.norm(corners[3] - corners[0]),
    ]
    return float(max(sides)), [float(s) for s in sides]


# ==============================================================================
# Inference
# ==============================================================================
def run_side_inference(yolo_model, resnet_model, img_bgr, device,
                        sticker_cm=STICKER_CM_DEFAULT, score_thresh=0.3):
    """
    Returns a dict with either:
      {"error": "..."}                                     on failure, or
      {"pred_kps", "box", "cmp", "corners"}                 on success
    """
    h_orig, w_orig = img_bgr.shape[:2]

    results = yolo_model(img_bgr, imgsz=1024, retina_masks=True, verbose=False)[0]
    if results.masks is None or len(results.boxes) == 0:
        return {"error": "No cow or calibration sticker detected in the image."}

    classes = results.boxes.cls.cpu().numpy().astype(int)
    masks_tensor = results.masks.data

    combined_mask_tensor = torch.any(masks_tensor > 0.5, dim=0)
    mask_binary = (combined_mask_tensor.cpu().numpy().astype(np.uint8)) * 255
    if mask_binary.shape[:2] != (h_orig, w_orig):
        mask_binary = cv2.resize(mask_binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # ── Sticker calibration ────────────────────────────────────────────
    sticker_indices = np.where(classes == 1)[0]
    corners = None
    cmp = None

    if len(sticker_indices) > 0:
        st_idx = sticker_indices[0]
        st_mask = (masks_tensor[st_idx].cpu().numpy().astype(np.uint8)) * 255
        if st_mask.shape[:2] != (h_orig, w_orig):
            st_mask = cv2.resize(st_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(st_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            sticker_contour = max(cnts, key=cv2.contourArea)
            corners = _contour_to_quad(sticker_contour)
            sticker_px, _ = _sticker_px_from_corners(corners)
            cmp = sticker_cm / sticker_px

    # Fallback: QR-code style detection
    if corners is None:
        det = cv2.QRCodeDetector()
        _, bbox, _ = det.detectAndDecode(img_bgr)
        if bbox is not None:
            corners = _order_points(bbox[0].astype(np.float32))
            sticker_px, _ = _sticker_px_from_corners(corners)
            cmp = sticker_cm / sticker_px

    if cmp is None:
        return {"error": "Calibration sticker not detected. Please retake the photo with the sticker clearly visible."}

    # ── Keypoint localization ──────────────────────────────────────────
    yellow_bg = np.zeros_like(img_bgr)
    yellow_bg[:] = [0, 255, 255]
    masked_canvas = np.where(mask_binary[:, :, None] == 255, img_bgr, yellow_bg)

    resnet_input = TF.to_tensor(
        Image.fromarray(cv2.cvtColor(masked_canvas, cv2.COLOR_BGR2RGB))
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = resnet_model(resnet_input)

    if not predictions or len(predictions) == 0 or "keypoints" not in predictions[0]:
        return {"error": "No cattle keypoints detected in the image."}

    pred = predictions[0]
    scores = pred["scores"].cpu().numpy()
    if len(scores) == 0 or scores[0] < score_thresh:
        return {"error": "No cattle keypoints detected with sufficient confidence. Try a clearer, unobstructed side-view photo."}

    best_idx = int(np.argmax(scores))
    pred_kps = pred["keypoints"][best_idx].cpu().numpy()
    box = pred["boxes"][best_idx].cpu().numpy()

    return {"pred_kps": pred_kps, "box": box, "cmp": cmp, "corners": corners}


# ==============================================================================
# Trait + weight computation
# ==============================================================================
def compute_weight_traits(pred_kps, cmp):
    """
    Computes:
      linear_body_depth_cm, linear_chest_height_cm, body_length_cm,
      heart_girth_cm, weight_kg
    """
    kps = np.array(pred_kps, dtype=np.float32)

    def p(name):
        return kps[SKP[name]][:2]

    def vis(name):
        return float(kps[SKP[name]][2]) > 0

    def safe_dist(n1, n2):
        if vis(n1) and vis(n2):
            return float(np.linalg.norm(p(n1) - p(n2)))
        return None

    body_length_px = safe_dist("shoulder_bone", "pin_bone")
    chest_height_px = safe_dist("chest_top", "chest_bottom")
    linear_body_depth_px = safe_dist("body_top", "body_bottom")

    linear_chest_height_cm = chest_height_px * cmp if chest_height_px else None
    linear_body_depth_cm = linear_body_depth_px * cmp if linear_body_depth_px else None
    body_length_raw_cm = body_length_px * cmp if body_length_px else None

    body_length_cm = None
    if body_length_raw_cm is not None and linear_body_depth_cm is not None:
        body_length_cm = -32.922 + 1.1758 * body_length_raw_cm + 0.3868 * linear_body_depth_cm
    heart_girth_cm = None
    if linear_chest_height_cm is not None:
        heart_girth_cm = 1.588 * linear_chest_height_cm + 73.43

    weight_kg = None
    if body_length_cm is not None and heart_girth_cm is not None:
        weight_kg = (heart_girth_cm * heart_girth_cm * body_length_cm) / 10840.0

    missing = []
    if linear_body_depth_cm is None:
        missing.append("body_top / body_bottom")
    if linear_chest_height_cm is None:
        missing.append("chest_top / chest_bottom")
    if body_length_px is None:
        missing.append("shoulder_bone / pin_bone")

    return {
        "linear_body_depth_cm": linear_body_depth_cm,
        "linear_chest_height_cm": linear_chest_height_cm,
        "body_length_cm": body_length_cm,
        "heart_girth_cm": heart_girth_cm,
        "weight_kg": weight_kg,
        "missing_keypoints": missing,
    }


# ==============================================================================
# Annotation (Body Length + Heart Girth ONLY)
# ==============================================================================
BRAND_BGR = (102, 40, 42)     # #2A2866 in BGR
LENGTH_COLOR_BGR = (80, 175, 76)   # green
GIRTH_COLOR_BGR = (0, 140, 255)    # orange


def draw_weight_annotation(img_bgr, pred_kps, traits, tag_id=None):
    out = img_bgr.copy()
    kps = np.array(pred_kps, dtype=np.float32)
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.6, w / 1500)
    tk = max(2, w // 450)

    def kp_pt(name):
        return (int(kps[SKP[name]][0]), int(kps[SKP[name]][1]))

    def vis(name):
        return float(kps[SKP[name]][2]) > 0

    # Body length line: shoulder_bone <-> pin_bone
    if vis("shoulder_bone") and vis("pin_bone"):
        p1, p2 = kp_pt("shoulder_bone"), kp_pt("pin_bone")
        cv2.line(out, p1, p2, LENGTH_COLOR_BGR, tk)
        for pt in (p1, p2):
            cv2.circle(out, pt, max(6, w // 130), BRAND_BGR, -1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 18)
        label = (f"Body Length: {traits['body_length_cm']:.1f} cm"
                  if traits.get("body_length_cm") else "Body Length: N/A")
        cv2.putText(out, label, mid, font, fs, (255, 255, 255), tk + 3)
        cv2.putText(out, label, mid, font, fs, LENGTH_COLOR_BGR, tk)

    # Heart girth reference line: chest_top <-> chest_bottom
    if vis("chest_top") and vis("chest_bottom"):
        p1, p2 = kp_pt("chest_top"), kp_pt("chest_bottom")
        cv2.line(out, p1, p2, GIRTH_COLOR_BGR, tk)
        for pt in (p1, p2):
            cv2.circle(out, pt, max(6, w // 130), BRAND_BGR, -1)
        mid = (p1[0] + 20, (p1[1] + p2[1]) // 2)
        label = (f"Heart Girth: {traits['heart_girth_cm']:.1f} cm"
                  if traits.get("heart_girth_cm") else "Heart Girth: N/A")
        cv2.putText(out, label, mid, font, fs, (255, 255, 255), tk + 3)
        cv2.putText(out, label, mid, font, fs, GIRTH_COLOR_BGR, tk)

    if tag_id:
        tag_label = f"Tag ID: {tag_id}"
        cv2.putText(out, tag_label, (15, 40), font, fs * 0.9, (255, 255, 255), tk + 3)
        cv2.putText(out, tag_label, (15, 40), font, fs * 0.9, BRAND_BGR, tk)

    return out
