import io
import os

import numpy as np
import torch
from flask import Flask, request, send_file, jsonify
from PIL import Image

import config
from tesunet import ResNet50UNet, neighborhood_fusion
from inference import process_full_image, get_inference_transform
from resnet50_cam import CAM as CamExtractor
from resnet50_irn import EdgeDisplacement, AffinityDisplacementLoss
from path_index import PathIndex
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, config.OUTPUT_DIR)
DEVICE = config.DEVICE


# Global models (lazy loaded)
_cam_net = None
_irnet = None
_unet_cam_only = None
_unet_cam_irn = None
_inference_transform = None


def _load_stage4_models():
    global _cam_net, _irnet, _inference_transform
    if _cam_net is not None and _irnet is not None and _inference_transform is not None:
        return

    device = DEVICE

    # ---- Load CAM network ----
    cam_net = CamExtractor().to(device)
    cam_path = os.path.join(OUTPUT_DIR, "cam_net_best.pth")
    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"CAM checkpoint not found: {cam_path}")
    checkpoint_cam = torch.load(cam_path, map_location=device, weights_only=False)
    cam_net.load_state_dict(checkpoint_cam["model_state_dict"])
    cam_net.eval()

    # ---- Load IRNet (EdgeDisplacement) ----
    # Match training feature-map size used in train_stage2_3
    irn_patch_size = 256
    irn_feat_h = irn_patch_size // 4
    irn_feat_w = irn_patch_size // 4

    path_idx = PathIndex(feat_h=irn_feat_h, feat_w=irn_feat_w, radius=config.RADIUS)
    irnet_temp = AffinityDisplacementLoss(path_idx).to(device)

    irn_path = os.path.join(OUTPUT_DIR, "irnet_best.pth")
    if not os.path.exists(irn_path):
        raise FileNotFoundError(f"IRNet checkpoint not found: {irn_path}")
    checkpoint_irn = torch.load(irn_path, map_location=device, weights_only=False)
    irnet_temp.load_state_dict(checkpoint_irn["model_state_dict"])

    irnet = EdgeDisplacement(crop_size=config.INFERENCE_PATCH_SIZE).to(device)
    # Filter out buffers not present in EdgeDisplacement
    state_dict = checkpoint_irn["model_state_dict"]
    filtered_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("path_indices") and k != "disp_target"
    }
    irnet.load_state_dict(filtered_state_dict, strict=False)
    irnet.eval()

    # Inference transform
    transform = get_inference_transform()

    _cam_net = cam_net
    _irnet = irnet
    _inference_transform = transform


def _load_stage5_models():
    global _unet_cam_only, _unet_cam_irn
    if _unet_cam_only is not None and _unet_cam_irn is not None:
        return

    device = DEVICE

    # Model trained with CAM-only pseudo labels
    cam_only_dir = os.path.join(OUTPUT_DIR, "stage5_unet")
    cam_only_ckpt = os.path.join(cam_only_dir, "best_model.pth")
    if not os.path.exists(cam_only_ckpt):
        raise FileNotFoundError(f"Stage5 UNet (CAM-only) checkpoint not found: {cam_only_ckpt}")

    unet_cam_only = ResNet50UNet(num_classes=2, pretrained=False).to(device)
    ckpt_cam_only = torch.load(cam_only_ckpt, map_location=device)
    unet_cam_only.load_state_dict(ckpt_cam_only["model_state_dict"])
    unet_cam_only.eval()

    # Model trained with CAM+IRN pseudo labels
    cam_irn_dir = os.path.join(OUTPUT_DIR, "stage5_unet_irn")
    cam_irn_ckpt = os.path.join(cam_irn_dir, "best_model.pth")
    if not os.path.exists(cam_irn_ckpt):
        raise FileNotFoundError(f"Stage5 UNet (CAM+IRN) checkpoint not found: {cam_irn_ckpt}")

    unet_cam_irn = ResNet50UNet(num_classes=2, pretrained=False).to(device)
    ckpt_cam_irn = torch.load(cam_irn_ckpt, map_location=device)
    unet_cam_irn.load_state_dict(ckpt_cam_irn["model_state_dict"])
    unet_cam_irn.eval()

    _unet_cam_only = unet_cam_only
    _unet_cam_irn = unet_cam_irn


def _read_image(file) -> np.ndarray:
    """Read an uploaded image (Flask FileStorage) into an RGB numpy array."""
    data = file.read()
    if not data:
        raise ValueError("Empty file")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")
    return np.array(img)


def _mask_to_png_bytes(mask: np.ndarray) -> io.BytesIO:
    # Expect binary mask 0/255 or 0/1
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.route("/stage5/unet", methods=["POST"])
def predict_stage5_unet():
    """Run Stage 5 ResNet50-UNet on a single image.

    variant="cam_only"  -> outputs/stage5_unet/best_model.pth
    variant="cam_irn"   -> outputs/stage5_unet_irn/best_model.pth
    Returns: PNG mask (binary) as image/png.
    """
    # Get file & params from Flask request
    file = request.files.get("image")
    variant = request.form.get("variant", "cam_irn")

    if file is None:
        return jsonify({"detail": "No file uploaded under field 'image'"}), 400

    try:
        _load_stage5_models()
    except FileNotFoundError as e:
        return jsonify({"detail": str(e)}), 500

    try:
        img_np = _read_image(file)
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400

    if variant not in {"cam_only", "cam_irn"}:
        return jsonify({"detail": "variant must be 'cam_only' or 'cam_irn'"}), 400

    device = DEVICE
    model = _unet_cam_only if variant == "cam_only" else _unet_cam_irn

    with torch.no_grad():
        pred_mask, prob_map = neighborhood_fusion(
            model,
            img_np,
            device,
            patch_size=config.STAGE5_PATCH_SIZE,
            step=80,
            batch_size=max(1, config.STAGE5_BATCH_SIZE // 2),
        )

    mask_u8 = (pred_mask * 255).astype(np.uint8)
    buf = _mask_to_png_bytes(mask_u8)
    return send_file(buf, mimetype="image/png")


@app.route("/inference/pseudo", methods=["POST"])
def predict_pseudo_label():
    """Run Stage 4 pseudo-label inference on a single image.

    mode="cam_only" -> CAM-only pseudo label
    mode="cam_irn"  -> CAM+IRN pseudo label
    Returns: PNG mask (binary) as image/png.
    """
    file = request.files.get("image")
    mode = request.form.get("mode", "cam_irn")

    if file is None:
        return jsonify({"detail": "No file uploaded under field 'image'"}), 400

    try:
        _load_stage4_models()
    except FileNotFoundError as e:
        return jsonify({"detail": str(e)}), 500

    try:
        img_np = _read_image(file)
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400

    if mode not in {"cam_only", "cam_irn"}:
        return jsonify({"detail": "mode must be 'cam_only' or 'cam_irn'"}), 400

    with torch.no_grad():
        cam, B, D, cam_only, pseudo = process_full_image(
            img_np,
            _cam_net,
            _irnet,
            _inference_transform,
            DEVICE,
            patch_size=config.INFERENCE_PATCH_SIZE,
            stride=config.INFERENCE_STRIDE,
        )

    out_mask = cam_only if mode == "cam_only" else pseudo

    buf = _mask_to_png_bytes(out_mask.astype(np.uint8))
    return send_file(buf, mimetype="image/png")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Run with: python api.py
    app.run(host="0.0.0.0", port=8000, debug=False)
