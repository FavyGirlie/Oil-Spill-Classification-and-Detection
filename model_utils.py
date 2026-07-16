"""
model_utils.py
----------------
Loads the two trained oil-spill segmentation models (SegFormer-B2 and
Mask2Former/Swin-Small) from the Hugging Face Hub and runs inference on a
single uploaded image.

The checkpoints are NOT stored in this repo. They are downloaded once (and
cached) from a Hugging Face model repo at app startup. This keeps the GitHub
repo small and avoids GitHub's 100MB file-size limit.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
import streamlit as st
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

HF_REPO_ID = "De-FavouredOne/oil-spill-segformer-mask2former"  
SEGFORMER_FILENAME = "segformer_lados_best.pth"
MASK2FORMER_FILENAME = "mask2former_lados_best.pth"

SEG_CKPT = "nvidia/mit-b2"
M2F_CKPT = "facebook/mask2former-swin-small-ade-semantic"  

IMG_SIZE = 512

LADOS_ID2LABEL = {
    0: "Background",
    1: "Oil",
    2: "Emulsion",
    3: "Sheen",
    4: "Ship",
    5: "Oil-platform",
}
LADOS_NUM_CLASSES = len(LADOS_ID2LABEL)

LADOS_PALETTE = {
    0: (200, 200, 200),  # Background — light grey
    1: (30, 30, 30),     # Oil — near-black
    2: (139, 90, 43),    # Emulsion — brown
    3: (173, 216, 230),  # Sheen — light blue
    4: (220, 50, 50),    # Ship — red
    5: (255, 165, 0),    # Oil-platform — orange
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download_checkpoint(filename: str) -> str:
    """Downloads (and caches) a checkpoint file from the HF Hub. Returns local path."""
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename
                          token=st.secrets.get("HF_TOKEN", None)


def load_segformer():
    """Builds SegFormer-B2 architecture and loads trained weights."""
    config = SegformerConfig.from_pretrained(SEG_CKPT, num_labels=LADOS_NUM_CLASSES)
    model = SegformerForSemanticSegmentation(config).to(DEVICE)

    ckpt_path = _download_checkpoint(SEGFORMER_FILENAME)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    
    clean_state_dict = {
        (k[6:] if k.startswith("model.") else k): v for k, v in checkpoint.items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    processor = SegformerImageProcessor.from_pretrained(
        SEG_CKPT, do_resize=True, size={"height": IMG_SIZE, "width": IMG_SIZE}
    )
    return model, processor


def load_mask2former():
    """Builds Mask2Former (Swin-Small) architecture and loads trained weights.

    NOTE: must match whatever architecture was actually trained. The training
    notebook was fixed to standardise on Swin-Small to match this loader.
    If you trained with Swin-Base instead, change M2F_CKPT above to
    'facebook/mask2former-swin-base-IN21k-ade-semantic' and remove the manual
    backbone_config overrides below.
    """
    config = Mask2FormerConfig.from_pretrained(
        M2F_CKPT, num_labels=LADOS_NUM_CLASSES, ignore_mismatched_sizes=True
    )
    # Swin-Small dimensions
    config.backbone_config.embed_dim = 128
    config.backbone_config.depths = [2, 2, 18, 2]
    config.backbone_config.num_heads = [4, 8, 16, 32]
    config.backbone_config.window_size = 12

    model = Mask2FormerForUniversalSegmentation(config).to(DEVICE)

    ckpt_path = _download_checkpoint(MASK2FORMER_FILENAME)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    processor = Mask2FormerImageProcessor.from_pretrained(
        M2F_CKPT,
        do_resize=True,
        size={"height": IMG_SIZE, "width": IMG_SIZE},
        ignore_index=255,
    )
    return model, processor


def mask_to_rgb(mask_array: np.ndarray, palette: dict) -> np.ndarray:
    """Converts a (H, W) class-id mask into an (H, W, 3) RGB image."""
    h, w = mask_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in palette.items():
        rgb[mask_array == cls_id] = colour
    return rgb


@torch.no_grad()
def run_inference(model, processor, image: Image.Image, model_type: str):
    """
    Runs a single PIL image through the given model.

    Returns:
        pred_mask: (H, W) numpy array of predicted class IDs, resized to the
                   original image dimensions.
    """
    orig_w, orig_h = image.size
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    outputs = model(**inputs)

    if model_type == "segformer":
        logits = F.interpolate(
            outputs.logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
    else:  # mask2former
        seg_maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(orig_h, orig_w)]
        )
        pred_mask = seg_maps[0].cpu().numpy()

    return pred_mask


def overlay_mask_on_image(image: Image.Image, mask_rgb: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blends the coloured prediction mask over the original image."""
    image_arr = np.array(image.convert("RGB")).astype(np.float32)
    mask_arr = mask_rgb.astype(np.float32)
    blended = (1 - alpha) * image_arr + alpha * mask_arr
    return Image.fromarray(blended.astype(np.uint8))


def class_pixel_breakdown(pred_mask: np.ndarray, id2label: dict) -> dict:
    """Returns {class_name: percentage_of_image} for the predicted mask."""
    total = pred_mask.size
    breakdown = {}
    for cls_id, name in id2label.items():
        pct = float((pred_mask == cls_id).sum()) / total * 100
        breakdown[name] = round(pct, 2)
    return breakdown
