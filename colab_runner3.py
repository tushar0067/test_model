import os
import sys
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

print("🚀 STARTING PROMPT SEEK ENGINE (DINO + SAM2)", flush=True)

# ============================================================
# 1. DEVICE SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using Device: {device}", flush=True)

if device.type == "cuda":
    print("🔥 GPU:", torch.cuda.get_device_name(0), flush=True)
    torch.backends.cudnn.benchmark = True

# ============================================================
# 2. PIN TRANSFORMERS — always runs, fixes DINO compatibility
# ============================================================
print("📌 Pinning transformers==4.38.2...", flush=True)
os.system("pip install -q transformers==4.38.2")

# ============================================================
# 3. BERT PATCH — must be before any groundingdino import
# ============================================================
import transformers
transformers.models.bert.modeling_bert.BertModel.get_head_mask = lambda self, x, y: [None] * y

# ============================================================
# 4. PATCH ms_deform_attn ON DISK — must be before import
#    Fixes '_C is not defined' when CUDA ops fail to compile
# ============================================================

def _patch_ms_deform_attn_file():
    ms_deform_file = "/content/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py"
    if not os.path.exists(ms_deform_file):
        print("ℹ️ ms_deform_attn.py not found yet (will patch after clone)", flush=True)
        return False

    with open(ms_deform_file, "r") as f:
        content = f.read()

    if "# ✅ PATCHED_FALLBACK" in content:
        print("✅ ms_deform_attn already patched", flush=True)
        return True

    # Write a completely replaced forward method
    patched_forward = '''
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        # ✅ PATCHED_FALLBACK: pure PyTorch — no _C CUDA ops needed
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = torch.nn.functional.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :] +
                sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] +
                sampling_offsets / self.n_points *
                reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        # Pure PyTorch attention
        N_, S_, M_, D_ = value.shape
        _, Lq_, M_, L_, P_, _ = sampling_locations.shape
        value_list = value.split([int(H_ * W_) for H_, W_ in input_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for lid_, (H_, W_) in enumerate(input_spatial_shapes):
            H_, W_ = int(H_), int(W_)
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = torch.nn.functional.grid_sample(
                value_l_, sampling_grid_l_, mode="bilinear",
                padding_mode="zeros", align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        attention_weights_r = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
        output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
                  attention_weights_r).sum(-1).view(N_, M_ * D_, Lq_)
        output = output.transpose(1, 2).contiguous()
        output = self.output_proj(output)
        return output
'''

    # Find and replace the existing forward method
    import re
    # Replace from 'def forward' to end of method (next 'def ' at same indent or end of class)
    new_content = re.sub(
        r'    def forward\(self, query.*?(?=\n    def |\nclass |\Z)',
        patched_forward,
        content,
        flags=re.DOTALL
    )

    if new_content != content:
        with open(ms_deform_file, "w") as f:
            f.write(new_content)
        print("✅ ms_deform_attn.py patched on disk!", flush=True)
        return True
    else:
        print("⚠️ Patch regex did not match — trying simple replacement", flush=True)
        # Simpler fallback: just comment out the _C line
        content = content.replace(
            "output = _C.ms_deform_attn_forward(",
            "output = None  # _C.ms_deform_attn_forward( -- DISABLED\n        raise RuntimeError('_C disabled')\n        output = _C.ms_deform_attn_forward("
        )
        with open(ms_deform_file, "w") as f:
            f.write(content)
        return True

_patch_ms_deform_attn_file()

# ============================================================
# 5. INSTALL DEPENDENCIES
# ============================================================

try:
    if os.path.exists("/content/GroundingDINO"):
        sys.path.insert(0, "/content/GroundingDINO")

    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    print("✅ All dependencies already installed", flush=True)

except ImportError:
    print("⚙️ Installing dependencies...", flush=True)

    os.system("apt-get install -qq -y ninja-build > /dev/null 2>&1")
    os.system("pip install -q wheel setuptools ninja")
    os.system("pip install -q ultralytics supervision")
    os.system("pip install -q git+https://github.com/facebookresearch/segment-anything-2.git")
    os.system("pip install -q supabase opencv-python-headless")

    # Fix CUDA symlink for T4
    os.system("ln -sf /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so 2>/dev/null")

    if not os.path.exists("/content/GroundingDINO"):
        print("   -> Cloning GroundingDINO...", flush=True)
        os.system("git clone -q https://github.com/IDEA-Research/GroundingDINO.git /content/GroundingDINO")

    # ✅ Patch the file RIGHT AFTER cloning, before install
    _patch_ms_deform_attn_file()

    print("   -> Installing GroundingDINO...", flush=True)
    ret = os.system(
        "cd /content/GroundingDINO && "
        "CUDA_HOME=/usr/local/cuda "
        "BUILD_WITH_CUDA=1 "
        "TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6' "
        "pip install -q -e . --no-build-isolation"
    )
    if ret != 0:
        print("   -> CUDA build failed, trying CPU-only fallback...", flush=True)
        os.system("cd /content/GroundingDINO && pip install -q -e . --no-build-isolation --no-deps")

    sys.path.insert(0, "/content/GroundingDINO")

    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T

# ============================================================
# 6. CONFIG
# ============================================================

SB_URL      = os.environ.get("SB_URL")
SB_KEY      = os.environ.get("SB_KEY")
JOB_ID      = os.environ.get("JOB_ID")
SESSION_ID  = os.environ.get("SESSION_ID")
TEXT_PROMPT = os.environ.get("TEXT_PROMPT", "").strip()

SAM_CHECKPOINT  = "sam2_hiera_large.pt"
SAM_CONFIG      = "sam2_hiera_l.yaml"
DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
DINO_CONFIG     = "GroundingDINO_SwinT_OGC.py"

# ============================================================
# 7. DOWNLOAD WEIGHTS
# ============================================================

if not os.path.exists(SAM_CHECKPOINT):
    print("⬇️ Downloading SAM2 weights...", flush=True)
    os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

if TEXT_PROMPT and not os.path.exists(DINO_CHECKPOINT):
    print("⬇️ Downloading GroundingDINO weights...", flush=True)
    os.system("wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
    os.system("wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# ============================================================
# 8. INIT SUPABASE
# ============================================================

supabase = create_client(SB_URL, SB_KEY)

# ============================================================
# 9. HELPERS
# ============================================================

def update_status(status, progress=0, logs=""):
    try:
        supabase.table("colab_sessions").update({
            "status": status,
            "progress": progress,
            "logs": logs,
            "last_heartbeat": "now()"
        }).eq("id", SESSION_ID).execute()
    except:
        pass

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 1.0, True)
    return approx.flatten().tolist()

# ============================================================
# 10. LOAD MODELS
# ============================================================

print("📦 Loading SAM2...", flush=True)
sam_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam_model)

print("📦 Loading YOLO fallback...", flush=True)
yolo_model = YOLO("yolov8n.pt").to(device)

dino_model = None
if TEXT_PROMPT:
    print(f"📦 Loading GroundingDINO for: '{TEXT_PROMPT}'", flush=True)
    try:
        dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT)
        dino_model = dino_model.to(device)
        dino_model.eval()
        print("✅ GroundingDINO loaded on GPU!", flush=True)
    except Exception as e:
        print(f"⚠️ DINO load failed, falling back to YOLO: {e}", flush=True)
        dino_model = None

# ============================================================
# 11. MAIN PROCESS
# ============================================================

try:
    update_status("running", 10, "Fetching images...")

    response = supabase.table("images_dev").select("*").eq("job_id", JOB_ID).execute()
    images = response.data

    if not images:
        print("❌ No images found.", flush=True)
        sys.exit(0)

    print(f"📂 Processing {len(images)} images | Mode: {'Prompt Seek → ' + TEXT_PROMPT if TEXT_PROMPT else 'General Scan (YOLO)'}", flush=True)

    for i, img_row in enumerate(images):

        storage_path = img_row["storage_path"]
        print(f"  [{i+1}/{len(images)}] {storage_path}", flush=True)

        # Download image
        img_url = f"{SB_URL}/storage/v1/object/public/datasets/{storage_path}"
        resp = requests.get(img_url, stream=True)
        pil_img = Image.open(resp.raw).convert("RGB")
        image_np = np.array(pil_img)

        sam_predictor.set_image(image_np)
        found_boxes = []

        # ------------------------------------------------
        # A. DETECTION
        # ------------------------------------------------

        if TEXT_PROMPT and dino_model:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            image_transformed, _ = transform(pil_img, None)
            image_transformed = image_transformed.to(device)

            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=dino_model,
                    image=image_transformed,
                    caption=TEXT_PROMPT,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=str(device)  # ✅ "cuda" or "cpu" as string
                )

            h, w = image_np.shape[:2]
            for box, label in zip(boxes, phrases):
                # DINO outputs normalized cx,cy,w,h → convert to pixel xyxy
                cx, cy, bw, bh = box.tolist()
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                found_boxes.append((label, [x1, y1, x2, y2]))

            print(f"     DINO found {len(found_boxes)} objects", flush=True)

        else:
            results = yolo_model(image_np, verbose=False)[0]
            for box in results.boxes:
                label = yolo_model.names[int(box.cls)]
                found_boxes.append((label, box.xyxy[0].tolist()))

            print(f"     YOLO found {len(found_boxes)} objects", flush=True)

        # ------------------------------------------------
        # B. SEGMENTATION
        # ------------------------------------------------

        new_annotations = []

        for j, (label, xyxy) in enumerate(found_boxes):
            try:
                masks, _, _ = sam_predictor.predict(
                    box=np.array(xyxy),
                    multimask_output=False
                )
                poly = mask_to_polygon(masks[0])
                if len(poly) < 6:
                    continue

                new_annotations.append({
                    "id": f"seek_{time.time_ns()}_{i}_{j}_{label}",
                    "label": label,
                    "type": "polygon",
                    "points": poly,
                    "x": xyxy[0],
                    "y": xyxy[1],
                    "width": xyxy[2] - xyxy[0],
                    "height": xyxy[3] - xyxy[1],
                    "user_id": img_row["user_id"],
                    "isNew": True
                })
            except Exception as seg_err:
                print(f"     ⚠️ Segmentation failed for '{label}': {seg_err}", flush=True)
                continue

        print(f"     ✅ {len(new_annotations)} annotations created", flush=True)

        # ------------------------------------------------
        # C. SAVE TO SUPABASE
        # ------------------------------------------------

        supabase.table("annotations_dev").delete().eq("image_id", storage_path).execute()

        supabase.table("annotations_dev").insert({
            "image_id": storage_path,
            "project_id": img_row["project_id"],
            "user_id": img_row["user_id"],
            "annotations": new_annotations
        }).execute()

        pct = int(10 + ((i + 1) / len(images) * 90))
        update_status("running", pct, f"Processed {i+1}/{len(images)} — {len(new_annotations)} found in {storage_path}")

    update_status("completed", 100, "Finished successfully!")
    print("✨ SUCCESS!", flush=True)

except Exception as e:
    import traceback
    err = traceback.format_exc()
    print(f"❌ ERROR:\n{err}", flush=True)
    update_status("failed", 0, str(e))
