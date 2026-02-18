import os
import sys
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

print("🚀 STARTING PROMPT SEEK ENGINE (DINO + SAM2)")

# ============================================================
# 1. DEVICE SETUP (GPU + CPU SAFE)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using Device: {device}")

if device.type == "cuda":
    print("🔥 GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

# ============================================================
# 2. INSTALL DEPENDENCIES (Colab Safe)
# ============================================================

try:
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
except ImportError:
    print("⚙️ Installing dependencies...")
    os.system("pip install -q ultralytics supervision")
    os.system("pip install -q git+https://github.com/facebookresearch/segment-anything-2.git")
    os.system("pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git")
    os.system("pip install -q supabase opencv-python-headless")

    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T

# ============================================================
# 3. CONFIG
# ============================================================

SB_URL = os.environ.get("SB_URL")
SB_KEY = os.environ.get("SB_KEY")
JOB_ID = os.environ.get("JOB_ID")
SESSION_ID = os.environ.get("SESSION_ID")
TEXT_PROMPT = os.environ.get("TEXT_PROMPT", "").strip()

SAM_CHECKPOINT = "sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"
DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
DINO_CONFIG = "GroundingDINO_SwinT_OGC.py"

# ============================================================
# 4. DOWNLOAD WEIGHTS
# ============================================================

if not os.path.exists(SAM_CHECKPOINT):
    os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

if TEXT_PROMPT and not os.path.exists(DINO_CHECKPOINT):
    print("⬇️ Downloading GroundingDINO weights...")
    os.system("wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
    os.system("wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# ============================================================
# 5. INIT SUPABASE
# ============================================================

supabase = create_client(SB_URL, SB_KEY)

# ============================================================
# 6. LOAD MODELS
# ============================================================

print("📦 Loading SAM2...")
sam_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam_model)

print("📦 Loading YOLO fallback...")
yolo_model = YOLO("yolov8n.pt").to(device)

dino_model = None
if TEXT_PROMPT:
    print(f"📦 Loading GroundingDINO for: {TEXT_PROMPT}")
    dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=device)
    dino_model.to(device)
    dino_model.eval()

# ============================================================
# 7. HELPERS
# ============================================================

def update_status(status, progress=0, logs=""):
    try:
        supabase.table("colab_sessions").update({
            "status": status,
            "progress": progress,
            "logs": logs
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
# 8. MAIN PROCESS
# ============================================================

try:
    update_status("running", 10, "Fetching images")

    response = supabase.table("images_dev").select("*").eq("job_id", JOB_ID).execute()
    images = response.data

    if not images:
        print("❌ No images found.")
        sys.exit()

    print(f"📂 Processing {len(images)} images")

    for i, img_row in enumerate(images):

        img_url = f"{SB_URL}/storage/v1/object/public/datasets/{img_row['storage_path']}"
        resp = requests.get(img_url, stream=True)
        pil_img = Image.open(resp.raw).convert("RGB")
        image_np = np.array(pil_img)

        sam_predictor.set_image(image_np)

        found_boxes = []

        # ====================================================
        # A. DETECTION
        # ====================================================

        if TEXT_PROMPT and dino_model:

            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ])

            image_transformed, _ = transform(pil_img, None)
            image_transformed = image_transformed.to(device)

            boxes, logits, phrases = predict(
                model=dino_model,
                image=image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=0.35,
                text_threshold=0.25,
                device=device
            )

            h, w = image_np.shape[:2]

            for box, label in zip(boxes, phrases):
                box = box * torch.tensor([w, h, w, h]).to(device)
                cx, cy, bw, bh = box.tolist()
                found_boxes.append((
                    label,
                    [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]
                ))

        else:
            results = yolo_model(image_np, verbose=False)[0]
            for box in results.boxes:
                label = yolo_model.names[int(box.cls)]
                found_boxes.append((label, box.xyxy[0].tolist()))

        # ====================================================
        # B. SEGMENTATION
        # ====================================================

        new_annotations = []

        for label, xyxy in found_boxes:
            masks, _, _ = sam_predictor.predict(
                box=np.array(xyxy),
                multimask_output=False
            )

            poly = mask_to_polygon(masks[0])
            if len(poly) < 6:
                continue

            new_annotations.append({
                "id": f"seek_{int(time.time())}",
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

        # ====================================================
        # C. SAVE
        # ====================================================

        supabase.table("annotations_dev").delete().eq(
            "image_id", img_row["storage_path"]
        ).execute()

        supabase.table("annotations_dev").insert({
            "image_id": img_row["storage_path"],
            "project_id": img_row["project_id"],
            "user_id": img_row["user_id"],
            "annotations": new_annotations
        }).execute()

        pct = int(10 + ((i + 1) / len(images) * 90))
        update_status("running", pct, f"Processed {i+1}/{len(images)}")

    update_status("completed", 100, "Finished successfully")
    print("✨ SUCCESS")

except Exception as e:
    print("❌ ERROR:", str(e))
    update_status("failed", 0, str(e))
