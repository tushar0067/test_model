import os
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

# --- 1. SETUP & INSTALLATION CHECK ---
print("🚀 STARTING REAL SAM 2 PIPELINE...", flush=True)

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("⚙️ Installing SAM 2 & Dependencies (this takes 1-2 mins)...", flush=True)
    os.system('pip install -q git+https://github.com/facebookresearch/segment-anything-2.git')
    os.system('pip install -q supabase requests opencv-python-headless')
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- 2. CONFIGURATION ---
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
job_id = os.environ.get('JOB_ID')
session_id = os.environ.get('SESSION_ID')

# Download SAM 2 Weights (Tiny version for speed in Colab Free Tier)
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
CHECKPOINT_PATH = "sam2_hiera_tiny.pt"
CONFIG_NAME = "sam2_hiera_t.yaml"

if not os.path.exists(CHECKPOINT_PATH):
    print("⬇️ Downloading SAM 2 Weights...", flush=True)
    os.system(f"wget -q {CHECKPOINT_URL}")

# --- 3. INITIALIZE AI MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Loading Model on {device}...", flush=True)

# Build the model
sam2_model = build_sam2(CONFIG_NAME, CHECKPOINT_PATH, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100.0
)

supabase = create_client(url, key)

def update_status(status, progress=0, logs=""):
    try:
        supabase.table("colab_sessions").update({
            "status": status,
            "progress": progress,
            "logs": logs,
            "last_heartbeat": "now()"
        }).eq("id", session_id).execute()
    except: pass

def mask_to_polygon(mask, tolerance=1.0):
    """Converts binary mask to [x, y, x, y...]"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    cnt = max(contours, key=cv2.contourArea)
    epsilon = tolerance * cv2.arcLength(cnt, True) / 1000
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx.flatten().tolist()

# --- 4. PROCESSING LOOP ---
try:
    update_status("running", progress=10, logs="Fetching images...")
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data

    if not images:
        print("❌ No images found.", flush=True)
        exit(0)

    print(f"📂 Processing {len(images)} images with SAM 2...", flush=True)

    for i, img_row in enumerate(images):
        target_image_id = img_row['storage_path'] # Matches React ID
        storage_path = img_row['storage_path']
        
        # Download Image
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        try:
            pil_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            image_np = np.array(pil_image)
        except Exception as e:
            print(f"⚠️ Failed to download {storage_path}: {e}", flush=True)
            continue

        # 🔥 REAL AI INFERENCE
        print(f"🧠 Segmenting: {storage_path}...", flush=True)
        masks = mask_generator.generate(image_np)
        
        # Convert SAM output to your DB Schema
        new_annotations = []
        for idx, ann in enumerate(masks):
            # Only keep confident masks
            if ann['stability_score'] < 0.85: continue
            
            # Convert binary mask to polygon
            polygon_points = mask_to_polygon(ann['segmentation'])
            
            if len(polygon_points) < 6: continue # Skip noise
            
            # Calculate Bounding Box
            xs = polygon_points[0::2]
            ys = polygon_points[1::2]
            
            new_annotations.append({
                "id": f"sam_{int(time.time())}_{idx}",
                "label": "object", # Or use a classifier here if you want
                "type": "polygon",
                "points": polygon_points,
                "x": min(xs),
                "y": min(ys),
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
                "user_id": img_row['user_id'],
                "isNew": True,
                "confidence": float(ann['stability_score'])
            })

        # FETCH EXISTING (to preserve manual work)
        existing = supabase.table("annotations_dev").select("annotations").eq("image_id", target_image_id).execute()
        existing_data = existing.data[0]['annotations'] if existing.data and existing.data[0]['annotations'] else []
        
        final_annotations = existing_data + new_annotations

        # DELETE OLD & INSERT NEW
        supabase.table("annotations_dev").delete().eq("image_id", target_image_id).execute()
        supabase.table("annotations_dev").insert({
            "image_id": target_image_id,
            "project_id": img_row['project_id'],
            "user_id": img_row['user_id'],
            "annotations": final_annotations
        }).execute()

        pct = int(10 + ((i + 1) / len(images) * 90))
        update_status("running", progress=pct, logs=f"Segmented {len(new_annotations)} objects in {storage_path}")

    update_status("completed", progress=100, logs="SAM 2 Processing Complete!")
    print("✨ SUCCESS: Real segmentation finished.", flush=True)

except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}", flush=True)
    update_status("failed", logs=str(e))
