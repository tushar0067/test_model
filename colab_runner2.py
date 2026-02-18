import os
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

# --- 1. INSTALLATION & IMPORTS ---
print("🚀 INITIALIZING YOLO + SAM 2 PIPELINE...", flush=True)

try:
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("⚙️ Installing AI Engines (YOLO & SAM 2)...", flush=True)
    os.system('pip install -q ultralytics git+https://github.com/facebookresearch/segment-anything-2.git')
    os.system('pip install -q supabase requests opencv-python-headless')
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 2. CONFIGURATION ---
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
job_id = os.environ.get('JOB_ID')
session_id = os.environ.get('SESSION_ID')

# AI Weights
YOLO_MODEL = 'yolov8n.pt' # Nano version (very fast)
SAM_CHECKPOINT = "sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"

if not os.path.exists(SAM_CHECKPOINT):
    print("⬇️ Downloading AI Weights...", flush=True)
    os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")

# Initialize Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 AI Hardware: {device}", flush=True)

yolo = YOLO(YOLO_MODEL).to(device)
sam2_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)

supabase = create_client(url, key)

def update_status(status, progress=0, logs=""):
    try:
        supabase.table("colab_sessions").update({
            "status": status, "progress": progress, "logs": logs, "last_heartbeat": "now()"
        }).eq("id", session_id).execute()
    except: pass

def mask_to_polygon(mask, tolerance=1.0):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    cnt = max(contours, key=cv2.contourArea)
    epsilon = tolerance * cv2.arcLength(cnt, True) / 1000
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx.flatten().tolist()

# --- 3. PROCESSING LOOP ---
try:
    update_status("running", progress=10, logs="Fetching images...")
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data

    if not images:
        print("❌ No images found for this job.", flush=True)
        exit(0)

    print(f"📂 Found {len(images)} images. Starting detection...", flush=True)

    for i, img_row in enumerate(images):
        # 🔥 VISIBILITY FIX: image_id = storage_path
        target_image_id = img_row['storage_path']
        storage_path = img_row['storage_path']
        
        # Download
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        try:
            pil_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            image_np = np.array(pil_image)
        except:
            print(f"⚠️ Failed download: {storage_path}", flush=True)
            continue

        print(f"🧠 Analyzing [{i+1}/{len(images)}]: {storage_path}", flush=True)
        
        # A. Run YOLO to find boxes and names
        results = yolo(image_np, verbose=False)[0]
        
        # B. Prepare SAM
        predictor.set_image(image_np)
        
        new_annotations = []
        for box in results.boxes:
            label_name = yolo.names[int(box.cls)] # e.g., "car"
            conf = float(box.conf)
            if conf < 0.25: continue # Ignore low confidence
            
            # Get Box Coords
            xyxy = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
            
            # C. Run SAM inside the YOLO box to get perfect polygon
            masks, scores, _ = predictor.predict(
                box=xyxy, 
                multimask_output=False
            )
            
            polygon_points = mask_to_polygon(masks[0])
            if len(polygon_points) < 6: continue
            
            new_annotations.append({
                "id": f"yolo_sam_{int(time.time())}_{label_name}",
                "label": label_name, # 🔥 REAL NAME (Car, Person, etc.)
                "type": "polygon",
                "points": polygon_points,
                "x": float(xyxy[0]), "y": float(xyxy[1]),
                "width": float(xyxy[2] - xyxy[0]), "height": float(xyxy[3] - xyxy[1]),
                "user_id": img_row['user_id'],
                "isNew": True
            })

        # FETCH EXISTING & MERGE
        existing = supabase.table("annotations_dev").select("annotations").eq("image_id", target_image_id).execute()
        existing_data = existing.data[0]['annotations'] if existing.data and existing.data[0]['annotations'] else []
        final_annotations = existing_data + new_annotations

        # UPSERT WITH STORAGE_PATH MATCH
        supabase.table("annotations_dev").delete().eq("image_id", target_image_id).execute()
        supabase.table("annotations_dev").insert({
            "image_id": target_image_id,
            "project_id": img_row['project_id'],
            "user_id": img_row['user_id'],
            "annotations": final_annotations
        }).execute()

        pct = int(10 + ((i + 1) / len(images) * 90))
        update_status("running", progress=pct, logs=f"Detected {len(new_annotations)} objects in {storage_path}")

    update_status("completed", progress=100, logs="YOLO+SAM Detection Finished!")
    print("✨ SUCCESS: Images labeled with real names.", flush=True)

except Exception as e:
    print(f"❌ Error: {str(e)}", flush=True)
    update_status("failed", logs=str(e))
