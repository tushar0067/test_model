import os
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

# 1. Setup Connections
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
session_id = os.environ.get('SESSION_ID')
job_id = os.environ.get('JOB_ID')
model_choice = os.environ.get('MODEL_VERSION', 'sam3')

print(f"🚀 Initializing {model_choice.upper()} Runner...", flush=True)

if not url or url == "undefined" or not key or key == "undefined":
    print("❌ ERROR: Supabase URL or Key is undefined. Check your .env/Vite variables.", flush=True)
    exit(1)

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
    """Converts binary mask to [x, y, x, y...] format for Konva/React"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
    # Take the largest detected object
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify the polygon to avoid UI lag in the browser
    epsilon = tolerance * cv2.arcLength(cnt, True) / 1000
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    return approx.flatten().tolist()

try:
    # 2. Model Loading (Logic for SAM 3 / 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Running on {device}", flush=True)
    update_status("running", progress=10, logs="Loading model weights...")
    
    # predictor = build_sam3_predictor(device=device) # Placeholder for actual import

    # 3. Fetch Job Data
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data
    
    if not images:
        update_status("completed", progress=100, logs="Job empty. No images found.")
        exit(0)

    print(f"📂 Found {len(images)} images to process.", flush=True)

    for i, img_row in enumerate(images):
        img_uuid = img_row['id'] # Matches your CSV column 'id'
        storage_path = img_row['storage_path']
        
        print(f"📸 Processing [{i+1}/{len(images)}]: {storage_path}", flush=True)
        update_status("running", progress=int(20 + (i/len(images)*80)), logs=f"Scanning: {storage_path}")

        # Download image
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        image_np = np.array(raw_image)

        # 4. RUN AI INFERENCE
        # binary_mask = predictor.predict(image_np) # Placeholder
        
        # Simulated Mask for logic verification:
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), 1, -1)

        # Convert Mask to coordinate points
        polygon_points = mask_to_polygon(mask)

        if not polygon_points: continue

        # 5. Build Annotation following your schema
        new_annotations = [
            {
                "id": f"colab_{int(time.time())}_{i}",
                "label": "auto_labeled",
                "type": "polygon",
                "points": polygon_points,
                "x": min(polygon_points[0::2]),
                "y": min(polygon_points[1::2]),
                "width": max(polygon_points[0::2]) - min(polygon_points[0::2]),
                "height": max(polygon_points[1::2]) - min(polygon_points[1::2]),
                "user_id": "COLAB_WORKER"
            }
        ]

        # 6. Upsert results
        supabase.table("annotations_dev").upsert({
            "image_id": img_uuid, 
            "project_id": img_row['project_id'],
            "user_id": "COLAB_WORKER",
            "annotations": new_annotations
        }).execute()

    update_status("completed", progress=100, logs="Success! All images annotated.")
    print("✨ Finished!", flush=True)

except Exception as e:
    print(f"❌ Error: {str(e)}", flush=True)
    update_status("failed", logs=str(e))
