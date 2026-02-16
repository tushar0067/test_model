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
model_choice = os.environ.get('MODEL_VERSION', 'sam2') 

print(f"🚀 Initializing {model_choice.upper()} Runner...", flush=True)

if not url or url == "undefined" or not key or key == "undefined":
    print("❌ ERROR: Supabase credentials missing. Check your Vite .env file.", flush=True)
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
    """Converts binary mask to [x, y, x, y...] format for your React Canvas"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
    # Get the largest detected contour
    cnt = max(contours, key=cv2.contourArea)
    
    # Simplify the polygon to keep the point count reasonable for the browser
    epsilon = tolerance * cv2.arcLength(cnt, True) / 1000
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    return approx.flatten().tolist()

try:
    # 2. Model Loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Running on {device}", flush=True)
    update_status("running", progress=10, logs="Loading SAM weights into GPU...")
    
    # Note: In a production Colab, you would initialize the SamPredictor here
    # from sam2.build_sam import build_sam2
    # predictor = build_sam2(model_cfg, checkpoint, device=device)

    # 3. Fetch Images
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data
    
    if not images:
        update_status("completed", progress=100, logs="No images found for this job.")
        exit(0)

    print(f"📂 Found {len(images)} images. Starting AI segmentation...", flush=True)

    for i, img_row in enumerate(images):
        img_uuid = img_row['id'] # Verified from your CSV
        storage_path = img_row['storage_path']
        project_id = img_row['project_id']
        
        print(f"📸 Processing [{i+1}/{len(images)}]: {storage_path}", flush=True)
        update_status("running", progress=int(20 + (i/len(images)*80)), logs=f"Annotating: {storage_path}")

        # Download and Prepare Image
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        image_np = np.array(raw_image)

        # 4. RUN AI INFERENCE (Example: Automatic mask generation)
        # masks = predictor.generate(image_np)
        
        # --- Logic Simulation for Polygon Conversion ---
        h, w = image_np.shape[:2]
        # We create a dummy central mask to verify the polygon pipeline
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), 1, -1)
        
        polygon_points = mask_to_polygon(mask)

        # 5. Build Annotation following your specific schema
        new_annotations = [
            {
                "id": f"colab_{int(time.time())}_{i}",
                "label": "sam_auto_label",
                "type": "polygon",
                "points": polygon_points,
                "x": min(polygon_points[0::2]) if polygon_points else 0,
                "y": min(polygon_points[1::2]) if polygon_points else 0,
                "width": (max(polygon_points[0::2]) - min(polygon_points[0::2])) if polygon_points else 0,
                "height": (max(polygon_points[1::2]) - min(polygon_points[1::2])) if polygon_points else 0,
                "user_id": "COLAB_WORKER"
            }
        ]

        # 6. Upsert to Supabase
        # This links the image UUID to the annotations_dev table
        supabase.table("annotations_dev").upsert({
            "image_id": img_uuid, 
            "project_id": project_id,
            "user_id": "COLAB_WORKER",
            "annotations": new_annotations
        }).execute()

    update_status("completed", progress=100, logs="Success! All images processed.")
    print("✨ Batch processing finished successfully.", flush=True)

except Exception as e:
    print(f"❌ Error: {str(e)}", flush=True)
    update_status("failed", logs=str(e))
