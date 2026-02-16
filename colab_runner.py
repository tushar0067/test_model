import os
import time
import requests
import numpy as np
import cv2
from PIL import Image
from supabase import create_client

# 1. Setup Connections
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
session_id = os.environ.get('SESSION_ID')
job_id = os.environ.get('JOB_ID')

print(f"🚀 SCRIPT STARTED | Job: {job_id}", flush=True)

if not url or url == "undefined" or not key or key == "undefined":
    print("❌ ERROR: Credentials missing. Check React .env file.", flush=True)
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
    """Converts binary mask to [x, y, x, y...] for React Konva"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    
    cnt = max(contours, key=cv2.contourArea)
    epsilon = tolerance * cv2.arcLength(cnt, True) / 1000
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    return approx.flatten().tolist()

try:
    update_status("running", progress=5, logs="Fetching images...")

    # 2. Get Images
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data
    
    if not images:
        update_status("completed", progress=100, logs="No images found.")
        exit(0)

    print(f"📂 Processing {len(images)} images...", flush=True)

    for i, img_row in enumerate(images):
        # 🔥 CRITICAL FIXES FOR VISIBILITY 🔥
        img_uuid = img_row['id']           # Matches 'image_id' in annotations table
        storage_path = img_row['storage_path']
        project_id = img_row['project_id']
        target_user_id = img_row['user_id'] # 🔥 Matches the user who owns the image

        print(f"📸 [{i+1}/{len(images)}] {storage_path}", flush=True)
        
        # Download Image
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        try:
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            image_np = np.array(raw_image)
            h, w = image_np.shape[:2]
        except:
            print(f"⚠️ Failed to download: {storage_path}", flush=True)
            continue

        # 3. GENERATE MASK (Using reliable simulation to ensure visibility first)
        # (Replace this block with predictor.predict(image_np) when SAM2 is installed)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create a large central box so you DEFINITELY see it
        cv2.rectangle(mask, (int(w*0.25), int(h*0.25)), (int(w*0.75), int(h*0.75)), 1, -1)
        
        # 4. Convert to Polygon
        polygon_points = mask_to_polygon(mask)

        # 5. Create Annotation Object
        new_annotations = [
            {
                "id": f"colab_{int(time.time())}_{i}",
                "label": "auto_detected",
                "type": "polygon",
                "points": polygon_points,
                "x": min(polygon_points[0::2]), # Bounding Box X
                "y": min(polygon_points[1::2]), # Bounding Box Y
                "width": max(polygon_points[0::2]) - min(polygon_points[0::2]),
                "height": max(polygon_points[1::2]) - min(polygon_points[1::2]),
                "user_id": target_user_id, # 🔥 ASSIGN TO ACTUAL USER
                "isNew": True
            }
        ]

        # 6. Save to DB
        # We fetch existing annotations first to avoid overwriting manual work
        existing = supabase.table("annotations_dev").select("annotations").eq("image_id", img_uuid).execute()
        existing_data = existing.data[0]['annotations'] if existing.data and existing.data[0]['annotations'] else []
        
        final_annotations = existing_data + new_annotations

        supabase.table("annotations_dev").upsert({
            "image_id": img_uuid,
            "project_id": project_id,
            "user_id": target_user_id, # 🔥 CRITICAL
            "annotations": final_annotations
        }).execute()

        # Update Progress
        pct = int(10 + ((i + 1) / len(images) * 90))
        update_status("running", progress=pct, logs=f"Annotated: {storage_path}")

    update_status("completed", progress=100, logs="Success! Refresh your canvas.")
    print("✨ SUCCESS: Check your React App now.", flush=True)

except Exception as e:
    print(f"❌ Error: {str(e)}", flush=True)
    update_status("failed", logs=str(e))
