import os
import time
import requests
import torch
import numpy as np
from PIL import Image
from supabase import create_client

# 1. Setup Connections
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
session_id = os.environ.get('SESSION_ID')
job_id = os.environ.get('JOB_ID')
model_choice = os.environ.get('MODEL_VERSION', 'sam3') 

# Safety check for credentials
if not url or url == "undefined" or not key or key == "undefined":
    print("❌ ERROR: Supabase URL or Key is undefined.")
    exit(1)

supabase = create_client(url, key)

def update_status(status, progress=0, logs=""):
    supabase.table("colab_sessions").update({
        "status": status,
        "progress": progress,
        "logs": logs,
        "last_heartbeat": "now()"
    }).eq("id", session_id).execute()

def download_image(img_url):
    response = requests.get(img_url, stream=True)
    return Image.open(response.raw).convert("RGB")

try:
    update_status("running", progress=5, logs=f"Initializing {model_choice.upper()}...")
    
    # 2. Get Images for the Job
    # Based on your CSV: columns are status, project_id, storage_path, id, job_id, etc.
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data
    
    if not images:
        update_status("completed", progress=100, logs="No images found for this job.")
        exit(0)

    update_status("running", progress=20, logs=f"Found {len(images)} images. Starting processing...")

    for i, img_row in enumerate(images):
        # 🔥 FIX: Use 'id' from the images_dev table
        img_uuid = img_row['id'] 
        storage_path = img_row['storage_path']
        project_id = img_row['project_id']
        
        # Construct public URL (Assumes bucket is 'datasets')
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        
        update_status("running", progress=int(20 + (i/len(images)*75)), logs=f"Processing: {storage_path}")

        # 3. Running SAM 3 / Logic Placeholder
        # (This is where you'd run your model inference)
        # For now, we create a sample polygon
        dummy_points = [50, 50, 150, 50, 150, 150, 50, 150] 
        
        new_annotations = [
            {
                "id": f"colab_{int(time.time())}_{i}",
                "label": "sam3_auto",
                "type": "polygon",
                "points": dummy_points,
                "x": 50, "y": 50, "width": 100, "height": 100,
                "user_id": "COLAB_WORKER"
            }
        ]

        # 4. Upsert to annotations_dev
        # Note: 'image_id' in annotations_dev references 'id' in images_dev
        supabase.table("annotations_dev").upsert({
            "image_id": img_uuid, 
            "project_id": project_id,
            "user_id": "COLAB_WORKER",
            "annotations": new_annotations # Or merge with existing if needed
        }).execute()

    update_status("completed", progress=100, logs="All images processed successfully.")

except Exception as e:
    print(f"Error occurred: {e}")
    update_status("failed", logs=f"Python Error: {str(e)}")
