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
# Default to sam3 if not specified by UI
model_choice = os.environ.get('MODEL_VERSION', 'sam3') 

supabase = create_client(url, key)

def update_status(status, progress=0, logs=""):
    supabase.table("colab_sessions").update({
        "status": status,
        "progress": progress,
        "logs": logs,
        "last_heartbeat": "now()"
    }).eq("id", session_id).execute()

def download_image(img_url):
    return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

try:
    update_status("running", progress=5, logs=f"Initializing {model_choice.upper()}...")

    # 2. Model Initialization (SAM 3)
    # Note: In a real Colab, you'd have !pip install sam3-hiera or similar
    # We use a placeholder for the predictor setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    update_status("running", progress=15, logs="Loading model weights to GPU...")
    # predictor = build_sam3_model(checkpoint="sam3_hiera_l.pt").to(device)
    
    # 3. Get Images for the Job
    images = supabase.table("images_dev").select("*").eq("job_id", job_id).execute().data
    update_status("running", progress=25, logs=f"Found {len(images)} images. Starting processing...")

    for i, img_row in enumerate(images):
        img_id = img_row['image_id']
        # Use your Supabase public URL logic here
        img_url = f"{url}/storage/v1/object/public/datasets/{img_row['storage_path']}"
        
        # Load image
        raw_image = download_image(img_url)
        width, height = raw_image.size

        # 4. Run SAM 3 (Concept-based segmentation)
        # For this example, we assume we're looking for 'objects'
        # results = predictor.predict(raw_image, text_prompts=["object"])
        
        # 5. Convert Mask to Polygons (Your UI expects points: [x, y, x, y...])
        # This is a placeholder for actual mask-to-polygon logic
        dummy_points = [100, 100, 200, 100, 200, 200, 100, 200] 
        
        new_annotations = [
            {
                "id": f"sam3_{int(time.time())}_{i}",
                "label": "sam3_detection",
                "type": "polygon",
                "x": 100, "y": 100, "width": 100, "height": 100,
                "points": dummy_points, # 🔥 This is what your CanvasArea.tsx needs
                "user_id": "COLAB_SAM3"
            }
        ]

        # 6. Upsert to annotations_dev
        # We fetch existing first to avoid overwriting manual work
        existing = supabase.table("annotations_dev").select("annotations").eq("image_id", img_id).execute().data
        existing_list = existing[0]['annotations'] if existing else []
        
        supabase.table("annotations_dev").upsert({
            "image_id": img_id,
            "project_id": img_row['project_id'],
            "user_id": "COLAB_WORKER",
            "annotations": existing_list + new_annotations
        }).execute()

        # Update Progress
        pct = int(25 + (i / len(images) * 75))
        update_status("running", progress=pct, logs=f"Finished {img_row['storage_path']}")

    update_status("completed", progress=100, logs="SAM 3 Batch Processing Complete.")

except Exception as e:
    print(f"Error: {e}")
    update_status("failed", logs=str(e))
