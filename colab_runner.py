import os
import time
from supabase import create_client

# 1. Setup Connections
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
session_id = os.environ.get('SESSION_ID')
job_id = os.environ.get('JOB_ID')

supabase = create_client(url, key)

def update_status(status, progress=0, logs=""):
    supabase.table("colab_sessions").update({
        "status": status,
        "progress": progress,
        "logs": logs,
        "last_heartbeat": "now()"
    }).eq("id", session_id).execute()

try:
    update_status("running", progress=10, logs="Downloading image list...")
    
    # 2. Get Images for the Job
    images = supabase.table("images_dev").select("*").eq("job_id", job_id).execute().data
    
    update_status("running", progress=30, logs=f"Found {len(images)} images. Starting AI Inference...")

    for i, img in enumerate(images):
        # 3. Simulate/Run Heavy AI Task (e.g., YOLOv10)
        # result = model.predict(img['storage_path'])
        
        # 4. Construct Annotations (Following your JSONB schema)
        new_annotations = [
            {
                "id": f"colab_{int(time.time())}",
                "label": "detected_object",
                "type": "rectangle",
                "x": 100, "y": 100, "width": 50, "height": 50, # Example coords
                "points": []
            }
        ]

        # 5. Upsert back to Supabase
        supabase.table("annotations_dev").upsert({
            "image_id": img['storage_path'],
            "job_id": job_id,
            "annotations": new_annotations,
            "user_id": "COLAB_WORKER"
        }).execute()

        # Update progress in UI
        pct = int(30 + (i / len(images) * 70))
        update_status("running", progress=pct, logs=f"Processed {img['storage_path']}")

    update_status("completed", progress=100, logs="All images processed successfully.")

except Exception as e:
    update_status("failed", logs=str(e))
