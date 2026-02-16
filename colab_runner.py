import os
import time
import requests
import sys
from supabase import create_client

# 🔥 IMMEDIATE LOGGING
print("🚀 SCRIPT INITIALIZED: Starting Colab Runner...", flush=True)

# 1. Setup Connections
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
session_id = os.environ.get('SESSION_ID')
job_id = os.environ.get('JOB_ID')

print(f"📡 Connecting to Supabase at: {url}", flush=True)
print(f"🔢 Target Job ID: {job_id}", flush=True)

if not url or url == "undefined" or not key or key == "undefined":
    print("❌ ERROR: Connection credentials are missing or 'undefined'.", flush=True)
    print("   Check your React .env file (VITE_SUPABASE_URL and VITE_SUPABASE_KEY).", flush=True)
    sys.exit(1)

try:
    supabase = create_client(url, key)
    print("✅ Supabase Client Created.", flush=True)
except Exception as e:
    print(f"❌ Failed to create Supabase client: {e}", flush=True)
    sys.exit(1)

def update_status(status, progress=0, logs=""):
    try:
        supabase.table("colab_sessions").update({
            "status": status,
            "progress": progress,
            "logs": logs,
            "last_heartbeat": "now()"
        }).eq("id", session_id).execute()
    except: pass

try:
    print("🔍 Fetching image list from database...", flush=True)
    update_status("running", progress=10, logs="Fetching image list...")
    
    # 2. Get Images for the Job
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data
    
    if not images:
        print(f"⚠️ No images found for Job ID: {job_id}", flush=True)
        update_status("completed", progress=100, logs="No images found.")
        sys.exit(0)

    print(f"📸 Found {len(images)} images. Beginning processing...", flush=True)

    for i, img_row in enumerate(images):
        img_uuid = img_row['id']  # Using 'id' from your CSV
        storage_path = img_row['storage_path']
        
        print(f"🔄 Processing [{i+1}/{len(images)}]: {storage_path}", flush=True)

        # ... (Your AI/SAM Logic Here) ...

        # Update progress in UI
        pct = int(10 + ((i + 1) / len(images) * 90))
        update_status("running", progress=pct, logs=f"Finished {storage_path}")

    update_status("completed", progress=100, logs="Success!")
    print("✨ ALL DONE!", flush=True)

except Exception as e:
    print(f"❌ CRITICAL ERROR: {str(e)}", flush=True)
    update_status("failed", logs=str(e))
