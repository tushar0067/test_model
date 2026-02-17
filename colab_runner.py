import os
import time
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from supabase import create_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn
import threading

# --- 1. SETUP & INSTALLATION ---
print("🚀 STARTING SAM 2 INTERACTIVE SERVER...", flush=True)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("⚙️ Installing SAM 2 & Server deps...", flush=True)
    os.system('pip install -q git+https://github.com/facebookresearch/segment-anything-2.git')
    os.system('pip install -q supabase requests opencv-python-headless fastapi uvicorn pyngrok python-multipart')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 2. CONFIGURATION ---
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
session_id = os.environ.get('SESSION_ID')

# Use 'large' for best click accuracy, 'tiny' if it crashes
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
CHECKPOINT_PATH = "sam2_hiera_large.pt"
CONFIG_NAME = "sam2_hiera_l.yaml"

if not os.path.exists(CHECKPOINT_PATH):
    print("⬇️ Downloading SAM 2 Weights...", flush=True)
    os.system(f"wget -q {CHECKPOINT_URL}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 AI running on: {device}", flush=True)

# Initialize Model
sam2_model = build_sam2(CONFIG_NAME, CHECKPOINT_PATH, device=device)
predictor = SAM2ImagePredictor(sam2_model)

supabase = create_client(url, key)
app = FastAPI()

# Cache for loaded images to avoid re-downloading/re-embedding
current_image_id = None
image_cache = {}

class SegmentRequest(BaseModel):
    image_path: str
    click_x: float
    click_y: float
    label: int = 1 # 1 = foreground, 0 = background

def mask_to_polygon(mask, tolerance=1.0):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    cnt = max(contours, key=cv2.contourArea)
    epsilon = tolerance * cv2.arcLength(cnt, True) / 1000
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx.flatten().tolist()

@app.post("/segment")
async def segment(req: SegmentRequest):
    global current_image_id
    
    try:
        # 1. Load Image (if changed)
        if current_image_id != req.image_path:
            print(f"📥 Loading new image: {req.image_path}")
            img_url = f"{url}/storage/v1/object/public/datasets/{req.image_path}"
            
            # Download
            resp = requests.get(img_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=404, detail="Image download failed")
            
            # Convert to Numpy
            pil_image = Image.open(BytesIO(resp.content)).convert("RGB")
            image_np = np.array(pil_image)
            
            # Set to SAM 2
            predictor.set_image(image_np)
            current_image_id = req.image_path
            
        # 2. Predict Mask from Click
        input_point = np.array([[req.click_x, req.click_y]])
        input_label = np.array([req.label])
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        # 3. Convert to Polygon
        best_mask = masks[0]
        polygon = mask_to_polygon(best_mask)
        
        return {
            "polygon": polygon,
            "score": float(scores[0])
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. START SERVER & TUNNEL ---
def start_tunnel():
    # Kill existing tunnels
    ngrok.kill()
    
    # Open HTTP tunnel on port 8000
    public_url = ngrok.connect(8000).public_url
    print(f"🔗 Public URL: {public_url}")
    
    # Save URL to Supabase so Frontend can find it
    supabase.table("colab_sessions").update({
        "status": "ready",
        "logs": f"Server running at {public_url}",
        "api_url": public_url,  # <--- New Column needed in DB
        "last_heartbeat": "now()"
    }).eq("id", session_id).execute()

if __name__ == "__main__":
    # Start ngrok in background
    threading.Thread(target=start_tunnel, daemon=True).start()
    
    # Start FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
