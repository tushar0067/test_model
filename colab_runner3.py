import os
import time
import requests
import torch
import numpy as np
import cv2
import sys
from PIL import Image
from supabase import create_client

# --- 1. INSTALLATION ---
print("🚀 INITIALIZING PROMPT SEEK (DINO + SAM 2) - STABLE VERSION...", flush=True)

try:
    import transformers
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, load_image, predict
    import groundingdino.datasets.transforms as T
except ImportError:
    print("⚙️ Installing Dependencies...", flush=True)
    os.system('pip install -q transformers==4.38.2')
    os.system('pip install -q ultralytics git+https://github.com/facebookresearch/segment-anything-2.git')
    os.system('pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git')
    os.system('pip install -q supabase requests opencv-python-headless supervision')
    
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from groundingdino.util.inference import load_model, load_image, predict
    import groundingdino.datasets.transforms as T

# --- 2. CONFIGURATION ---
url = os.environ.get('SB_URL')
key = os.environ.get('SB_KEY')
job_id = os.environ.get('JOB_ID')
session_id = os.environ.get('SESSION_ID')
text_prompt = os.environ.get('TEXT_PROMPT', '').strip()

SAM_CHECKPOINT = "sam2_hiera_large.pt"
SAM_CONFIG = "sam2_hiera_l.yaml"
DINO_CONFIG = "GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"

if not os.path.exists(SAM_CHECKPOINT):
    os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
if text_prompt and not os.path.exists(DINO_CHECKPOINT):
    os.system("wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
    os.system("wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supabase = create_client(url, key)

# --- 3. THE "get_head_mask" & ".to()" PATCHES ---
# Fix 1: BertModel Attribute Patch
import transformers
transformers.models.bert.modeling_bert.BertModel.get_head_mask = lambda self, x, y: None

# Initialize Models
sam2_predictor = SAM2ImagePredictor(build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device=device))
yolo_model = YOLO('yolov8n.pt').to(device)

dino_model = None
if text_prompt:
    print(f"📥 Loading Grounding DINO for: {text_prompt}", flush=True)
    dino_model = load_model(DINO_CONFIG, DINO_CHECKPOINT, device=device)

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

# --- 4. PROCESSING LOOP ---
try:
    update_status("running", progress=10, logs="Fetching images...")
    response = supabase.table("images_dev").select("*").eq("job_id", job_id).execute()
    images = response.data

    if not images:
        sys.exit(0)

    for i, img_row in enumerate(images):
        target_image_id = img_row['storage_path']
        storage_path = img_row['storage_path']
        
        img_url = f"{url}/storage/v1/object/public/datasets/{storage_path}"
        resp = requests.get(img_url, stream=True)
        pil_img = Image.open(resp.raw).convert("RGB")
        image_np = np.array(pil_img)
        sam2_predictor.set_image(image_np)
        
        found_boxes = []

        if text_prompt and dino_model:
            # 🔥 FIX: Safe Tensor Transformation for DINO
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(pil_img, None)
            
            # 🔥 FIX: Manually move to device to avoid the .to() argument error
            image_input = image_transformed.to(device=device)

            boxes, logits, phrases = predict(
                model=dino_model, 
                image=image_input, # Use the manually moved tensor
                caption=text_prompt, 
                box_threshold=0.35, 
                text_threshold=0.25,
                device=str(device) # Pass device as string to satisfy DINO internal checks
            )
            
            h, w = image_np.shape[:2]
            for box, label in zip(boxes, phrases):
                box = box * torch.Tensor([w, h, w, h])
                cx, cy, bw, bh = box.tolist()
                found_boxes.append((label, [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]))
        else:
            results = yolo_model(image_np, verbose=False)[0]
            for box in results.boxes:
                found_boxes.append((yolo_model.names[int(box.cls)], box.xyxy[0].tolist()))

        # SEGMENTATION & SAVE
        new_annotations = []
        for label, xyxy in found_boxes:
            masks, _, _ = sam2_predictor.predict(box=np.array(xyxy), multimask_output=False)
            poly = mask_to_polygon(masks[0])
            if len(poly) < 6: continue
            
            new_annotations.append({
                "id": f"seek_{int(time.time())}_{label}",
                "label": label, "type": "polygon", "points": poly,
                "x": xyxy[0], "y": xyxy[1], "width": xyxy[2]-xyxy[0], "height": xyxy[3]-xyxy[1],
                "user_id": img_row['user_id'], "isNew": True
            })

        supabase.table("annotations_dev").delete().eq("image_id", target_image_id).execute()
        supabase.table("annotations_dev").insert({
            "image_id": target_image_id, "project_id": img_row['project_id'],
            "user_id": img_row['user_id'], "annotations": new_annotations
        }).execute()

        update_status("running", progress=int(10+(i+1)/len(images)*90), logs=f"Processed {storage_path}")

    update_status("completed", progress=100, logs="Done!")
    print("✨ SUCCESS!", flush=True)

except Exception as e:
    print(f"❌ Error: {str(e)}", flush=True)
    update_status("failed", logs=str(e))
