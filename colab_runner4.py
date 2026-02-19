import os
import cv2
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from supabase import create_client, Client
from ultralytics import YOLO

# ==============================
# ENV VARIABLES
# ==============================

SB_URL = os.environ.get("SB_URL")
SB_KEY = os.environ.get("SB_KEY")
SESSION_ID = os.environ.get("SESSION_ID")
JOB_ID = os.environ.get("JOB_ID")
TEXT_PROMPT = os.environ.get("TEXT_PROMPT", "").strip()

print("🚀 STARTING PROMPT SEEK ENGINE")
print("📌 Prompt:", TEXT_PROMPT if TEXT_PROMPT else "YOLO Fallback")

# ==============================
# SUPABASE INIT
# ==============================

supabase: Client = create_client(SB_URL, SB_KEY)

def update_session(status=None, logs=None):
    supabase.table("colab_sessions")\
        .update({
            "status": status,
            "logs": logs
        })\
        .eq("id", SESSION_ID)\
        .execute()

# ==============================
# DEVICE
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Device:", DEVICE)

# ==============================
# LOAD MODELS
# ==============================

print("📦 Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")

grounding_model = None

if TEXT_PROMPT:
    print("📦 Loading GroundingDINO...")
    from groundingdino.util.inference import load_model, predict

    grounding_model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/groundingdino_swint_ogc.pth"
    )
    grounding_model = grounding_model.to(DEVICE)
    print("✅ GroundingDINO Loaded")

# ==============================
# GET IMAGES FROM SUPABASE
# ==============================

print("📂 Fetching job images...")

images_res = supabase.table("job_images")\
    .select("*")\
    .eq("job_id", JOB_ID)\
    .execute()

images = images_res.data

if not images:
    update_session(status="failed", logs="No images found")
    raise Exception("No images found for job")

print(f"📦 Processing {len(images)} images")

update_session(status="running", logs=f"Processing {len(images)} images")

# ==============================
# PROCESS IMAGES
# ==============================

for index, img_record in enumerate(images):
    try:
        print(f"[{index+1}/{len(images)}] Processing:", img_record["path"])

        # Download image
        image_url = img_record["public_url"]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_np = np.array(image)

        boxes = []
        labels = []

        # ==============================
        # PROMPT SEEK MODE
        # ==============================
        if TEXT_PROMPT:
            boxes_out, logits, phrases = predict(
                model=grounding_model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=0.35,
                text_threshold=0.25,
                device=DEVICE
            )

            h, w, _ = image_np.shape

            for box, phrase in zip(boxes_out, phrases):
                x1, y1, x2, y2 = box
                boxes.append([int(x1*w), int(y1*h), int(x2*w), int(y2*h)])
                labels.append(phrase)

        # ==============================
        # YOLO FALLBACK
        # ==============================
        else:
            results = yolo_model(image_np)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    labels.append(label)

        # ==============================
        # DRAW BOXES
        # ==============================

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(image_np, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image_np, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0), 2)

        # Save locally
        output_path = f"annotated_{index}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # Upload back to Supabase
        with open(output_path, "rb") as f:
            supabase.storage.from_("annotated")\
                .upload(f"{JOB_ID}/annotated_{index}.jpg", f, {"upsert": True})

        print("✅ Uploaded annotated image")

    except Exception as e:
        print("❌ Error:", str(e))
        update_session(status="failed", logs=str(e))
        raise

# ==============================
# COMPLETE
# ==============================

update_session(status="completed", logs="Processing complete")
print("🎉 JOB COMPLETED")
