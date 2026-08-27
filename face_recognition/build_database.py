import os
import pickle

import cv2
import numpy as np
import onnxruntime as ort

from ultralytics import YOLO
from insightface.model_zoo import get_model
from insightface.utils.face_align import norm_crop


# =========================================================
# SETTINGS
# =========================================================

FACES_FOLDER = "faces"
DATABASE_FILE = "face_database.pkl"

YOLO_MODEL = "models/yolov11n-face.pt"

# Buffalo model location
ARCFACE_MODEL = os.path.expanduser(
    "~/.insightface/models/buffalo_l/w600k_r50.onnx"
)

YOLO_CONFIDENCE = 0.50


# =========================================================
# GPU CHECK
# =========================================================

print("=" * 50)
print("Checking ONNX Runtime")
print("=" * 50)

providers = ort.get_available_providers()

print("Available providers:")
for p in providers:
    print(" -", p)

if "CUDAExecutionProvider" not in providers:
    raise RuntimeError(
        "CUDAExecutionProvider not found!"
    )

print("\nCUDA is available for ONNX Runtime.")


# =========================================================
# LOAD YOLO
# =========================================================

print("\n" + "=" * 50)
print("Loading YOLO")
print("=" * 50)

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# =========================================================
# CHECK ARCFACE MODEL
# =========================================================

print("\n" + "=" * 50)
print("Checking ArcFace model")
print("=" * 50)

if not os.path.exists(ARCFACE_MODEL):

    raise FileNotFoundError(
        f"\nArcFace model not found:\n{ARCFACE_MODEL}\n\n"
        "Make sure buffalo_l has been downloaded."
    )

print("ArcFace model:")
print(ARCFACE_MODEL)


# =========================================================
# LOAD ARCFACE
# =========================================================

print("\nLoading ArcFace...")

recognizer = get_model(
    ARCFACE_MODEL,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

recognizer.prepare(ctx_id=0)

print("ArcFace loaded on CUDA.")


# =========================================================
# FUNCTION:
# ESTIMATE 5 LANDMARKS FROM BOUNDING BOX
# =========================================================

def estimate_landmarks(box):
    """
    Estimate 5 facial landmarks from a face bounding box.

    Order:
        1. left eye
        2. right eye
        3. nose
        4. left mouth
        5. right mouth

    NOTE:
    These are estimated positions, not real landmarks.
    """

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    landmarks = np.array([
        [x1 + 0.30 * w, y1 + 0.38 * h],  # left eye
        [x1 + 0.70 * w, y1 + 0.38 * h],  # right eye
        [x1 + 0.50 * w, y1 + 0.55 * h],  # nose
        [x1 + 0.35 * w, y1 + 0.75 * h],  # left mouth
        [x1 + 0.65 * w, y1 + 0.75 * h],  # right mouth
    ], dtype=np.float32)

    return landmarks


# =========================================================
# FUNCTION:
# CREATE EMBEDDING
# =========================================================

def create_embedding(image, box):

    # Convert box to integers
    x1, y1, x2, y2 = map(int, box)

    # Make sure coordinates are inside image
    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    # -----------------------------------------------------
    # Estimate 5 landmarks
    # -----------------------------------------------------

    landmarks = estimate_landmarks(
        [x1, y1, x2, y2]
    )

    # -----------------------------------------------------
    # Align face for ArcFace
    # -----------------------------------------------------

    aligned_face = norm_crop(
        image,
        landmarks,
        image_size=112
    )

    # -----------------------------------------------------
    # ArcFace embedding
    # -----------------------------------------------------

    embedding = recognizer.get_feat(
        aligned_face
    )

    embedding = embedding.flatten()

    # -----------------------------------------------------
    # Normalize vector
    # -----------------------------------------------------

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding.astype(np.float32)


# =========================================================
# DATABASE
# =========================================================

known_embeddings = []
known_names = []


# =========================================================
# READ FACES FOLDER
# =========================================================

if not os.path.exists(FACES_FOLDER):

    raise FileNotFoundError(
        f"Folder not found: {FACES_FOLDER}"
    )


files = os.listdir(FACES_FOLDER)

image_files = [
    f for f in files
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]


if len(image_files) == 0:

    raise RuntimeError(
        "No images found inside faces/"
    )


print("\n" + "=" * 50)
print("Building database")
print("=" * 50)


# =========================================================
# PROCESS EVERY IMAGE
# =========================================================

for filename in image_files:

    image_path = os.path.join(
        FACES_FOLDER,
        filename
    )

    print(f"\nProcessing: {filename}")

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:

        print("ERROR: Could not read image.")
        continue


    # -----------------------------------------------------
    # YOLO FACE DETECTION
    # -----------------------------------------------------

    results = yolo.predict(
        source=image,
        device="cuda:0",
        conf=YOLO_CONFIDENCE,
        verbose=False
    )


    boxes = results[0].boxes


    # -----------------------------------------------------
    # Check face
    # -----------------------------------------------------

    if boxes is None or len(boxes) == 0:

        print("WARNING: No face detected.")
        continue


    # -----------------------------------------------------
    # Choose largest face
    # -----------------------------------------------------

    best_box = None
    best_area = 0

    for box in boxes.xyxy.cpu().numpy():

        x1, y1, x2, y2 = box

        area = (x2 - x1) * (y2 - y1)

        if area > best_area:

            best_area = area
            best_box = box


    if best_box is None:

        print("WARNING: Could not select face.")
        continue


    # -----------------------------------------------------
    # Create embedding
    # -----------------------------------------------------

    embedding = create_embedding(
        image,
        best_box
    )


    if embedding is None:

        print("WARNING: Could not create embedding.")
        continue


    # -----------------------------------------------------
    # Name = filename without extension
    # -----------------------------------------------------

    name = os.path.splitext(filename)[0]


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    known_embeddings.append(embedding)
    known_names.append(name)


    print(
        f"Added: {name}"
    )

    print(
        f"Vector shape: {embedding.shape}"
    )


# =========================================================
# SAVE DATABASE
# =========================================================

database = {
    "embeddings": known_embeddings,
    "names": known_names
}


with open(
    DATABASE_FILE,
    "wb"
) as file:

    pickle.dump(
        database,
        file
    )


# =========================================================
# RESULT
# =========================================================

print("\n" + "=" * 50)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 50)

print(
    f"People: {len(known_names)}"
)

print(
    f"Database: {DATABASE_FILE}"
)

if len(known_embeddings) > 0:

    print(
        f"Vector size: {known_embeddings[0].shape}"
    )

print("=" * 50)
