import os
import sys
import cv2
import pickle
import time
import numpy as np
import torch

# =========================================================
# CUDA 12 DLL PATHS FOR ONNX RUNTIME
# =========================================================

ENV_SITE_PACKAGES = os.path.join(
    sys.prefix,
    "Lib",
    "site-packages"
)

CUDA12_PATHS = [

    os.path.join(
        ENV_SITE_PACKAGES,
        "nvidia",
        "cublas",
        "bin"
    ),

    os.path.join(
        ENV_SITE_PACKAGES,
        "nvidia",
        "cudnn",
        "bin"
    ),

    os.path.join(
        ENV_SITE_PACKAGES,
        "nvidia",
        "cuda_runtime",
        "bin"
    ),

    os.path.join(
        ENV_SITE_PACKAGES,
        "nvidia",
        "cuda_nvrtc",
        "bin"
    )

]


# Add CUDA 12 DLL directories
# only for this Python process

for dll_path in CUDA12_PATHS:

    if os.path.isdir(dll_path):

        try:

            os.add_dll_directory(
                dll_path
            )

        except Exception:

            pass


# =========================================================
# IMPORT ONNX RUNTIME AFTER DLL PATH SETUP
# =========================================================

import onnxruntime as ort


# =========================================================
# IMPORT YOLO / INSIGHTFACE
# =========================================================

from ultralytics import YOLO

from insightface.model_zoo import get_model

from insightface.utils.face_align import norm_crop





# =========================================================
# SETTINGS
# =========================================================

# Load CUDA / cuDNN DLLs from Python packages
# ort.preload_dlls(directory="")

YOLO_MODEL = "models/yolov11n-face.pt"

DATABASE_FILE = "face_database.pkl"

ARCFACE_MODEL = r"C:\Users\Reza\.insightface\models\buffalo_l\w600k_r50.onnx"

CONFIDENCE = 0.50

# Recognition threshold
# Higher = stricter
RECOGNITION_THRESHOLD = 0.45


# =========================================================
# CHECK ONNX RUNTIME
# =========================================================

print("=" * 60)
print("ONNX Runtime")
print("=" * 60)

print("Version:", ort.__version__)

providers = ort.get_available_providers()

print("Available providers:")

for provider in providers:
    print(" -", provider)


# =========================================================
# CHECK CUDA PROVIDER
# =========================================================

if "CUDAExecutionProvider" in providers:

    print("\nCUDAExecutionProvider: AVAILABLE")

else:

    print("\nWARNING:")
    print("CUDAExecutionProvider is NOT available.")
    print("ArcFace will NOT be able to use CUDA.")


# =========================================================
# LOAD DATABASE
# =========================================================

print("\n" + "=" * 60)
print("Loading face database")
print("=" * 60)

with open(DATABASE_FILE, "rb") as file:

    database = pickle.load(file)


known_embeddings = database["embeddings"]
known_names = database["names"]


# Convert embeddings to numpy array

known_embeddings = np.asarray(
    known_embeddings,
    dtype=np.float32
)


print("People:", len(known_names))

print("Names:")

for name in known_names:
    print(" -", name)


# =========================================================
# NORMALIZE DATABASE EMBEDDINGS
# =========================================================

embedding_norms = np.linalg.norm(
    known_embeddings,
    axis=1,
    keepdims=True
)

embedding_norms[embedding_norms == 0] = 1.0

known_embeddings = (
    known_embeddings /
    embedding_norms
)

known_embeddings = known_embeddings.astype(
    np.float32
)


# =========================================================
# LOAD YOLO
# =========================================================

print("\n" + "=" * 60)
print("Loading YOLO")
print("=" * 60)

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")

# Force YOLO to GPU

YOLO_DEVICE = "cuda:0"

print("YOLO device:", YOLO_DEVICE)


# =========================================================
# LOAD ARCFACE
# =========================================================

print("\n" + "=" * 60)
print("Loading ArcFace")
print("=" * 60)

recognizer = get_model(

    ARCFACE_MODEL,

    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

)

# Prepare ArcFace

recognizer.prepare(ctx_id=0)


# =========================================================
# SHOW ACTUAL ARCFACE PROVIDER
# =========================================================

print("\n" + "=" * 60)
print("ArcFace Session")
print("=" * 60)

try:

    arcface_providers = (
        recognizer.session.get_providers()
    )

    print(
        "Session providers:",
        arcface_providers
    )

except Exception as e:

    arcface_providers = []

    print(
        "Could not determine provider."
    )

    print("Error:", e)


# Determine actual ArcFace provider

if (
    len(arcface_providers) > 0 and
    arcface_providers[0] == "CUDAExecutionProvider"
):

    ARCFACE_DEVICE = "CUDA"

else:

    ARCFACE_DEVICE = "CPU"


print(
    "ArcFace device:",
    ARCFACE_DEVICE
)


# =========================================================
# CUDA STATUS
# =========================================================

YOLO_STATUS = "CUDA"

if ARCFACE_DEVICE == "CUDA":

    ARCFACE_STATUS = "CUDA"

else:

    ARCFACE_STATUS = "CPU"


print("\n" + "=" * 60)
print("GPU STATUS")
print("=" * 60)

print("YOLO   :", YOLO_STATUS)
print("ArcFace:", ARCFACE_STATUS)


# =========================================================
# ESTIMATE LANDMARKS
# =========================================================

def estimate_landmarks(box):

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    landmarks = np.array([

        # Left eye
        [
            x1 + 0.30 * w,
            y1 + 0.38 * h
        ],

        # Right eye
        [
            x1 + 0.70 * w,
            y1 + 0.38 * h
        ],

        # Nose
        [
            x1 + 0.50 * w,
            y1 + 0.55 * h
        ],

        # Left mouth
        [
            x1 + 0.35 * w,
            y1 + 0.75 * h
        ],

        # Right mouth
        [
            x1 + 0.65 * w,
            y1 + 0.75 * h
        ]

    ], dtype=np.float32)

    return landmarks


# =========================================================
# CREATE FACE EMBEDDING
# =========================================================

def get_embedding(frame, box):

    x1, y1, x2, y2 = map(
        int,
        box
    )

    h, w = frame.shape[:2]

    # -----------------------------------------
    # Keep coordinates inside image
    # -----------------------------------------

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:

        return None

    # -----------------------------------------
    # Estimate landmarks
    # -----------------------------------------

    landmarks = estimate_landmarks(
        [x1, y1, x2, y2]
    )

    # -----------------------------------------
    # Align face
    # -----------------------------------------

    aligned_face = norm_crop(
        frame,
        landmarks,
        image_size=112
    )

    # -----------------------------------------
    # ArcFace
    # -----------------------------------------

    embedding = recognizer.get_feat(
        aligned_face
    )
    embedding = embedding.flatten()

    # -----------------------------------------
    # Normalize
    # -----------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return None

    embedding = embedding / norm

    return embedding.astype(
        np.float32
    )


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    embedding1,
    embedding2
):

    return np.dot(
        embedding1,
        embedding2
    )


# =========================================================
# OPEN WEBCAM
# =========================================================

print("\n" + "=" * 60)
print("Opening webcam")
print("=" * 60)

cap = cv2.VideoCapture(0)

# Resolution

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

# Try to reduce camera buffering

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# =========================================================
# FPS VARIABLES
# =========================================================

previous_time = time.perf_counter()

fps = 0.0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # -----------------------------------------
    # Read frame
    # -----------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Could not read webcam.")

        break


    # -----------------------------------------
    # YOLO
    # -----------------------------------------

    results = yolo.predict(

        source=frame,

        device=YOLO_DEVICE,

        conf=CONFIDENCE,

        verbose=False

    )


    boxes = results[0].boxes


    # -----------------------------------------
    # Process faces
    # -----------------------------------------

    if boxes is not None and len(boxes) > 0:

        for box in boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # ---------------------------------
            # Create embedding
            # ---------------------------------

            embedding = get_embedding(

                frame,

                [x1, y1, x2, y2]

            )


            if embedding is None:

                continue


            # ---------------------------------
            # Compare with database
            # ---------------------------------

            similarities = np.dot(

                known_embeddings,

                embedding

            )


            # ---------------------------------
            # Find best match
            # ---------------------------------

            best_index = np.argmax(
                similarities
            )

            best_similarity = similarities[
                best_index
            ]

            best_name = known_names[
                best_index
            ]


            # ---------------------------------
            # Recognition decision
            # ---------------------------------

            if (
                best_similarity >=
                RECOGNITION_THRESHOLD
            ):

                name = best_name

            else:

                name = "Unknown"


            # ---------------------------------
            # Draw bounding box
            # ---------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            # ---------------------------------
            # Text
            # ---------------------------------

            label = (

                f"{name} "

                f"{best_similarity:.2f}"

            )


            cv2.putText(

                frame,

                label,

                (x1, max(30, y1 - 10)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2

            )


    # =====================================================
    # FPS
    # =====================================================

    current_time = time.perf_counter()

    time_difference = (

        current_time -

        previous_time

    )


    if time_difference > 0:

        current_fps = (

            1.0 /

            time_difference

        )

        # Smooth FPS

        if fps == 0:

            fps = current_fps

        else:

            fps = (

                0.9 * fps +

                0.1 * current_fps

            )


    previous_time = current_time


    # =====================================================
    # DISPLAY FPS
    # =====================================================

    cv2.putText(

        frame,

        f"FPS: {fps:.1f}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2

    )


    # =====================================================
    # DISPLAY YOLO STATUS
    # =====================================================

    cv2.putText(

        frame,

        f"YOLO: {YOLO_STATUS}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    # =====================================================
    # DISPLAY ARCFACE STATUS
    # =====================================================

    arcface_color = (
        (0, 255, 0)
        if ARCFACE_STATUS == "CUDA"
        else
        (0, 0, 255)
    )


    cv2.putText(

        frame,

        f"ArcFace: {ARCFACE_STATUS}",

        (20, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        arcface_color,

        2

    )


    # =====================================================
    # DISPLAY GPU
    # =====================================================

    cv2.putText(

        frame,

        "GPU: NVIDIA GTX 1650",

        (20, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    # =====================================================
    # DISPLAY RECOGNITION THRESHOLD
    # =====================================================

    cv2.putText(

        frame,

        f"Threshold: {RECOGNITION_THRESHOLD:.2f}",

        (20, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2

    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(

        "YOLO Face Recognition",

        frame

    )


    # -----------------------------------------
    # Press Q to exit
    # -----------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("\nWebcam closed.")