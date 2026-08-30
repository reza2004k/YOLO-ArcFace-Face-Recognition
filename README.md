# YOLO + ArcFace Face Recognition

## Project Overview

This project is a **real-time face recognition system** built with two main parts:

1. **Build Database**
2. **Webcam Face Recognition**

The overall pipeline is:

```text
Image / Webcam
      ↓
     YOLO
      ↓
Face Detection
      ↓
Face Landmarks
      ↓
Face Alignment
      ↓
    ArcFace
      ↓
512-D Face Embedding
      ↓
Cosine Similarity
      ↓
Name / Unknown
```

---

# 1. Build Database

The first part creates a face database from images stored in the `faces/` directory.

For example:

```text
faces/
├── Barack Obama.jpg
├── Donald Trump.jpg
├── Elon Musk.jpg
├── Mark Zuckerberg.jpg

```

Each image represents a person whose face we want the system to recognize.

The process for every image is:

```text
Image
 ↓
YOLO Face Detection
 ↓
Face Landmarks
 ↓
Face Alignment
 ↓
ArcFace
 ↓
512-D Embedding
 ↓
Save to Database
```

---

## 2. Face Detection with YOLO

The project uses **YOLO (You Only Look Once)** for face detection.

YOLO's job is to answer:

> **Where is the face in the image?**

It returns a **bounding box** around the detected face:

```text
(x1, y1, x2, y2)
```

The project uses a face-specific YOLO model:

```text
yolov11n-face.pt
```

This model is specifically trained for **face detection**.

YOLO is responsible for **detection**, not recognition.

```text
YOLO → "There is a face here."
ArcFace → "This face is probably Barack Obama."
```

---

# 3. Face Landmarks

After detecting the face, we need to know where important facial points are located.

These points can include:

* Left eye
* Right eye
* Nose
* Left mouth corner
* Right mouth corner

For more detailed landmark models, many additional points can be detected around the eyes, nose, mouth, and face contour/jaw.

Instead of manually guessing these coordinates from the bounding box, it is better to use a **Face Landmark Model**.

A landmark model is a neural network trained to look at a face and predict the coordinates of facial keypoints.

For example:

```text
Input: Face image

        ↓

Face Landmark Model

        ↓

(eye_x, eye_y)
(nose_x, nose_y)
(mouth_x, mouth_y)
...
```

These coordinates are stored as the `landmarks` array.

---

# 4. Face Alignment

The detected face may be rotated, tilted, or positioned differently from one image to another.

For example:

```text
     ↙ Face              Face →
```

ArcFace works better when faces are presented in a consistent format.

Therefore, the landmarks are used to **align the face**.

The alignment process geometrically transforms the face so that important points such as the eyes and nose are placed in approximately standard positions.

The result is an aligned face, typically resized to:

```text
112 × 112
```

The pipeline becomes:

```text
Original Image
      ↓
Detected Face
      ↓
Landmarks
      ↓
Alignment
      ↓
112 × 112 Aligned Face
```

---

# 5. ArcFace

After face alignment, the aligned face is passed to ArcFace.

ArcFace's job is to convert the face image into a numerical representation called a face embedding.

The model used in this project is:

w600k_r50.onnx

This is an ONNX version of an ArcFace recognition model based on a ResNet-50 backbone.

The model is executed using ONNX Runtime with CUDAExecutionProvider, meaning that ArcFace is currently running on the NVIDIA GTX 1650 GPU, rather than the CPU.

The current execution architecture is:

Aligned Face
     ↓
ArcFace
     ↓
ONNX Runtime
     ↓
CUDAExecutionProvider
     ↓
NVIDIA GTX 1650
     ↓
512-D Embedding

The neural network processes the aligned face through its learned layers and extracts high-level facial representations.

It does not simply store raw pixels.

The network learns patterns and features useful for distinguishing identities, such as facial structure and combinations of visual patterns around the eyes, nose, mouth, and overall face shape.

The final result is a 512-dimensional vector:

[
  0.021,
 -0.134,
  0.452,
  ...
  0.087
]

This vector is called the face embedding.

So:

112 × 112 Face
      ↓
   ArcFace
      ↓
ONNX Runtime
      ↓
CUDA / NVIDIA GPU
      ↓
512 numbers

---

# 6. Building `face_database.pkl`

After generating an embedding for every person, the project stores the embeddings together with their names.

The database has a dictionary structure:

```python
database = {
    "embeddings": known_embeddings,
    "names": known_names
}
```

For example:

```text
known_embeddings
    ↓
[
    embedding_of_Barack Obama,
    embedding_of_Donald Trump,
    embedding_of_Elon Musk,
    ...
]

known_names
    ↓
[
    "Barack Obama",
    "Donald Trump",
    "Elon Musk",
    ...
]
```

The dictionary is then serialized using Python `pickle` and saved as:

```text
face_database.pkl
```

The `.pkl` file is **not the ArcFace model**.

It is simply our local face database containing the embeddings and corresponding names.

---

# 7. Webcam Face Recognition

The second part of the project is the real-time webcam system.

The webcam pipeline is almost identical to the database-building process.

```text
Webcam Frame
      ↓
YOLO
      ↓
Face Detection
      ↓
Face Landmarks
      ↓
Face Alignment
      ↓
ArcFace
      ↓
512-D Embedding
      ↓
Compare with Database
      ↓
Name / Unknown
```

---

# 8. YOLO and ArcFace on the GPU

The system uses the NVIDIA GPU for both major neural-network components.

YOLO

YOLO runs through PyTorch and CUDA:

YOLO
 ↓
PyTorch
 ↓
CUDA
 ↓
NVIDIA GTX 1650
ArcFace

ArcFace runs through ONNX Runtime and CUDA:

ArcFace
 ↓
ONNX Runtime
 ↓
CUDAExecutionProvider
 ↓
NVIDIA GTX 1650

Therefore, the main computational pipeline is GPU-accelerated:

                    NVIDIA GTX 1650
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
           YOLO                       ArcFace
        Detection                   Recognition
             │                           │
       PyTorch/CUDA              ONNX Runtime/CUDA
             │                           │
             └─────────────┬─────────────┘
                           ↓
                    Real-Time Result

The landmark model can also use ONNX Runtime's CUDA execution provider when configured accordingly.

---

# 9. Generating the Webcam Embedding

When YOLO detects a face in the webcam frame, the same processing pipeline used during database creation is applied:

```text
Detected Face
     ↓
Landmarks
     ↓
Alignment
     ↓
ArcFace
     ↓
512-D Embedding
```

This is important because the database embeddings and webcam embeddings must be generated using the **same preprocessing and recognition model**.

---

# 10. Comparing Face Embeddings

The webcam embedding is compared with all embeddings stored in:

```text
face_database.pkl
```

The project uses **cosine similarity**.

Because the embeddings are normalized, cosine similarity can be calculated using the dot product:

```python
similarity = np.dot(embedding1, embedding2)
```

The system calculates the similarity between the webcam face and every stored face.

For example:

```text
Webcam face
    ↓
Compare with Barack Obama      → 0.82
Compare with Donald Trump      → 0.31
Compare with Elon Mus          → 0.27
Compare with Mark Zuckerberg   → 0.42
```

The highest similarity is selected.

In this example:

```text
Barack Obama → 0.82
```

is the best match.

---

## 11. Dot Product and Cosine Similarity

The dot product of two vectors is related to the angle between them:

Example

Consider two vectors:

$$ \mathbf{a} = [1,1] $$

and

$$ \mathbf{b} = [2,2] $$
Step 1: Calculate the dot product
$$ \mathbf{a} \cdot \mathbf{b} = (1 \times 2) + (1 \times 2) = 4 $$
Step 2: Calculate the magnitudes
$$ |\mathbf{a}| = \sqrt{1^2 + 1^2} = \sqrt{2} $$ $$ |\mathbf{b}| = \sqrt{2^2 + 2^2} = \sqrt{8} = 2\sqrt{2} $$
Step 3: Calculate cosine similarity
$$ \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}} {|\mathbf{a}|\,|\mathbf{b}|} $$

Substituting the values:

$$ \cos(\theta) = \frac{4} {\sqrt{2} \times 2\sqrt{2}} $$ $$ \cos(\theta) = \frac{4}{4} = 1 $$

Therefore:

$$ \cos(\theta) = 1 $$


In face recognition, a cosine similarity close to `1` means that the two normalized embeddings are very similar in the learned feature space.

---

# 12. Recognition Threshold

The project uses a recognition threshold:

```python
RECOGNITION_THRESHOLD = 0.45
```

The system first finds the person with the highest similarity.

Then:

```text
Similarity ≥ 0.45
        ↓
Accept the match
        ↓
Display the person's name
```

Otherwise:

```text
Similarity < 0.45
        ↓
Reject the match
        ↓
Display "Unknown"
```

For example:

```text
Best match:
Barack Obama → 0.82

0.82 ≥ 0.45

Result:
Barack Obama
```

While:

```text
Best match:
Donald Trump → 0.32

0.32 < 0.45

Result:
Unknown
```

The threshold is an application parameter and should ideally be tuned using validation data rather than assumed to be universally optimal.

---

# 13. Buffalo Model

The **Buffalo_L** model package from InsightFace contains several different neural networks for different tasks.

The important models used in this project are:

| Model            | Purpose                                   |
| ---------------- | ----------------------------------------- |
| `det_10g.onnx`   | Face detection                            |
| `2d106det.onnx`  | 2D facial landmark detection              |
| `w600k_r50.onnx` | ArcFace face recognition                  |
| `genderage.onnx` | Gender and age estimation                 |
| `1k3d68.onnx`    | 3D/68-point facial landmark-related model |

In this project, the important pipeline is:

```text
YOLO
 ↓
Face Detection

2d106det
 ↓
Face Landmarks

Alignment
 ↓
Aligned Face

w600k_r50
 ↓
512-D Embedding
```

Using a real landmark model is preferable to manually estimating landmark positions from the YOLO bounding box because the landmark network **actually analyzes the face and predicts the keypoint coordinates**.

---

# 14. Why CUDA Is Important

For real-time performance, we want the computationally expensive neural networks to run on the GPU.

The current architecture uses:

YOLO
 ↓
PyTorch + CUDA
 ↓
NVIDIA GTX 1650

and:

ArcFace
 ↓
ONNX Runtime
 ↓
CUDAExecutionProvider
 ↓
NVIDIA GTX 1650

Therefore, the desired final pipeline is:

Webcam
  ↓
YOLO + CUDA
  ↓
Face Bounding Box
  ↓
Face Landmark Model
  ↓
Facial Keypoints
  ↓
Face Alignment
  ↓
ArcFace + CUDA
  ↓
512-D Embedding
  ↓
Cosine Similarity
  ↓
Name / Unknown

The important point is that ArcFace is now confirmed to be executing on the GPU.

The ONNX Runtime session reports:

Session providers:
['CUDAExecutionProvider', 'CPUExecutionProvider']

The presence of CUDAExecutionProvider as the active first provider confirms that the ArcFace ONNX session is configured to execute using CUDA on the NVIDIA GPU.

# 15. Final Architecture

The complete project can be summarized as:

                         ┌───────────────┐
                         │    Webcam     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │     YOLO      │
                         │ Face Detection│
                         │   CUDA/GPU    │
                         └───────┬───────┘
                                 │
                           Bounding Box
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │ Face Landmark Model  │
                    │     2d106det.onnx    │
                    └──────────┬───────────┘
                               │
                         Facial Keypoints
                               │
                               ▼
                       ┌──────────────┐
                       │Face Alignment│
                       └──────┬───────┘
                              │
                         112 × 112 Face
                              │
                              ▼
                       ┌──────────────┐
                       │   ArcFace    │
                       │ w600k_r50    │
                       │   CUDA/GPU   │
                       └──────┬───────┘
                              │
                         512-D Vector
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Cosine Similarity      │
                 │                         │
                 │       ↕                 │
                 │ face_database.pkl       │
                 └───────────┬─────────────┘
                             │
                             ▼
                       Name / Unknown
# Summary

The project separates detection from recognition:

YOLO
→ Finds the face
→ Runs on NVIDIA GPU using PyTorch/CUDA

Face Landmark Model
→ Finds important facial keypoints

Face Alignment
→ Standardizes the face

ArcFace
→ Converts the face into a 512-D embedding
→ Runs on NVIDIA GPU using ONNX Runtime/CUDAExecutionProvider

Cosine Similarity
→ Compares the embedding with known embeddings

Threshold
→ Decides whether the identity is accepted

face_database.pkl
→ Stores known embeddings and their names
Current GPU status
GPU:
NVIDIA GeForce GTX 1650

YOLO:
CUDA / GPU

ArcFace:
CUDA / GPU

ONNX Runtime:
1.23.2

ArcFace provider:
CUDAExecutionProvider
