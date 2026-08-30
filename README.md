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

# 2. Face Detection with YOLO

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

Instead of manually guessing these coordinates from the bounding box, the project uses a **Face Landmark Model**.

A landmark model is a neural network trained to look at a face and predict the coordinates of facial keypoints.

The relevant landmark model from the InsightFace Buffalo_L package is:

```text
2d106det.onnx
```

This model predicts detailed 2D facial landmarks instead of relying on fixed estimated positions inside the YOLO bounding box.

The pipeline is:

```text
Input Face
     ↓
Face Landmark Model
     ↓
Facial Keypoints
```

---

# 4. Face Alignment

The detected face may be rotated, tilted, or positioned differently from one image to another.

For example:

```text
     ↙ Face              Face →
```

ArcFace works better when faces are presented in a consistent format.

Therefore, the facial landmarks are used to **align the face**.

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
Facial Landmarks
      ↓
Alignment
      ↓
112 × 112 Aligned Face
```

---

# 5. ArcFace

After alignment, the face is passed to **ArcFace**.

ArcFace's job is to convert the face image into a numerical representation called a **face embedding**.

The model used in this project is:

```text
w600k_r50.onnx
```

This is an **ONNX version of an ArcFace recognition model based on a ResNet-50 backbone**.

The neural network processes the face through many learned layers and extracts high-level facial representations.

It does not simply store raw pixels.

The network learns patterns and features useful for distinguishing identities, such as facial structure and combinations of visual patterns around areas such as the eyes, nose, mouth, and overall face shape.

The final result is a **512-dimensional vector**:

```text
[
  0.021,
 -0.134,
  0.452,
  ...
  0.087
]
```

This vector is called the **face embedding**.

So:

```text
Face Image
     ↓
ArcFace
     ↓
512-D Face Embedding
```

---

# 6. ArcFace GPU Acceleration

In the current implementation, **ArcFace runs on the NVIDIA GPU using ONNX Runtime and CUDA**.

The execution architecture is:

```text
ArcFace
   ↓
ONNX Runtime
   ↓
CUDAExecutionProvider
   ↓
NVIDIA GeForce GTX 1650
```

The project loads ArcFace using:

```python
recognizer = get_model(
    ARCFACE_MODEL,
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)
```

The model is then prepared with:

```python
recognizer.prepare(ctx_id=0)
```

The actual ONNX Runtime session can be verified using:

```python
recognizer.session.get_providers()
```

The expected result is:

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

The first provider is:

```text
CUDAExecutionProvider
```

which means that ArcFace inference is running through the CUDA execution provider on the NVIDIA GPU.

The current GPU used by the project is:

```text
NVIDIA GeForce GTX 1650
```

Therefore, the recognition pipeline is:

```text
Face
 ↓
Alignment
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
```

This allows ArcFace to participate in the real-time GPU-accelerated face recognition pipeline.

---

# 7. Building `face_database.pkl`

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

# 8. Webcam Face Recognition

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

# 9. YOLO on the GPU

YOLO runs using the NVIDIA GPU through CUDA.

Instead of processing the image only on the CPU:

```text
YOLO
 ↓
CPU
```

the project uses:

```text
YOLO
 ↓
PyTorch
 ↓
CUDA
 ↓
NVIDIA GPU
```

This significantly improves the speed of real-time face detection.

The current GPU used by the project is:

```text
NVIDIA GeForce GTX 1650
```

---

# 10. Generating the Webcam Embedding

When YOLO detects a face in the webcam frame, the same processing pipeline used during database creation is applied:

```text
Detected Face
     ↓
Face Landmark Model
     ↓
Facial Keypoints
     ↓
Alignment
     ↓
ArcFace
     ↓
512-D Embedding
```

This is important because the database embeddings and webcam embeddings must be generated using the **same preprocessing and recognition model**.

---

# 11. Comparing Face Embeddings

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
Compare with Elon Musk         → 0.27
Compare with Mark Zuckerberg   → 0.42
```

The highest similarity is selected.

In this example:

```text
Barack Obama → 0.82
```

is the best match.

---

# 12. Dot Product and Cosine Similarity

The project uses **cosine similarity** to compare face embeddings.

Cosine similarity measures the angle between two vectors.

The general formula is:

$$
\text{cosine similarity}
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

Where:

```text
a · b   = Dot Product of vectors a and b
||a||   = Magnitude (L2 norm) of vector a
||b||   = Magnitude (L2 norm) of vector b
```

---

## Example

Consider two vectors:

$$
\mathbf{a} = [1,1]
$$

$$
\mathbf{b} = [2,2]
$$

### Step 1: Calculate the Dot Product

The dot product is:

$$
\mathbf{a}\cdot\mathbf{b}
=
(a_1b_1)+(a_2b_2)
$$

Substituting the values:

$$
\mathbf{a}\cdot\mathbf{b}
=
(1\times2)+(1\times2)
$$

Therefore:

$$
\mathbf{a}\cdot\mathbf{b}=4
$$

---

## Step 2: Calculate the Magnitude of Vector a

The magnitude is:

$$
\|\mathbf{a}\|
=
\sqrt{a_1^2+a_2^2}
$$

Substituting:

$$
\|\mathbf{a}\|
=
\sqrt{1^2+1^2}
$$

Therefore:

$$
\|\mathbf{a}\|=\sqrt{2}
$$

---

## Step 3: Calculate the Magnitude of Vector b

Similarly:

$$
\|\mathbf{b}\|
=
\sqrt{b_1^2+b_2^2}
$$

Substituting:

$$
\|\mathbf{b}\|
=
\sqrt{2^2+2^2}
$$

Therefore:

$$
\|\mathbf{b}\|
=
\sqrt{4+4}
=
\sqrt{8}
=
2\sqrt{2}
$$

---

## Step 4: Calculate Cosine Similarity

The formula is:

$$
\cos(\theta)
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|\|\mathbf{b}\|}
$$

Substituting the calculated values:

$$
\cos(\theta)
=
\frac{4}
{\sqrt{2}\times2\sqrt{2}}
$$

Since:

$$
\sqrt{2}\times2\sqrt{2}=4
$$

we get:

$$
\cos(\theta)
=
\frac{4}{4}
$$

Therefore:

$$
\boxed{\cos(\theta)=1}
$$

This means the two vectors point in exactly the same direction.

---

## Why Can We Use Dot Product Directly?

In this project, face embeddings are **L2-normalized** before comparison.

Normalization is performed using:

$$
\mathbf{a}_{normalized}
=
\frac{\mathbf{a}}{\|\mathbf{a}\|}
$$

After normalization:

$$
\|\mathbf{a}_{normalized}\|=1
$$

The same applies to the second embedding:

$$
\|\mathbf{b}_{normalized}\|=1
$$

Therefore, the cosine similarity formula becomes:

$$
\cos(\theta)
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{1\times1}
$$

which simplifies to:

$$
\boxed{\cos(\theta)=\mathbf{a}\cdot\mathbf{b}}
$$

Therefore, the project can calculate cosine similarity directly using:

```python
similarity = np.dot(
    embedding1,
    embedding2
)
```

For comparing one webcam embedding against the complete database:

```python
similarities = np.dot(
    known_embeddings,
    embedding
)
```

The process is therefore:

```text
Normalized Embeddings
        ↓
     Dot Product
        ↓
Cosine Similarity
        ↓
Similarity Score
```

---

## Interpretation of Similarity

Cosine similarity is bounded by:

$$
-1 \leq \cos(\theta) \leq 1
$$

In general:

```text
Similarity ≈ 1
        ↓
Very similar vector directions
```

```text
Similarity ≈ 0
        ↓
Little directional similarity
```

```text
Similarity ≈ -1
        ↓
Opposite vector directions
```

For face recognition, a higher similarity generally indicates that two face embeddings are closer in the learned feature space.

---

# 13. Recognition Threshold

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

# 14. Buffalo Model

The **Buffalo_L** model package from InsightFace contains several different neural networks for different tasks.

The important models used in this project are:

| Model            | Purpose                                   |
| ---------------- | ----------------------------------------- |
| `det_10g.onnx`   | Face detection                            |
| `2d106det.onnx`  | Detailed 2D facial landmark detection     |
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

The landmark model is used to obtain actual facial keypoints rather than estimating eye, nose, and mouth positions from simple fixed percentages of the YOLO bounding box.

This makes the alignment process more robust because the landmark model analyzes the actual face.

---

# 15. Why CUDA Is Important

For real-time performance, we want the computationally expensive neural networks to run on the GPU.

The architecture is:

```text
                 NVIDIA GTX 1650
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
        YOLO                       ArcFace
      Detection                   Recognition
          │                           │
       PyTorch                    ONNX Runtime
          │                           │
        CUDA                 CUDAExecutionProvider
          │                           │
          └─────────────┬─────────────┘
                        ↓
                  Real-Time Result
```

YOLO uses:

```text
PyTorch
   ↓
CUDA
   ↓
NVIDIA GTX 1650
```

ArcFace uses:

```text
ONNX Runtime
   ↓
CUDAExecutionProvider
   ↓
NVIDIA GTX 1650
```

The landmark model can also be executed through ONNX Runtime with CUDA when configured with the CUDA execution provider.

Therefore, the final GPU-accelerated pipeline is:

```text
Webcam
  ↓
YOLO + CUDA
  ↓
Face Bounding Box
  ↓
Face Landmark Model + CUDA
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
```

---

# 16. Final Architecture

The complete project can be summarized as:

```text
                         ┌───────────────┐
                         │    Webcam     │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │     YOLO      │
                         │ Face Detection│
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
                       └──────┬───────┘
                              │
                       ONNX Runtime
                              │
                    CUDAExecutionProvider
                              │
                              ▼
                     NVIDIA GTX 1650
                              │
                              ▼
                         512-D Vector
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  Cosine Similarity      │
                 │                         │
                 │          ↕              │
                 │  face_database.pkl      │
                 └───────────┬─────────────┘
                             │
                             ▼
                       Name / Unknown
```

---

# Summary

The project separates **detection**, **landmark detection**, and **recognition**:

```text
YOLO
→ Finds the face

Face Landmark Model
→ Finds important facial keypoints

Face Alignment
→ Standardizes the face

ArcFace
→ Converts the face into a 512-D embedding

ONNX Runtime + CUDA
→ Runs ArcFace inference on the NVIDIA GTX 1650 GPU

Cosine Similarity
→ Compares the embedding with known embeddings

Threshold
→ Decides whether the identity is accepted

face_database.pkl
→ Stores known embeddings and their names
```

The complete system combines three main neural-network stages:

```text
YOLO
Detection
   ↓
2d106det
Landmark Detection
   ↓
ArcFace
Face Recognition
```

with GPU acceleration:

```text
YOLO
 ↓
PyTorch + CUDA
 ↓
NVIDIA GTX 1650

ArcFace
 ↓
ONNX Runtime
 ↓
CUDAExecutionProvider
 ↓
NVIDIA GTX 1650
```

This creates a complete **GPU-accelerated real-time face detection and recognition pipeline**.
