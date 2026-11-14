# Biometric Face Detection & Recognition System

A real-time Computer Vision system capable of detecting human faces and recognizing identities using **Deep Feature Embeddings (SFace)** and dual detection algorithms: **DNN (YuNet)** and **Haar Feature-based Cascades**.

---

## Highlights & Features

- **Dual Detection Engine**:
  - **DNN (YuNet)**: Fast, modern ONNX deep learning detector that outputs bounding boxes, confidence scores, and 5 facial landmarks (eyes, nose, mouth corners) even under head rotations and varied lighting.
  - **Haar Cascades**: Classic Viola-Jones cascade classifier with histogram equalization and estimated landmark alignment.
  - **Runtime Switcher**: Hot-swap between DNN and Haar Cascade on-the-fly in the live video feed by pressing `m`.
- **Deep Feature Recognition (SFace)**:
  - Extracts 128-dimensional L2-normalized feature embeddings from aligned face crops.
  - Matches faces against known identities using Cosine Similarity ($[-1, 1]$ metric) or Euclidean L2 distance.
  - Calibrated default cosine threshold `0.363` (empirically validated on LFW dataset).
- **Persistent Database & Caching**:
  - Automatically loads and indexes faces from the `known_faces/` folder.
  - Caches precomputed embeddings in `database/embeddings.pkl` with file timestamp validation for millisecond startup.
  - Supports multiple reference photos per identity.
- **Biometric HUD Overlay**:
  - Stylized biometric corner brackets and bounding boxes (Emerald Green for recognized, Coral Red for unknown).
  - Floating identity badges with name and similarity percentage.
  - 5-point facial landmark tracking dots.
  - Top telemetry bar showing FPS, active detector, face counts, and similarity threshold.
- **Interactive Live Feed & Enrollment**:
  - Live webcam stream with real-time recognition.
  - One-click face enrollment from the live stream (press `e`).
  - High-resolution snapshot capture (press `s`).
  - Batch image and prerecorded video processing modes.

---

## Project Structure

```
face_recognition_system/
├── models/                               # Pretrained ONNX & XML models
│   ├── face_detection_yunet_2023mar.onnx # DNN Face Detector
│   ├── face_recognition_sface_2021dec.onnx # SFace 128-D Feature Extractor
│   └── haarcascade_frontalface_default.xml # Haar Cascade Classifier
├── known_faces/                          # Reference dataset of enrolled persons
│   ├── Alice/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Bob.jpg
├── database/
│   └── embeddings.pkl                    # High-speed cached embeddings
├── test_samples/                         # Sample queries and composite images
├── output/                               # Saved results, snapshots, and processed videos
├── face_system/                          # Core Python package
│   ├── __init__.py
│   ├── detectors.py                      # YuNet & Haar Cascade detector implementations
│   ├── recognizer.py                     # SFace feature extractor, matcher, & database
│   ├── visualizer.py                     # Biometric HUD, badges, corners, and FPS counter
│   └── utils.py                          # Camera initialization & model auto-downloader
├── main.py                               # Primary application entry point (live/image/video)
├── enroll.py                             # Identity enrollment CLI (webcam or file)
├── setup_samples.py                      # Automatic sample dataset downloader
├── download_models.py                    # Model download helper
└── requirements.txt                      # Project dependencies
```

---

## Installation & Setup

### 1. Requirements

- Python 3.8+ (tested and compatible with Python 3.10 – 3.14)
- Webcam (optional, for live mode)

### 2. Install Dependencies

```bash
cd face_recognition_system
pip install -r requirements.txt
```

### 3. Setup Sample Reference Faces (Optional)

To immediately populate `known_faces/` with sample identities and generate test images:

```bash
python setup_samples.py
```

---

## Usage Guide

### 1. Live Webcam Feed

Launch real-time recognition using your default camera:

```bash
python main.py --mode live
```

Or specify a different webcam index:
```bash
python main.py --mode live --source 1
```

#### Live Interactive Keyboard Controls

| Key | Action |
|---|---|
| `m` | Switch face detector between **DNN (YuNet)** and **Haar Cascade** |
| `e` | **Quick Enroll**: Pause frame and enroll current face into database |
| `s` | **Snapshot**: Capture and save frame to `output/snapshot_<timestamp>.jpg` |
| `+` / `=` | Increase recognition confidence threshold (+0.02) |
| `-` | Decrease recognition confidence threshold (-0.02) |
| `q` or `ESC` | Exit application |

---

### 2. Process Still Images

Run face detection and recognition on any photo:

```bash
python main.py --mode image --source test_samples/test_dual_faces.jpg
```

Save result to custom output path:
```bash
python main.py --mode image --source photo.jpg --output output/result.jpg --no-gui
```

Choose specific detector (DNN or Haar):
```bash
python main.py --mode image --source photo.jpg --detector haar
```

---

### 3. Process Prerecorded Videos

Process a video file and save the annotated output video:

```bash
python main.py --mode video --source input_video.mp4 --output output/processed_video.mp4
```

---

### 4. Enrolling New Faces

You can enroll new individuals using either the CLI tool `enroll.py`, direct file placement, or inside the live feed.

#### A. Interactive Live Webcam Capture:
```bash
python enroll.py --name "John Doe" --webcam
```
*Look at the camera and press `SPACE` when the green bounding box appears.*

#### B. From an Image File:
```bash
python enroll.py --name "Jane Smith" --image path/to/jane.jpg
```

#### C. Manual Folder Placement:
Create a folder inside `known_faces/` named after the person (e.g. `known_faces/Elon Musk/`) and add one or more `.jpg` or `.png` photos. The system will automatically index them on the next run.

#### D. Manage Database:
- **List all enrolled persons**:
  ```bash
  python enroll.py --list
  ```
- **Delete an enrolled identity**:
  ```bash
  python enroll.py --delete "John Doe"
  ```
- **Force rebuild cache**:
  ```bash
  python main.py --rebuild-db
  ```

---

## Technical Details

### Face Alignment & Normalization
SFace achieves high accuracy by aligning detected facial landmarks.
1. The 5 key landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner) are detected.
2. An affine transformation rotates and scales the face so that the eyes and mouth sit in standard positions.
3. The aligned face is normalized to $112 \times 112$ pixels before feature extraction.

### Feature Embeddings & Matching
- **Vector Space**: Each face is mapped to a 128-dimensional hypersphere ($\mathbb{R}^{128}$) with unit L2 norm ($||\mathbf{v}||_2 = 1$).
- **Cosine Similarity**:
  $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}||_2 ||\mathbf{v}||_2} = \mathbf{u} \cdot \mathbf{v}$$
- When multiple reference photos exist for an identity, the system compares against all embeddings and selects the highest similarity score.
