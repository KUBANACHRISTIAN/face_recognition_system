import os
import pickle
import glob
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .detectors import FaceDetection, BaseDetector, YuNetDetector, HaarCascadeDetector


class FaceRecognizer:
    """
    Face Feature Extractor & Matcher using OpenCV's SFace deep neural network.
    Extracts 128-dimensional normalized embeddings for face identification.
    """

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                "face_recognition_sface_2021dec.onnx"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SFace model file not found at: {model_path}")

        self.model_path = model_path
        self.recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

    def extract_feature(self, image: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract a 128-D feature embedding from a detected face.
        Uses facial landmarks for alignment when available.
        """
        if image is None or image.size == 0:
            return None

        try:
            if detection.raw is not None and len(detection.raw) >= 15:
                # Use SFace's built-in landmark alignment and crop
                aligned = self.recognizer.alignCrop(image, detection.raw)
            else:
                # Fallback: Crop bounding box with margin and resize to 112x112
                x, y, w, h = detection.bbox
                h_img, w_img = image.shape[:2]
                margin = int(max(w, h) * 0.1)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w_img, x + w + margin)
                y2 = min(h_img, y + h + margin)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    return None
                aligned = cv2.resize(crop, (112, 112))

            feature = self.recognizer.feature(aligned)
            # Ensure float32 and shape (1, 128)
            feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
            # L2 normalize
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm
            return feature
        except Exception as e:
            print(f"[Warning] Feature extraction failed: {e}")
            return None

    def match(
        self,
        feature1: np.ndarray,
        feature2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity or distance between two 128-D feature embeddings.
        - metric='cosine': returns cosine similarity (higher is more similar, typically in [-1, 1])
        - metric='l2': returns L2 distance (lower is more similar, >= 0)
        """
        f1 = feature1.reshape(1, -1)
        f2 = feature2.reshape(1, -1)

        if metric.lower() == "cosine":
            # Cosine similarity via dot product of normalized vectors
            sim = float(np.dot(f1, f2.T)[0, 0])
            return sim
        elif metric.lower() == "l2":
            dist = float(np.linalg.norm(f1 - f2))
            return dist
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'cosine' or 'l2'.")


class FaceDatabase:
    """
    Manages enrolled face identities and cached embeddings.
    Supports directory scanning, multi-photo per identity, and disk persistence.
    """

    def __init__(
        self,
        known_faces_dir: Optional[str] = None,
        cache_path: Optional[str] = None,
        recognizer: Optional[FaceRecognizer] = None,
        detector: Optional[BaseDetector] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(__file__))

        if known_faces_dir is None:
            known_faces_dir = os.path.join(base_dir, "known_faces")
        if cache_path is None:
            # If base directory is not writable (e.g. serverless runtime), use /tmp
            if not os.access(base_dir, os.W_OK):
                cache_path = os.path.join("/tmp", "database", "embeddings.pkl")
            else:
                cache_path = os.path.join(base_dir, "database", "embeddings.pkl")

        self.known_faces_dir = known_faces_dir
        self.cache_path = cache_path
        try:
            os.makedirs(self.known_faces_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        except OSError:
            pass

        self.recognizer = recognizer if recognizer else FaceRecognizer()
        self.detector = detector if detector else YuNetDetector(
            os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")
        )

        # Dictionary of person_name -> List of 128-D numpy embeddings
        self.identities: Dict[str, List[np.ndarray]] = {}
        self.load()

    def load(self, force_refresh: bool = False):
        """Load enrolled embeddings from disk cache or rebuild from image folder."""
        if not force_refresh and self._load_cache():
            return

        print("[FaceDatabase] Building face embeddings from dataset...")
        self.build()
        self._save_cache()

    def _get_files_metadata(self) -> Dict[str, float]:
        """Collect file paths and modification times to detect changes."""
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        meta = {}
        for root, _, files in os.walk(self.known_faces_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    full_path = os.path.join(root, file)
                    meta[full_path] = os.path.getmtime(full_path)
        return meta

    def _load_cache(self) -> bool:
        """Attempt loading from pkl cache if all file timestamps match."""
        if not os.path.exists(self.cache_path):
            return False

        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)

            cached_meta = data.get("meta", {})
            current_meta = self._get_files_metadata()

            if cached_meta != current_meta:
                print("[FaceDatabase] Dataset changes detected. Rebuilding cache...")
                return False

            self.identities = data.get("identities", {})
            total_faces = sum(len(embs) for embs in self.identities.values())
            print(f"[FaceDatabase] Loaded {len(self.identities)} people ({total_faces} embeddings) from cache.")
            return True
        except Exception as e:
            print(f"[FaceDatabase] Cache read error: {e}")
            return False

    def _save_cache(self):
        """Persist identities and metadata to cache."""
        try:
            meta = self._get_files_metadata()
            data = {
                "identities": self.identities,
                "meta": meta
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[FaceDatabase] Cached {len(self.identities)} identities to {self.cache_path}.")
        except Exception as e:
            print(f"[FaceDatabase] Cache write error: {e}")

    def build(self):
        """Scans known_faces_dir and extracts embeddings for all faces."""
        self.identities = {}
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if not os.path.exists(self.known_faces_dir):
            return

        entries = os.listdir(self.known_faces_dir)
        for entry in sorted(entries):
            entry_path = os.path.join(self.known_faces_dir, entry)

            if os.path.isdir(entry_path):
                # Directory named after person containing one or more photos
                person_name = entry
                image_files = [
                    os.path.join(entry_path, f)
                    for f in os.listdir(entry_path)
                    if os.path.splitext(f)[1].lower() in valid_exts
                ]
            elif os.path.isfile(entry_path):
                # Single photo file named PersonName.jpg
                ext = os.path.splitext(entry)[1].lower()
                if ext not in valid_exts:
                    continue
                person_name = os.path.splitext(entry)[0]
                image_files = [entry_path]
            else:
                continue

            for img_path in image_files:
                self._process_and_add(person_name, img_path)

        total_faces = sum(len(embs) for embs in self.identities.values())
        print(f"[FaceDatabase] Indexed {len(self.identities)} people with {total_faces} face samples.")

    def _process_and_add(self, person_name: str, img_path: str):
        """Reads an image, detects the principal face, and extracts its embedding."""
        img = cv2.imread(img_path)
        if img is None:
            print(f"[FaceDatabase] Warning: Failed to read image {img_path}")
            return

        detections = self.detector.detect(img)
        if not detections:
            print(f"[FaceDatabase] Warning: No face detected in {img_path}")
            return

        # Pick the largest face in case of multiple
        best_det = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
        emb = self.recognizer.extract_feature(img, best_det)
        if emb is not None:
            if person_name not in self.identities:
                self.identities[person_name] = []
            self.identities[person_name].append(emb)

    def enroll(self, person_name: str, image: np.ndarray, save_to_disk: bool = True) -> bool:
        """
        Enrolls a new person from an in-memory image or webcam frame.
        Optionally saves to known_faces folder.
        """
        detections = self.detector.detect(image)
        if not detections:
            return False

        best_det = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
        emb = self.recognizer.extract_feature(image, best_det)
        if emb is None:
            return False

        if person_name not in self.identities:
            self.identities[person_name] = []
        self.identities[person_name].append(emb)

        if save_to_disk:
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            existing_count = len(glob.glob(os.path.join(person_dir, "*.jpg")))
            filename = f"sample_{existing_count + 1}.jpg"
            save_path = os.path.join(person_dir, filename)

            # Crop face with margin or save full image
            cv2.imwrite(save_path, image)
            self._save_cache()

        return True

    def identify(
        self,
        feature: np.ndarray,
        cosine_threshold: float = 0.363,
        metric: str = "cosine"
    ) -> Tuple[str, float, bool]:
        """
        Compare an unknown face embedding against all enrolled identities.
        Returns: (name, similarity_score, is_match)
        """
        if not self.identities or feature is None:
            return "Unknown", 0.0, False

        best_person = "Unknown"
        best_score = -1.0 if metric == "cosine" else float("inf")
        is_match = False

        for name, embeddings in self.identities.items():
            for stored_emb in embeddings:
                score = self.recognizer.match(feature, stored_emb, metric=metric)

                if metric == "cosine":
                    if score > best_score:
                        best_score = score
                        best_person = name
                else:  # l2 metric
                    if score < best_score:
                        best_score = score
                        best_person = name

        if metric == "cosine":
            if best_score >= cosine_threshold:
                is_match = True
            else:
                best_person = "Unknown"
        else:
            if best_score <= cosine_threshold:
                is_match = True
            else:
                best_person = "Unknown"

        return best_person, float(best_score), is_match
