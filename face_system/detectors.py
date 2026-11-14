import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceDetection:
    """Represents a single detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None  # [(re_x, re_y), (le_x, le_y), (nose_x, nose_y), (rm_x, rm_y), (lm_x, lm_y)]
    raw: Optional[np.ndarray] = None  # 15-dim raw vector [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]


class BaseDetector(ABC):
    """Abstract base class for face detectors."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the detector."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the given BGR image.
        Returns a list of FaceDetection objects.
        """
        pass


class YuNetDetector(BaseDetector):
    """
    DNN-based Face Detector using OpenCV's YuNet ONNX model.
    Provides fast, highly accurate bounding boxes and 5 facial landmarks.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._current_input_size = (320, 320)

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self._current_input_size,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    @property
    def name(self) -> str:
        return "DNN (YuNet)"

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        if self._current_input_size != (w, h):
            self._current_input_size = (w, h)
            self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return []

        detections = []
        for face in faces:
            # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, w_box, h_box = face[0:4].astype(int)
            score = float(face[14])

            # Ensure bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w_box = min(w_box, w - x)
            h_box = min(h_box, h - y)

            landmarks = [
                (float(face[4]), float(face[5])),   # Right eye
                (float(face[6]), float(face[7])),   # Left eye
                (float(face[8]), float(face[9])),   # Nose tip
                (float(face[10]), float(face[11])), # Right mouth corner
                (float(face[12]), float(face[13]))  # Left mouth corner
            ]

            detections.append(FaceDetection(
                bbox=(x, y, w_box, h_box),
                confidence=score,
                landmarks=landmarks,
                raw=face.copy()
            ))

        return detections


class HaarCascadeDetector(BaseDetector):
    """
    Classic Computer Vision Face Detector using OpenCV's Haar Feature-based Cascade Classifiers.
    Fast, CPU-friendly, does not require GPU.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ):
        if cascade_path is None:
            # Check local models directory first
            local_model = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                "haarcascade_frontalface_default.xml"
            )
            if os.path.exists(local_model):
                cascade_path = local_model
            else:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade XML not found at: {cascade_path}")

        self.cascade_path = cascade_path
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        self.cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {self.cascade_path}")

    @property
    def name(self) -> str:
        return "Haar Cascade"

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better contrast under varying lighting
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        detections = []
        for (x, y, w, h) in faces:
            # Estimate anthropometric 5 landmarks to allow alignment with SFace
            re_x, re_y = float(x + 0.32 * w), float(y + 0.38 * h)
            le_x, le_y = float(x + 0.68 * w), float(y + 0.38 * h)
            nt_x, nt_y = float(x + 0.50 * w), float(y + 0.56 * h)
            rm_x, rm_y = float(x + 0.35 * w), float(y + 0.76 * h)
            lm_x, lm_y = float(x + 0.65 * w), float(y + 0.76 * h)

            landmarks = [
                (re_x, re_y),
                (le_x, le_y),
                (nt_x, nt_y),
                (rm_x, rm_y),
                (lm_x, lm_y)
            ]

            raw_vec = np.array([
                float(x), float(y), float(w), float(h),
                re_x, re_y,
                le_x, le_y,
                nt_x, nt_y,
                rm_x, rm_y,
                lm_x, lm_y,
                0.90  # Haar doesn't output confidence score, nominal 0.90 assigned
            ], dtype=np.float32)

            detections.append(FaceDetection(
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=0.90,
                landmarks=landmarks,
                raw=raw_vec
            ))

        return detections


def get_detector(
    detector_type: str = "dnn",
    yunet_model_path: Optional[str] = None
) -> BaseDetector:
    """
    Factory function to instantiate face detector.
    detector_type: 'dnn' or 'yunet' for DNN detector; 'haar' or 'cascade' for Haar Cascade.
    """
    norm_type = detector_type.lower().strip()
    if norm_type in ("dnn", "yunet"):
        if yunet_model_path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models",
                "face_detection_yunet_2023mar.onnx"
            )
            yunet_model_path = default_path
        return YuNetDetector(model_path=yunet_model_path)
    elif norm_type in ("haar", "cascade", "haarcascade"):
        return HaarCascadeDetector()
    else:
        raise ValueError(f"Unknown detector type '{detector_type}'. Choose 'dnn' or 'haar'.")
