"""
Face Detection & Recognition System
Modular architecture supporting DNN (YuNet) and Haar Cascades detection,
and deep feature embedding recognition (SFace).
"""

from .detectors import BaseDetector, YuNetDetector, HaarCascadeDetector, FaceDetection, get_detector
from .recognizer import FaceRecognizer, FaceDatabase
from .visualizer import FaceVisualizer

__all__ = [
    "BaseDetector",
    "YuNetDetector",
    "HaarCascadeDetector",
    "FaceDetection",
    "get_detector",
    "FaceRecognizer",
    "FaceDatabase",
    "FaceVisualizer"
]
