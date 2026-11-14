import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .detectors import FaceDetection


class FaceVisualizer:
    """
    Renders high-quality bounding boxes, identity badges, landmarks,
    and real-time HUD overlays for live video feeds and images.
    """

    COLOR_KNOWN = (72, 219, 116)      # Emerald Green (BGR)
    COLOR_UNKNOWN = (74, 85, 239)     # Crimson / Coral (BGR)
    COLOR_LANDMARK = (255, 220, 0)    # Cyan / Yellow glow (BGR)
    COLOR_HUD_BG = (22, 22, 24)       # Dark charcoal (BGR)
    COLOR_TEXT = (250, 250, 250)      # Crisp white (BGR)
    COLOR_MUTED = (160, 160, 160)     # Muted gray (BGR)
    COLOR_ACCENT = (245, 166, 35)     # Amber accent (BGR)

    def __init__(self):
        self._fps_history = []
        self._last_time = time.time()
        self._flash_counter = 0
        self._notification_text = ""
        self._notification_expiry = 0.0

    def update_fps(self) -> float:
        """Calculates smoothed frames per second."""
        current_time = time.time()
        delta = current_time - self._last_time
        self._last_time = current_time
        if delta > 0:
            fps = 1.0 / delta
            self._fps_history.append(fps)
            if len(self._fps_history) > 30:
                self._fps_history.pop(0)
            return float(np.mean(self._fps_history))
        return 0.0

    def trigger_flash(self, message: str = "Snapshot Saved!"):
        """Triggers a visual flash and notification message."""
        self._flash_counter = 4
        self._notification_text = message
        self._notification_expiry = time.time() + 2.5

    def draw_hud(
        self,
        frame: np.ndarray,
        detector_name: str,
        fps: float,
        num_faces: int,
        num_known: int,
        num_enrolled: int,
        threshold: float
    ):
        """Draws top information banner and bottom keyboard guide."""
        h, w = frame.shape[:2]

        # Top banner overlay
        header_height = 44
        top_overlay = frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, header_height), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(top_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, header_height), (w, header_height), (50, 50, 55), 1)

        # FPS indicator
        fps_text = f"FPS: {fps:4.1f}"
        cv2.putText(frame, fps_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        # Active Detector Badge
        det_text = f"Detector: {detector_name}"
        cv2.putText(frame, det_text, (130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEXT, 1, cv2.LINE_AA)

        # Faces & Enrolled Stats
        stats_text = f"Faces: {num_faces} ({num_known} known)  |  DB: {num_enrolled} enrolled  |  Thresh: {threshold:.2f}"
        cv2.putText(frame, stats_text, (w - 530, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Bottom guide overlay
        footer_height = 30
        bottom_overlay = frame.copy()
        cv2.rectangle(bottom_overlay, (0, h - footer_height), (w, h), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(bottom_overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, h - footer_height), (w, h - footer_height), (50, 50, 55), 1)

        guide_text = "[M] Switch Detector   [E] Enroll Face   [S] Snapshot   [+/-] Threshold   [Q] Quit"
        cv2.putText(
            frame, guide_text, (16, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_MUTED, 1, cv2.LINE_AA
        )

        # Flash effect when taking snapshots
        if self._flash_counter > 0:
            white_overlay = np.full_like(frame, 255)
            alpha = self._flash_counter / 8.0
            cv2.addWeighted(white_overlay, alpha, frame, 1.0 - alpha, 0, frame)
            self._flash_counter -= 1

        # Temporary notification banner
        if time.time() < self._notification_expiry and self._notification_text:
            text_size, _ = cv2.getTextSize(self._notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = (w - text_size[0]) // 2
            by = 80
            padding = 10
            cv2.rectangle(
                frame,
                (bx - padding, by - text_size[1] - padding),
                (bx + text_size[0] + padding, by + padding),
                (30, 30, 30),
                -1
            )
            cv2.rectangle(
                frame,
                (bx - padding, by - text_size[1] - padding),
                (bx + text_size[0] + padding, by + padding),
                self.COLOR_ACCENT,
                1
            )
            cv2.putText(
                frame, self._notification_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

    def draw_face(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        identity: str,
        similarity: float,
        is_known: bool,
        draw_landmarks: bool = True
    ):
        """
        Draws stylized biometric bounding box, identity badge,
        and landmark dots for a single detected face.
        """
        x, y, w, h = detection.bbox
        color = self.COLOR_KNOWN if is_known else self.COLOR_UNKNOWN

        # Draw subtle translucent box interior
        sub_overlay = frame.copy()
        cv2.rectangle(sub_overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(sub_overlay, 0.08, frame, 0.92, 0, frame)

        # Main bounding box border
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

        # Tech HUD Corner Brackets
        corner_len = max(12, int(min(w, h) * 0.18))
        thick = 3
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thick, cv2.LINE_AA)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thick, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thick, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thick, cv2.LINE_AA)

        # Draw facial landmarks if present
        if draw_landmarks and detection.landmarks:
            for lx, ly in detection.landmarks:
                ix, iy = int(round(lx)), int(round(ly))
                cv2.circle(frame, (ix, iy), 3, self.COLOR_LANDMARK, -1, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Identity Label Pill Badge
        if is_known:
            score_pct = max(0.0, min(100.0, (similarity + 1.0) / 2.0 * 100))  # Normalize cosine score to pct
            label_text = f"{identity} ({score_pct:.1f}%)"
        else:
            if similarity > 0:
                score_pct = max(0.0, min(100.0, (similarity + 1.0) / 2.0 * 100))
                label_text = f"Unknown ({score_pct:.1f}%)"
            else:
                label_text = "Unknown"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thick = 1
        text_size, baseline = cv2.getTextSize(label_text, font, font_scale, font_thick)

        # Pill position (placed right above the bounding box if fits, else inside top)
        badge_h = text_size[1] + 12
        badge_w = text_size[0] + 16

        badge_y1 = y - badge_h - 4
        badge_y2 = y - 4
        if badge_y1 < 48:  # avoid overlapping header
            badge_y1 = y + 4
            badge_y2 = y + 4 + badge_h

        badge_x1 = x
        badge_x2 = x + badge_w

        # Draw solid badge background
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Text in dark or white for maximum contrast
        text_color = (15, 15, 15) if is_known else (255, 255, 255)
        text_pos = (badge_x1 + 8, badge_y2 - 6)
        cv2.putText(frame, label_text, text_pos, font, font_scale, text_color, font_thick, cv2.LINE_AA)
