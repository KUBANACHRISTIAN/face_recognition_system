import argparse
import datetime
import os
import sys
import time
import cv2
import numpy as np

from face_system.detectors import get_detector, YuNetDetector, HaarCascadeDetector
from face_system.recognizer import FaceRecognizer, FaceDatabase
from face_system.visualizer import FaceVisualizer
from face_system.utils import ensure_models_exist, open_camera


class FaceRecognitionApp:
    """
    Main application engine coordinating face detection,
    feature embedding extraction, database matching, and real-time visualization.
    """

    def __init__(
        self,
        detector_type: str = "dnn",
        threshold: float = 0.363,
        rebuild_db: bool = False
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        ensure_models_exist()

        print("[System] Initializing detectors...")
        self.yunet_model_path = os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")
        self.sface_model_path = os.path.join(base_dir, "models", "face_recognition_sface_2021dec.onnx")

        # Prepare both detectors so user can hot-swap at runtime
        self.dnn_detector = YuNetDetector(self.yunet_model_path)
        self.haar_detector = HaarCascadeDetector()

        self.current_detector_type = "dnn" if detector_type.lower() in ("dnn", "yunet") else "haar"
        self.active_detector = self.dnn_detector if self.current_detector_type == "dnn" else self.haar_detector

        print("[System] Initializing SFace recognizer & feature database...")
        self.recognizer = FaceRecognizer(self.sface_model_path)
        self.database = FaceDatabase(
            recognizer=self.recognizer,
            detector=self.dnn_detector
        )

        if rebuild_db:
            print("[System] Forcing database rebuild...")
            self.database.load(force_refresh=True)

        self.visualizer = FaceVisualizer()
        self.threshold = threshold

    def toggle_detector(self):
        """Switch between YuNet DNN and Haar Cascade on the fly."""
        if self.current_detector_type == "dnn":
            self.current_detector_type = "haar"
            self.active_detector = self.haar_detector
        else:
            self.current_detector_type = "dnn"
            self.active_detector = self.dnn_detector
        print(f"[Detector] Switched to {self.active_detector.name}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Runs detection, extracts embeddings, matches against database,
        and annotates the frame with bounding boxes and identities.
        Returns: (annotated_frame, total_faces_count, known_faces_count)
        """
        detections = self.active_detector.detect(frame)
        num_faces = len(detections)
        num_known = 0

        annotated = frame.copy()

        for det in detections:
            # 1. Extract 128-D normalized embedding vector
            feature = self.recognizer.extract_feature(frame, det)

            # 2. Match against enrolled identities
            if feature is not None:
                identity, similarity, is_known = self.database.identify(
                    feature=feature,
                    cosine_threshold=self.threshold,
                    metric="cosine"
                )
            else:
                identity, similarity, is_known = "Unknown", 0.0, False

            if is_known:
                num_known += 1

            # 3. Draw bounding box and identity badge
            self.visualizer.draw_face(
                frame=annotated,
                detection=det,
                identity=identity,
                similarity=similarity,
                is_known=is_known,
                draw_landmarks=(self.current_detector_type == "dnn")
            )

        return annotated, num_faces, num_known

    def run_image(self, image_path: str, output_path: Optional[str] = None, show_gui: bool = True):
        """Process a single still image."""
        if not os.path.exists(image_path):
            print(f"[Error] Image not found: {image_path}")
            return

        print(f"[Image Mode] Loading {image_path}...")
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] Failed to read {image_path}")
            return

        annotated, num_faces, num_known = self.process_frame(img)
        fps = 0.0

        self.visualizer.draw_hud(
            frame=annotated,
            detector_name=self.active_detector.name,
            fps=fps,
            num_faces=num_faces,
            num_known=num_known,
            num_enrolled=len(self.database.identities),
            threshold=self.threshold
        )

        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"result_{timestamp}.jpg")

        cv2.imwrite(output_path, annotated)
        print(f"[Result] Found {num_faces} face(s) ({num_known} recognized). Saved to: {output_path}")

        if show_gui:
            win_name = "Face Recognition (Press any key to close)"
            cv2.imshow(win_name, annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def run_video(self, video_path: str, output_path: Optional[str] = None, show_gui: bool = True):
        """Process a prerecorded video file."""
        if not os.path.exists(video_path):
            print(f"[Error] Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Could not open video file: {video_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

        print(f"[Video Mode] Processing {video_path} ({total_frames} frames @ {src_fps:.1f} FPS)...")
        frame_idx = 0
        win_name = "Face Recognition Video (Press Q to stop)"

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            annotated, num_faces, num_known = self.process_frame(frame)
            fps = self.visualizer.update_fps()

            self.visualizer.draw_hud(
                frame=annotated,
                detector_name=self.active_detector.name,
                fps=fps,
                num_faces=num_faces,
                num_known=num_known,
                num_enrolled=len(self.database.identities),
                threshold=self.threshold
            )

            if writer is not None:
                writer.write(annotated)

            if show_gui:
                cv2.imshow(win_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    print("[Video Mode] Aborted by user.")
                    break
                elif key == ord('m'):
                    self.toggle_detector()

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        cap.release()
        if writer is not None:
            writer.release()
            print(f"[Video Mode] Output saved to: {output_path}")
        cv2.destroyAllWindows()

    def run_live(self, camera_index: int = 0):
        """Run real-time face detection & recognition from a live camera stream."""
        cap, actual_idx = open_camera(camera_index)
        if cap is None:
            print(f"[Error] Could not open camera at index {camera_index}.")
            print("Tip: If you do not have a webcam attached, you can test images using:")
            print("     python main.py --mode image --source test_samples/sample.jpg")
            return

        print(f"[Live Feed] Camera #{actual_idx} stream active. Press 'q' to quit.")
        win_name = "Real-Time Face Detection & Recognition"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Live Feed] Failed to read frame from camera.")
                break

            annotated, num_faces, num_known = self.process_frame(frame)
            fps = self.visualizer.update_fps()

            self.visualizer.draw_hud(
                frame=annotated,
                detector_name=self.active_detector.name,
                fps=fps,
                num_faces=num_faces,
                num_known=num_known,
                num_enrolled=len(self.database.identities),
                threshold=self.threshold
            )

            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):  # Quit
                break
            elif key == ord('m'):  # Switch Detector
                self.toggle_detector()
            elif key == ord('s'):  # Snapshot
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(self.output_dir, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, annotated)
                self.visualizer.trigger_flash(f"Saved: {os.path.basename(snap_path)}")
                print(f"[Snapshot] Saved frame to {snap_path}")
            elif key in (ord('+'), ord('=')):  # Increase threshold
                self.threshold = min(0.95, self.threshold + 0.02)
                self.visualizer.trigger_flash(f"Threshold: {self.threshold:.2f}")
            elif key in (ord('-'), ord('_')):  # Decrease threshold
                self.threshold = max(0.10, self.threshold - 0.02)
                self.visualizer.trigger_flash(f"Threshold: {self.threshold:.2f}")
            elif key == ord('e'):  # Quick enroll current face
                self._handle_quick_enroll(frame)

        cap.release()
        cv2.destroyAllWindows()

    def _handle_quick_enroll(self, raw_frame: np.ndarray):
        """Pauses feed and prompts user for name to enroll current face."""
        detections = self.dnn_detector.detect(raw_frame)
        if not detections:
            self.visualizer.trigger_flash("No face detected to enroll!")
            return

        print("\n" + "=" * 50)
        print(" [QUICK ENROLL] Detected face in frame!")
        try:
            name = input(" Enter person's name (or press Enter to cancel): ").strip()
        except EOFError:
            name = ""

        if not name:
            print(" Enrollment canceled.")
            return

        success = self.database.enroll(name, raw_frame, save_to_disk=True)
        if success:
            msg = f"Enrolled '{name}' successfully!"
            print(f" [SUCCESS] {msg}")
            self.visualizer.trigger_flash(msg)
        else:
            print(" [FAILED] Could not process face features.")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Biometric Face Detection & Recognition System (Haar / DNN + SFace)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", type=str, choices=["live", "image", "video"], default="live",
        help="Operating mode:\n  live  - real-time webcam feed\n  image - process still photo\n  video - process prerecorded video"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Input source:\n  Camera index (e.g. 0) for live mode\n  Path to image or video file for file modes"
    )
    parser.add_argument(
        "--detector", type=str, choices=["dnn", "haar"], default="dnn",
        help="Detection algorithm: 'dnn' (YuNet, recommended) or 'haar' (Haar Cascade)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.363,
        help="Cosine similarity threshold for face match (default: 0.363)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path for processed image or video"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Headless execution without GUI display window"
    )
    parser.add_argument(
        "--rebuild-db", action="store_true",
        help="Force recomputation of face embeddings cache"
    )

    args = parser.parse_args()

    app = FaceRecognitionApp(
        detector_type=args.detector,
        threshold=args.threshold,
        rebuild_db=args.rebuild_db
    )

    if args.mode == "live":
        try:
            cam_idx = int(args.source)
        except ValueError:
            cam_idx = 0
        app.run_live(camera_index=cam_idx)
    elif args.mode == "image":
        app.run_image(image_path=args.source, output_path=args.output, show_gui=not args.no_gui)
    elif args.mode == "video":
        app.run_video(video_path=args.source, output_path=args.output, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
