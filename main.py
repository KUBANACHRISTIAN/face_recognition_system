import argparse
import base64
import datetime
import io
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from face_system.detectors import get_detector, YuNetDetector, HaarCascadeDetector
from face_system.recognizer import FaceRecognizer, FaceDatabase
from face_system.visualizer import FaceVisualizer
from face_system.utils import ensure_models_exist, open_camera, get_default_models_dir


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
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError:
            pass

        ensure_models_exist()

        models_dir = get_default_models_dir()
        self.yunet_model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        self.sface_model_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

        # Prepare detectors
        self.dnn_detector = YuNetDetector(self.yunet_model_path)
        try:
            self.haar_detector = HaarCascadeDetector()
        except Exception as e:
            print(f"[Detector] Haar Cascade notice: {e}")
            self.haar_detector = self.dnn_detector

        self.current_detector_type = "dnn" if detector_type.lower() in ("dnn", "yunet") else "haar"
        self.active_detector = self.dnn_detector if self.current_detector_type == "dnn" else self.haar_detector

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
        """Runs detection, matching, and visualizer annotation."""
        detections = self.active_detector.detect(frame)
        num_faces = len(detections)
        num_known = 0

        annotated = frame.copy()

        for det in detections:
            feature = self.recognizer.extract_feature(frame, det)

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
        cap, actual_idx = open_camera(camera_index)
        if cap is None:
            print(f"[Error] Could not open camera at index {camera_index}.")
            print("Tip: If you do not have a webcam attached, you can test images using:")
            print("     python main.py --mode image --source test_samples/test_dual_faces.jpg")
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

            if key in (27, ord('q')):
                break
            elif key == ord('m'):
                self.toggle_detector()
            elif key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(self.output_dir, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, annotated)
                self.visualizer.trigger_flash(f"Saved: {os.path.basename(snap_path)}")
                print(f"[Snapshot] Saved frame to {snap_path}")
            elif key in (ord('+'), ord('=')):
                self.threshold = min(0.95, self.threshold + 0.02)
                self.visualizer.trigger_flash(f"Threshold: {self.threshold:.2f}")
            elif key in (ord('-'), ord('_')):
                self.threshold = max(0.10, self.threshold - 0.02)
                self.visualizer.trigger_flash(f"Threshold: {self.threshold:.2f}")
            elif key == ord('e'):
                self._handle_quick_enroll(frame)

        cap.release()
        cv2.destroyAllWindows()

    def _handle_quick_enroll(self, raw_frame: np.ndarray):
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


# -------------------------------------------------------------
# Web Application Export (FastAPI) for Vercel / Cloud / Browser
# -------------------------------------------------------------
app = FastAPI(
    title="Biometric Face Detection & Recognition API",
    description="Real-time Face Detection and Recognition using YuNet & SFace",
    version="1.0.0"
)


@app.middleware("http")
async def fix_vercel_path_middleware(request, call_next):
    """Normalizes request path if Vercel serverless rewrites prepend filename."""
    raw_path = request.scope.get("path", "")
    for prefix in ["/main.py", "/api/index.py", "/index.py"]:
        if raw_path == prefix:
            request.scope["path"] = "/"
            break
        elif raw_path.startswith(prefix + "/"):
            request.scope["path"] = raw_path[len(prefix):]
            break
    return await call_next(request)


# Lazy singleton engine instance for web requests
_engine: Optional[FaceRecognitionApp] = None


def get_engine() -> FaceRecognitionApp:
    global _engine
    if _engine is None:
        _engine = FaceRecognitionApp(detector_type="dnn")
    return _engine


class FramePayload(BaseModel):
    image: str  # Base64 encoded JPEG data URL


class EnrollFramePayload(BaseModel):
    name: str
    image: str


def decode_base64_image(data_url: str) -> Optional[np.ndarray]:
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Web] Base64 decode error: {e}")
        return None


@app.get("/", response_class=HTMLResponse)
@app.get("/main.py", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
async def serve_index():
    """Serves the interactive web webcam & image analysis dashboard."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Face Recognition API is running.</h1><p>Visit /docs for OpenAPI specifications.</p>")


@app.get("/health")
@app.get("/api/health")
@app.get("/main.py/health")
@app.get("/api/index.py/health")
async def health_check():
    engine = get_engine()
    return {
        "status": "healthy",
        "detector": engine.active_detector.name,
        "enrolled_count": len(engine.database.identities)
    }


@app.get("/api/identities")
async def get_identities():
    engine = get_engine()
    return {
        "identities": list(engine.database.identities.keys()),
        "total": len(engine.database.identities)
    }


@app.post("/api/detect_frame")
async def detect_frame(payload: FramePayload):
    """Processes a webcam frame from the browser and returns face bounding boxes and identities."""
    frame = decode_base64_image(payload.image)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image payload"})

    engine = get_engine()
    detections = engine.dnn_detector.detect(frame)
    faces_data = []

    for det in detections:
        feat = engine.recognizer.extract_feature(frame, det)
        if feat is not None:
            identity, similarity, is_known = engine.database.identify(
                feature=feat,
                cosine_threshold=engine.threshold,
                metric="cosine"
            )
        else:
            identity, similarity, is_known = "Unknown", 0.0, False

        faces_data.append({
            "bbox": list(det.bbox),
            "name": identity,
            "similarity": float(similarity),
            "is_known": bool(is_known),
            "landmarks": [list(pt) for pt in det.landmarks] if det.landmarks else []
        })

    return {
        "faces": faces_data,
        "count": len(faces_data),
        "enrolled_count": len(engine.database.identities)
    }


@app.post("/api/annotate_image")
async def annotate_uploaded_image(file: UploadFile = File(...)):
    """Receives an uploaded photo, annotates it with bounding boxes and badges, and returns the image."""
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

    engine = get_engine()
    annotated, num_faces, num_known = engine.process_frame(img)
    _, buffer = cv2.imencode(".jpg", annotated)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/api/enroll")
async def enroll_uploaded_image(name: str = Form(...), file: UploadFile = File(...)):
    """Enrolls a new person from an uploaded photo."""
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    engine = get_engine()
    success = engine.database.enroll(name.strip(), img, save_to_disk=True)
    if success:
        return {"status": "success", "message": f"Successfully enrolled '{name}' into database!"}
    return JSONResponse(status_code=400, content={"error": "No face detected in photo. Please use a clearer photo."})


@app.post("/api/enroll_frame")
async def enroll_webcam_frame(payload: EnrollFramePayload):
    """Enrolls a face captured directly from the browser webcam."""
    img = decode_base64_image(payload.image)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image payload"})

    engine = get_engine()
    success = engine.database.enroll(payload.name.strip(), img, save_to_disk=True)
    if success:
        return {"status": "success", "message": f"Successfully enrolled '{payload.name}'!"}
    return JSONResponse(status_code=400, content={"error": "No face detected to enroll."})


# -------------------------------------------------------------
# Desktop CLI Entry Point
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Biometric Face Detection & Recognition System (Haar / DNN + SFace)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode", type=str, choices=["live", "image", "video", "web"], default="live",
        help="Operating mode:\n  live  - real-time desktop webcam feed\n  image - process still photo\n  video - process prerecorded video\n  web   - launch local web server"
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
        "--port", type=int, default=8000,
        help="Port for local web server mode"
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

    if args.mode == "web":
        import uvicorn
        print(f"[Web Server] Starting local web dashboard on http://localhost:{args.port} ...")
        uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)
        return

    cli_app = FaceRecognitionApp(
        detector_type=args.detector,
        threshold=args.threshold,
        rebuild_db=args.rebuild_db
    )

    if args.mode == "live":
        try:
            cam_idx = int(args.source)
        except ValueError:
            cam_idx = 0
        cli_app.run_live(camera_index=cam_idx)
    elif args.mode == "image":
        cli_app.run_image(image_path=args.source, output_path=args.output, show_gui=not args.no_gui)
    elif args.mode == "video":
        cli_app.run_video(video_path=args.source, output_path=args.output, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
