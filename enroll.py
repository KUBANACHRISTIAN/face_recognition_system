import argparse
import os
import shutil
import sys
import cv2

from face_system.detectors import YuNetDetector
from face_system.recognizer import FaceRecognizer, FaceDatabase
from face_system.utils import ensure_models_exist, open_camera


def enroll_from_image(name: str, image_path: str, db: FaceDatabase) -> bool:
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return False

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Failed to load image: {image_path}")
        return False

    success = db.enroll(name, img, save_to_disk=True)
    if success:
        print(f"[Success] Enrolled '{name}' from {image_path}")
    else:
        print(f"[Failure] Could not detect a clear face in {image_path}")
    return success


def enroll_from_webcam(name: str, db: FaceDatabase, camera_idx: int = 0):
    cap, actual_idx = open_camera(camera_idx)
    if cap is None:
        print("[Error] Could not access webcam. Please check connections.")
        return False

    print(f"\n[Live Enrollment] Camera #{actual_idx} opened.")
    print("Instructions:")
    print("  - Look directly at the camera with clear lighting.")
    print("  - Press [SPACE] when the green bounding box appears to capture.")
    print("  - Press [ESC] or [Q] to cancel.\n")

    detector = db.detector
    window_name = f"Enroll: {name} (SPACE to Capture, ESC to Cancel)"
    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)

        if detections:
            best_det = max(detections, key=lambda d: d.bbox[2] * d.bbox[3])
            x, y, w, h = best_det.bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display, f"Face Detected! Press SPACE to capture {name}",
                (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                display, "No face detected. Position yourself in frame...",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC or Q
            print("[Canceled] Enrollment aborted by user.")
            break
        elif key == 32:  # SPACE
            if detections:
                success = db.enroll(name, frame, save_to_disk=True)
                if success:
                    print(f"[Success] Successfully enrolled '{name}'!")
                    captured = True
                    # Visual feedback flash
                    flash = display.copy()
                    cv2.putText(flash, f"ENROLLED: {name}!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.imshow(window_name, flash)
                    cv2.waitKey(1000)
                else:
                    print("[Error] Failed to process face features. Try again.")
                break
            else:
                print("[Warning] No face in frame. Look at camera and press SPACE.")

    cap.release()
    cv2.destroyAllWindows()
    return captured


def list_enrolled(db: FaceDatabase):
    print("\n--- Enrolled Identities Database ---")
    if not db.identities:
        print("Database is empty. No faces enrolled yet.")
        return

    for name, embs in sorted(db.identities.items()):
        print(f"  * {name:<20} ({len(embs)} face sample{'s' if len(embs) > 1 else ''})")
    print(f"Total: {len(db.identities)} people enrolled.\n")


def delete_person(name: str, db: FaceDatabase):
    person_dir = os.path.join(db.known_faces_dir, name)
    file_single = os.path.join(db.known_faces_dir, f"{name}.jpg")

    found = False
    if os.path.isdir(person_dir):
        shutil.rmtree(person_dir)
        found = True
    elif os.path.isfile(file_single):
        os.remove(file_single)
        found = True

    if found or name in db.identities:
        db.identities.pop(name, None)
        db._save_cache()
        print(f"[Success] Removed identity '{name}'.")
    else:
        print(f"[Warning] Identity '{name}' was not found in database.")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Enrollment CLI")
    parser.add_argument("--name", type=str, help="Name of the person to enroll")
    parser.add_argument("--image", type=str, help="Path to image file for enrollment")
    parser.add_argument("--webcam", action="store_true", help="Enroll interactively via live webcam")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--list", action="store_true", help="List all currently enrolled identities")
    parser.add_argument("--delete", type=str, help="Delete an enrolled identity")

    args = parser.parse_args()

    ensure_models_exist()
    db = FaceDatabase()

    if args.list:
        list_enrolled(db)
        return

    if args.delete:
        delete_person(args.delete, db)
        return

    if args.name:
        if args.webcam:
            enroll_from_webcam(args.name, db, args.camera)
        elif args.image:
            enroll_from_image(args.name, args.image, db)
        else:
            print("[Error] Please specify --image <path> or --webcam with --name.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
