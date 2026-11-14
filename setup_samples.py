import os
import urllib.request
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_DIR = os.path.join(BASE_DIR, "known_faces")
TEST_DIR = os.path.join(BASE_DIR, "test_samples")

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Sample images available in official OpenCV repository
SAMPLES = {
    "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
    "messi5.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg"
}

def download_image(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    arr = np.asarray(bytearray(urllib.request.urlopen(req, timeout=15).read()), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def setup_samples():
    print("[Sample Setup] Downloading reference faces from OpenCV repository...")

    # 1. Download Lena
    lena_img = download_image(SAMPLES["lena.jpg"])
    if lena_img is not None:
        lena_dir = os.path.join(KNOWN_DIR, "Lena")
        os.makedirs(lena_dir, exist_ok=True)
        cv2.imwrite(os.path.join(lena_dir, "lena_ref.jpg"), lena_img)
        # Also save a test query image
        cv2.imwrite(os.path.join(TEST_DIR, "test_lena.jpg"), lena_img)
        print("  [+] Setup 'Lena' in known_faces/Lena/")

    # 2. Download Messi
    messi_img = download_image(SAMPLES["messi5.jpg"])
    if messi_img is not None:
        messi_dir = os.path.join(KNOWN_DIR, "Messi")
        os.makedirs(messi_dir, exist_ok=True)
        cv2.imwrite(os.path.join(messi_dir, "messi_ref.jpg"), messi_img)
        # Also save as test query image
        cv2.imwrite(os.path.join(TEST_DIR, "test_messi.jpg"), messi_img)
        print("  [+] Setup 'Messi' in known_faces/Messi/")

    # 3. Create a composite test image with both Lena and Messi side-by-side to test multi-face detection!
    if lena_img is not None and messi_img is not None:
        h = 400
        l_resized = cv2.resize(lena_img, (int(lena_img.shape[1] * h / lena_img.shape[0]), h))
        m_resized = cv2.resize(messi_img, (int(messi_img.shape[1] * h / messi_img.shape[0]), h))
        composite = np.hstack([l_resized, m_resized])
        composite_path = os.path.join(TEST_DIR, "test_dual_faces.jpg")
        cv2.imwrite(composite_path, composite)
        print(f"  [+] Created composite test image at {composite_path}")

    print("[Sample Setup] Done! Ready to run recognition tests.")

if __name__ == "__main__":
    setup_samples()
