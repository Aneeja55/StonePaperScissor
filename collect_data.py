import cv2
import os

from hand_tracking import create_hand_detector

LABELS = ("rock", "paper", "scissors")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
IMAGE_SIZE = 224
OFFSET = 20


def main():
    for label in LABELS:
        os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

    detector = create_hand_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open the webcam. Check the camera connection and permissions.")

    img_count = 0
    current_label = "rock"

    try:
        while True:
            success, img = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img)
            img_resize = None

            if hands:
                x, y, w, h = hands[0]["bbox"]
                height, width = img.shape[:2]
                x1 = max(0, x - OFFSET)
                y1 = max(0, y - OFFSET)
                x2 = min(width, x + w + OFFSET)
                y2 = min(height, y + h + OFFSET)
                img_crop = img[y1:y2, x1:x2]
                if img_crop.size:
                    img_resize = cv2.resize(img_crop, (IMAGE_SIZE, IMAGE_SIZE))
                    cv2.imshow("Hand Crop", img_resize)

            cv2.putText(img, f"Label: {current_label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Dataset Collector", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                current_label = "rock"
            elif key == ord("p"):
                current_label = "paper"
            elif key == ord("s"):
                current_label = "scissors"
            elif key == ord("c") and img_resize is not None:
                img_path = os.path.join(DATASET_DIR, current_label, f"{img_count}.jpg")
                if not cv2.imwrite(img_path, img_resize):
                    raise OSError(f"Could not save image: {img_path}")
                img_count += 1
                print("Saved:", img_path)
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()