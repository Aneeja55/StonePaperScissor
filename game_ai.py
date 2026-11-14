from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from hand_tracking import create_hand_detector

MODEL_PATH = Path(__file__).resolve().with_name("rps_model.h5")
LABELS = ["paper", "rock", "scissors"]
IMAGE_SIZE = 224
OFFSET = 20


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    detector = create_hand_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open the webcam. Check the camera connection and permissions.")

    try:
        while True:
            success, img = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img)

            if hands:
                x, y, w, h = hands[0]["bbox"]
                height, width = img.shape[:2]
                x1 = max(0, x - OFFSET)
                y1 = max(0, y - OFFSET)
                x2 = min(width, x + w + OFFSET)
                y2 = min(height, y + h + OFFSET)
                img_crop = img[y1:y2, x1:x2]

                if img_crop.size:
                    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                    img_resize = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
                    img_input = np.expand_dims(img_resize.astype(np.float32) / 255.0, axis=0)

                    prediction = model.predict(img_input, verbose=0)
                    gesture = LABELS[int(np.argmax(prediction))]

                    cv2.putText(img, f"Gesture: {gesture}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Hand", cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR))

            cv2.imshow("Game", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()