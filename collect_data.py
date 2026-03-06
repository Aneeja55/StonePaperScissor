import cv2
import os

cap = cv2.VideoCapture(0)

labels = ["rock","paper","scissors"]

for label in labels:
    os.makedirs(f"dataset/{label}", exist_ok=True)

img_count = 0
current_label = "rock"

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    cv2.putText(frame, f"Label: {current_label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        current_label = "rock"

    if key == ord('p'):
        current_label = "paper"

    if key == ord('s'):
        current_label = "scissors"

    if key == ord('c'):
        img_path = f"dataset/{current_label}/{img_count}.jpg"
        cv2.imwrite(img_path, frame)
        img_count += 1
        print("Saved:", img_path)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()