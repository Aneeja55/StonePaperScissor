import cv2
import os
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

labels = ["rock","paper","scissors"]

for label in labels:
    os.makedirs(f"dataset/{label}", exist_ok=True)

img_count = 0
current_label = "rock"

offset = 20
imgSize = 224

while True:

    success, img = cap.read()
    img = cv2.flip(img,1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgResize = cv2.resize(imgCrop,(imgSize,imgSize))

        cv2.imshow("Hand Crop", imgResize)

    cv2.putText(img, f"Label: {current_label}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Dataset Collector", img)

    key = cv2.waitKey(1)

    if key == ord('r'):
        current_label = "rock"

    if key == ord('p'):
        current_label = "paper"

    if key == ord('s'):
        current_label = "scissors"

    if key == ord('c') and hands:
        img_path = f"dataset/{current_label}/{img_count}.jpg"
        cv2.imwrite(img_path, imgResize)
        img_count += 1
        print("Saved:", img_path)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()