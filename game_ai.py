import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

model = tf.keras.models.load_model("rps_model.h5")

labels = ["rock","paper","scissors"]

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

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

        imgInput = imgResize / 255.0
        imgInput = np.reshape(imgInput,(1,224,224,3))

        prediction = model.predict(imgInput)
        index = np.argmax(prediction)

        gesture = labels[index]

        cv2.putText(img,f"Gesture: {gesture}",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.imshow("Hand", imgResize)

    cv2.imshow("Game", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()