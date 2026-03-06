import cv2
import numpy as np
import tensorflow as tf
import random

model = tf.keras.models.load_model("rps_model.h5")

labels = ["rock","paper","scissors"]

cap = cv2.VideoCapture(0)

player_score = 0
ai_score = 0

def predict(frame):

    img = cv2.resize(frame,(224,224))
    img = img/255.0
    img = np.reshape(img,(1,224,224,3))

    prediction = model.predict(img)

    index = np.argmax(prediction)

    return labels[index]

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    move = predict(frame)

    ai_move = random.choice(labels)

    cv2.putText(frame,f"Player: {move}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"AI: {ai_move}",(20,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("AI Rock Paper Scissors",frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()