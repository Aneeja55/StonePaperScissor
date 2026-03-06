import cv2
import random
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

playerScore = 0
aiScore = 0

timer = 0
stateResult = False
startGame = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    cv2.putText(img, f'Player: {playerScore}', (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(img, f'AI: {aiScore}', (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if startGame:

        if stateResult is False:
            timer = time.time()

            if time.time() - timer > 2:

                stateResult = True

                if hands:

                    hand = hands[0]
                    fingers = detector.fingersUp(hand)

                    playerMove = None

                    if fingers == [0,0,0,0,0]:
                        playerMove = "rock"

                    if fingers == [1,1,1,1,1]:
                        playerMove = "paper"

                    if fingers == [0,1,1,0,0]:
                        playerMove = "scissors"

                    aiMove = random.choice(["rock","paper","scissors"])

                    print("Player:", playerMove)
                    print("AI:", aiMove)

                    if playerMove == "rock" and aiMove == "scissors":
                        playerScore += 1
                    elif playerMove == "paper" and aiMove == "rock":
                        playerScore += 1
                    elif playerMove == "scissors" and aiMove == "paper":
                        playerScore += 1
                    elif playerMove != aiMove:
                        aiScore += 1

    cv2.imshow("AI Rock Paper Scissors", img)

    key = cv2.waitKey(1)

    if key == ord('s'):
        startGame = True
        stateResult = False

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()