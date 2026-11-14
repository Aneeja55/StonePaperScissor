import random
import time

import cv2

from game_rules import countdown_remaining, gesture_from_fingers, round_winner
from hand_tracking import create_hand_detector


ROUND_DURATION = 2.0


def main():
    detector = create_hand_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open the webcam. Check the camera connection and permissions.")

    player_score = 0
    ai_score = 0
    round_started_at = None
    round_message = "Press S to start a round"

    try:
        while True:
            success, img = cap.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            img = cv2.flip(img, 1)
            hands, img = detector.findHands(img)

            if round_started_at is not None:
                remaining = countdown_remaining(
                    round_started_at, time.monotonic(), ROUND_DURATION
                )
                if remaining == 0:
                    round_started_at = None
                    if not hands:
                        round_message = "No hand detected. Press S to retry"
                    else:
                        player_move = gesture_from_fingers(detector.fingersUp(hands[0]))
                        if player_move is None:
                            round_message = "Gesture not recognized. Press S to retry"
                        else:
                            ai_move = random.choice(("rock", "paper", "scissors"))
                            winner = round_winner(player_move, ai_move)
                            if winner == "player":
                                player_score += 1
                                result = "You win"
                            elif winner == "ai":
                                ai_score += 1
                                result = "AI wins"
                            else:
                                result = "Tie"
                            round_message = f"You: {player_move} | AI: {ai_move} | {result}"
                else:
                    round_message = f"Show your move: {remaining:.1f}"

            cv2.putText(img, f"Player: {player_score}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"AI: {ai_score}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, round_message, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, "S: start/retry    Q: quit", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("AI Rock Paper Scissors", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                round_started_at = time.monotonic()
                round_message = "Show your move"
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
