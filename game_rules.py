from typing import Optional, Sequence


_FINGER_GESTURES = {
    (0, 0, 0, 0, 0): "rock",
    (1, 1, 1, 1, 1): "paper",
    (0, 1, 1, 0, 0): "scissors",
}

_BEATS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


def gesture_from_fingers(fingers: Sequence[int]) -> Optional[str]:
    return _FINGER_GESTURES.get(tuple(fingers))


def round_winner(player_move: str, ai_move: str) -> str:
    if player_move not in _BEATS or ai_move not in _BEATS:
        raise ValueError("Moves must be rock, paper, or scissors.")
    if player_move == ai_move:
        return "tie"
    if _BEATS[player_move] == ai_move:
        return "player"
    return "ai"


def countdown_remaining(started_at: float, current_time: float, duration: float) -> float:
    return max(0.0, duration - (current_time - started_at))
