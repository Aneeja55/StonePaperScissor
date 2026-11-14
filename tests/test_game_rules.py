import unittest

from game_rules import countdown_remaining, gesture_from_fingers, round_winner


class GameRulesTests(unittest.TestCase):
    def test_recognizes_supported_finger_patterns(self):
        self.assertEqual(gesture_from_fingers([0, 0, 0, 0, 0]), "rock")
        self.assertEqual(gesture_from_fingers([1, 1, 1, 1, 1]), "paper")
        self.assertEqual(gesture_from_fingers([0, 1, 1, 0, 0]), "scissors")

    def test_unrecognized_finger_pattern_returns_none(self):
        self.assertIsNone(gesture_from_fingers([1, 0, 1, 0, 0]))

    def test_resolves_all_round_outcomes(self):
        self.assertEqual(round_winner("rock", "scissors"), "player")
        self.assertEqual(round_winner("paper", "scissors"), "ai")
        self.assertEqual(round_winner("rock", "rock"), "tie")

    def test_rejects_unknown_moves(self):
        with self.assertRaises(ValueError):
            round_winner("lizard", "rock")

    def test_countdown_uses_elapsed_time_and_stops_at_zero(self):
        self.assertEqual(countdown_remaining(10.0, 11.5, 2.0), 0.5)
        self.assertEqual(countdown_remaining(10.0, 12.5, 2.0), 0.0)


if __name__ == "__main__":
    unittest.main()
