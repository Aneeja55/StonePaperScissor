import os
import tempfile

from cvzone.HandTrackingModule import HandDetector


def create_hand_detector(max_hands=1):
    working_directory = os.getcwd()
    model_directory = tempfile.gettempdir()
    if not model_directory.isascii():
        windows_temp = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Temp")
        if os.path.isdir(windows_temp) and windows_temp.isascii():
            model_directory = windows_temp
        else:
            raise RuntimeError(
                "Hand tracking needs an ASCII-only temporary directory on this system."
            )

    try:
        # MediaPipe's native model loader can fail when its working path contains Unicode.
        os.chdir(model_directory)
        return HandDetector(maxHands=max_hands)
    finally:
        os.chdir(working_directory)
