# Stone Paper Scissors

A webcam Rock-Paper-Scissors game with hand tracking.

## Play the game

Use Python 3.12, then install the runtime dependencies:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Press **S** to start a round, hold up rock, paper, or scissors during the
two-second countdown, and press **Q** to quit. The game uses cvzone hand
landmarks and does not need TensorFlow or the included model file.

The webcam must be connected and accessible to Python. If the default camera
is not available, update the camera index in `main.py`.
The first start may download the hand-tracking model into your temporary
directory.

## Collect data and train the optional model

The `dataset` folders are initially empty, so training requires collecting
examples first. Install TensorFlow, run `collect_data.py`, and use **R**, **P**,
or **S** to select a label and **C** to save each hand crop. Collect samples
for all three gestures, then press **Q**. Run the following from the project
folder:

```powershell
python -m pip install tensorflow
python collect_data.py
python train_model.py
python game_ai.py
```

`game_ai.py` is a separate model-prediction demo; the main game currently uses
the hand-tracking finger patterns rather than the CNN.
