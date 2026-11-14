from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
LABELS = ["paper", "rock", "scissors"]
IMAGE_SIZE = 224
BATCH_SIZE = 32


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    for label in LABELS:
        label_dir = DATASET_DIR / label
        image_count = sum(
            1 for path in label_dir.glob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if label_dir.is_dir() else 0
        if image_count < 2:
            raise ValueError(
                f"Need at least two images for '{label}' in {label_dir}; "
                "collect samples with collect_data.py first."
            )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_data = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        classes=LABELS,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_data = train_datagen.flow_from_directory(
        str(DATASET_DIR),
        classes=LABELS,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save(Path(__file__).resolve().with_name("rps_model.h5"))


if __name__ == "__main__":
    main()