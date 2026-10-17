import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"
MODEL_DIR = BASE_DIR / "model"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42


def main():
    if not TRAIN_DIR.exists() or not VALIDATION_DIR.exists():
        raise SystemExit(
            "Dataset folders are missing. See README.md for the required dataset structure."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("Add images to at least two class folders before training.")
    if val_ds.class_names != class_names:
        raise SystemExit(
            "Training and validation folders must contain the same class subfolder names."
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
    ])

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.30),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_DIR / "freshness_model.keras")
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {accuracy:.3f}")
    print("Saved model to model/freshness_model.keras")
    print("Saved class labels to model/class_names.json")


if __name__ == "__main__":
    main()
