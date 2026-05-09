import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")

MODEL_NAME = "manual_model"

IMG_W = 160
IMG_H = 120

CROP_TOP_PERCENT = 0.00
CROP_BOTTOM_PERCENT = 1.00

BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_SPLIT = 0.2

RANDOM_SEED = 42

TRAIN_STEERING_ONLY = True

MIN_ABS_THROTTLE = 0.02

# USE_ONLY_SOURCES = None
# Наприклад:
USE_ONLY_SOURCES = ["fpv"]
# USE_ONLY_SOURCES = ["opencv"]
# USE_ONLY_SOURCES = ["fpv", "opencv"]

DROP_ZERO_STEERING_PART = 0.60
# 0.60 означає: залишити тільки 60% кадрів, де steering майже 0.
# Це потрібно, щоб модель не навчилась тільки їхати прямо.

ZERO_STEERING_LIMIT = 0.05


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def preprocess_image_bgr(img_bgr):
    h, w, _ = img_bgr.shape

    y1 = int(h * CROP_TOP_PERCENT)
    y2 = int(h * CROP_BOTTOM_PERCENT)

    img_bgr = img_bgr[y1:y2, 0:w]

    img_bgr = cv2.resize(img_bgr, (IMG_W, IMG_H))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_rgb = img_rgb.astype(np.float32) / 255.0

    return img_rgb


def load_dataset_index():
    rows = []

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    session_dirs = sorted(DATASET_DIR.glob("session_*"))

    if not session_dirs:
        raise RuntimeError("No session_* folders found in dataset/")

    print("Found sessions:", len(session_dirs))

    for session_dir in session_dirs:
        csv_path = session_dir / "data.csv"

        if not csv_path.exists():
            print("Skip, no data.csv:", session_dir)
            continue

        df = pd.read_csv(csv_path)

        required = ["image", "steering", "throttle"]

        for col in required:
            if col not in df.columns:
                raise RuntimeError(f"Column '{col}' not found in {csv_path}")

        if "source" not in df.columns:
            df["source"] = session_dir.name

        if USE_ONLY_SOURCES is not None:
            df = df[df["source"].isin(USE_ONLY_SOURCES)]

        df["steering"] = pd.to_numeric(df["steering"], errors="coerce")
        df["throttle"] = pd.to_numeric(df["throttle"], errors="coerce")

        df = df.dropna(subset=["image", "steering", "throttle"])

        df = df[df["throttle"].abs() > MIN_ABS_THROTTLE]

        for _, row in df.iterrows():
            img_path = session_dir / str(row["image"])

            if not img_path.exists():
                continue

            rows.append({
                "image_path": str(img_path),
                "steering": float(row["steering"]),
                "throttle": float(row["throttle"]),
                "source": str(row["source"])
            })

    if not rows:
        raise RuntimeError("No valid training rows found.")

    df_all = pd.DataFrame(rows)

    print()
    print("Raw valid rows:", len(df_all))
    print("Sources:")
    print(df_all["source"].value_counts())
    print()
    print("Steering stats:")
    print(df_all["steering"].describe())
    print()
    print("Throttle stats:")
    print(df_all["throttle"].describe())

    return df_all


def balance_zero_steering(df):
    straight = df[df["steering"].abs() < ZERO_STEERING_LIMIT]
    turning = df[df["steering"].abs() >= ZERO_STEERING_LIMIT]

    if len(straight) == 0 or len(turning) == 0:
        return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    keep_n = int(len(straight) * DROP_ZERO_STEERING_PART)

    straight_keep = straight.sample(
        n=max(1, keep_n),
        random_state=RANDOM_SEED
    )

    df_balanced = pd.concat([straight_keep, turning], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print()
    print("After simple balancing:")
    print("Rows:", len(df_balanced))
    print("Straight kept:", len(straight_keep), "from", len(straight))
    print("Turning kept:", len(turning))
    print("Steering stats:")
    print(df_balanced["steering"].describe())

    return df_balanced


def load_image_and_label(image_path, steering, throttle):
    image_path = image_path.numpy().decode("utf-8")

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
    else:
        img = preprocess_image_bgr(img_bgr)

    if TRAIN_STEERING_ONLY:
        y = np.array([steering], dtype=np.float32)
    else:
        y = np.array([steering, throttle], dtype=np.float32)

    return img.astype(np.float32), y


def tf_load_image_and_label(image_path, steering, throttle):
    img, y = tf.py_function(
        func=load_image_and_label,
        inp=[image_path, steering, throttle],
        Tout=[tf.float32, tf.float32]
    )

    img.set_shape((IMG_H, IMG_W, 3))

    if TRAIN_STEERING_ONLY:
        y.set_shape((1,))
    else:
        y.set_shape((2,))

    return img, y


def make_tf_dataset(df, shuffle):
    image_paths = df["image_path"].values.astype(str)
    steering = df["steering"].values.astype(np.float32)
    throttle = df["throttle"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, steering, throttle))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    ds = ds.map(tf_load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def build_model():
    output_units = 1 if TRAIN_STEERING_ONLY else 2

    inputs = keras.Input(shape=(IMG_H, IMG_W, 3))

    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu")(inputs)
    x = layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(48, (5, 5), strides=(2, 2), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)

    x = layers.Flatten()(x)

    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dense(10, activation="relu")(x)

    outputs = layers.Dense(output_units, activation="tanh")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"]
    )

    return model


def plot_history(history):
    MODELS_DIR.mkdir(exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODELS_DIR / f"{MODEL_NAME}_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history.history["mae"], label="train_mae")
    plt.plot(history.history["val_mae"], label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig(MODELS_DIR / f"{MODEL_NAME}_mae.png", dpi=150)
    plt.close()


def save_preprocess_config():
    MODELS_DIR.mkdir(exist_ok=True)

    config = {
        "img_w": IMG_W,
        "img_h": IMG_H,
        "crop_top_percent": CROP_TOP_PERCENT,
        "crop_bottom_percent": CROP_BOTTOM_PERCENT,
        "train_steering_only": TRAIN_STEERING_ONLY,
        "input_color_after_cv2_imread": "BGR",
        "model_input_color": "RGB",
        "pixel_range": "0_to_1",
        "output": "steering" if TRAIN_STEERING_ONLY else "steering_throttle"
    }

    with open(MODELS_DIR / f"{MODEL_NAME}_preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = MODELS_DIR / f"{MODEL_NAME}.tflite"

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("Saved TFLite model:", tflite_path)


def quick_prediction_check(model, df):
    print()
    print("Quick prediction check:")

    sample = df.sample(n=min(10, len(df)), random_state=RANDOM_SEED)

    for _, row in sample.iterrows():
        img_bgr = cv2.imread(row["image_path"])

        if img_bgr is None:
            continue

        img = preprocess_image_bgr(img_bgr)
        x = np.expand_dims(img, axis=0)

        pred = model.predict(x, verbose=0)[0]

        if TRAIN_STEERING_ONLY:
            pred_steering = float(pred[0])
            print(
                "real steering:",
                round(float(row["steering"]), 3),
                "pred:",
                round(pred_steering, 3),
                "| source:",
                row["source"]
            )
        else:
            pred_steering = float(pred[0])
            pred_throttle = float(pred[1])
            print(
                "real steering:",
                round(float(row["steering"]), 3),
                "pred:",
                round(pred_steering, 3),
                "| real throttle:",
                round(float(row["throttle"]), 3),
                "pred:",
                round(pred_throttle, 3),
                "| source:",
                row["source"]
            )


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_dataset_index()
    df = balance_zero_steering(df)

    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED
    )

    print()
    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))

    train_ds = make_tf_dataset(train_df, shuffle=True)
    val_ds = make_tf_dataset(val_df, shuffle=False)

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{MODEL_NAME}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    keras_path = MODELS_DIR / f"{MODEL_NAME}.keras"
    model.save(keras_path)

    print()
    print("Saved Keras model:", keras_path)

    plot_history(history)
    save_preprocess_config()
    convert_to_tflite(model)
    quick_prediction_check(model, val_df)

    print()
    print("Done.")
    print("Files created:")
    print(MODELS_DIR / f"{MODEL_NAME}.keras")
    print(MODELS_DIR / f"{MODEL_NAME}.tflite")
    print(MODELS_DIR / f"{MODEL_NAME}_best.keras")
    print(MODELS_DIR / f"{MODEL_NAME}_loss.png")
    print(MODELS_DIR / f"{MODEL_NAME}_mae.png")
    print(MODELS_DIR / f"{MODEL_NAME}_preprocess_config.json")


if __name__ == "__main__":
    main()