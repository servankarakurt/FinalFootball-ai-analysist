import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import os

DATA_FILE = "data/processed/super_lig_final_veri_scored.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "futbol_ann_ga.h5")
SCALER_MEAN_FILE = os.path.join(MODEL_DIR, "scaler_mean.npy")
SCALER_SCALE_FILE = os.path.join(MODEL_DIR, "scaler_scale.npy")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.txt")


def load_data(path):
    df = pd.read_csv(path)

    # Hedef: G+A_p90
    y = df["G+A_p90"].fillna(0.0).values

    # Kimlik kolonlarını düş
    drop_cols = [
        "Player", "Team", "Pos",
        "FM_Mevki", "FM_Rol", "FM_Rol_Base", "FM_Rol_Gorev"
    ]
    df_num = df.drop(columns=drop_cols, errors="ignore")
    # Yine de sadece numerik kolonları al
    df_num = df_num.select_dtypes(include=[np.number])

    # Hedefi feature'lardan çıkar
    if "G+A_p90" in df_num.columns:
        df_num = df_num.drop(columns=["G+A_p90"])

    X = df_num.values
    feature_names = df_num.columns.tolist()

    return X, y, feature_names


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Veri yükleniyor:", DATA_FILE)
    X, y, feature_names = load_data(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])

    print("Model eğitiliyor...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test MSE: {loss:.4f}, MAE: {mae:.4f}")

    model.save(MODEL_FILE)
    np.save(SCALER_MEAN_FILE, scaler.mean_)
    np.save(SCALER_SCALE_FILE, scaler.scale_)
    with open(FEATURE_NAMES_FILE, "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    print("Model ve scaler kaydedildi:", MODEL_FILE)
    print("Feature sayısı:", len(feature_names))


if __name__ == "__main__":
    main()
