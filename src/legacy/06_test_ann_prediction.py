import pandas as pd
import numpy as np
from tensorflow import keras

DATA_FILE = "data/processed/super_lig_final_veri_scored.csv"
MODEL_FILE = "models/futbol_ann_ga.h5"
SCALER_MEAN_FILE = "models/scaler_mean.npy"
SCALER_SCALE_FILE = "models/scaler_scale.npy"
FEATURE_NAMES_FILE = "models/feature_names.txt"


def load_model_and_scaler():
    # compile=False: eski HDF5 modeldeki 'mse' metric'ini deserialize etmeye çalışma
    model = keras.models.load_model(MODEL_FILE, compile=False)
    mean_ = np.load(SCALER_MEAN_FILE)
    scale_ = np.load(SCALER_SCALE_FILE)
    with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f.readlines()]
    return model, mean_, scale_, feature_names


def prepare_features(row: pd.Series, feature_names, mean_, scale_):
    # Eğitimde kullandığımız kolon isimlerini (feature_names) kullanarak sırayla değer çekiyoruz
    x = np.array([row.get(col, 0.0) for col in feature_names], dtype=float)
    x_scaled = (x - mean_) / scale_
    return x_scaled.reshape(1, -1)


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    model, mean_, scale_, feature_names = load_model_and_scaler()

    name = "Victor Osimhen"   # burada isimle oynayabilirsin
    rows = df[df["Player"].str.lower() == name.lower()]

    if rows.empty:
        print(f"{name} bulunamadı, CSV içindeki ismi kontrol et.")
    else:
        row = rows.iloc[0]

        x_scaled = prepare_features(row, feature_names, mean_, scale_)
        y_pred = model.predict(x_scaled, verbose=0)[0, 0]

        print(f"Oyuncu: {row['Player']} ({row['Team']})")
        print(f"Gerçek G+A_p90: {row['G+A_p90']:.3f}")
        print(f"Model tahmini G+A_p90: {y_pred:.3f}")
