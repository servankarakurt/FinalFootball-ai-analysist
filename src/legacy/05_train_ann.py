import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

DATA_FILE = "data/processed/super_lig_final_veri_scored.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "futbol_ann_ga.h5")
SCALER_MEAN_FILE = os.path.join(MODEL_DIR, "scaler_mean.npy")
SCALER_SCALE_FILE = os.path.join(MODEL_DIR, "scaler_scale.npy")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.txt")

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
        
    df = pd.read_csv(path)

    y = df["G+A_p90"].fillna(0.0).values

    leak_cols = [
        "Gls_p90", "Ast_p90", "G+A_p90", 
        "asist_p90_sentetik", "xA_p90_sentetik", "xG_p90_sentetik", 
        "Gls", "Ast", "G+A",
        "Hucum_Skoru", "Defans_Skoru", "OyunKurucu_Skoru", "Overall_Skoru", "Kaleci_Skoru"
    ]
    
    id_cols = [
        "Player", "Team", "Pos", "MevkiGroup", 
        "FM_Mevki", "FM_Rol", "FM_Rol_Base", "FM_Rol_Gorev",
        "League", "Similarity", "Top5Percentile_Overall_Skoru",
        "PosGroup", "TeamSheet", "Is_Attacker", "Is_Defender"
    ]
    
    df_num = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in leak_cols + id_cols if c in df_num.columns]
    df_num = df_num.drop(columns=cols_to_drop, errors="ignore")

    X = df_num.values
    feature_names = df_num.columns.tolist()
    
    # --- KORELASYON KONTROLÜ ---
    print("\n🔍 Hedef (G+A) ile En Yüksek Korelasyonlu Özellikler:")
    corrs = []
    for i, col in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            corrs.append((col, corr))
    
    for name, val in sorted(corrs, key=lambda x: x[1], reverse=True)[:5]:
        print(f"   - {name}: {val:.3f}")
    print("-" * 40)

    return X, y, feature_names

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        
        keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
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

    print(f"Veri yükleniyor: {DATA_FILE}")
    try:
        X, y, feature_names = load_data(DATA_FILE)
    except Exception as e:
        print(f"HATA: {e}")
        return

    print(f"✅ Feature Sayısı: {len(feature_names)}")

    # Eğitim / Test ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sample_weights = 1.0 + (y_train * 5.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(X_train_scaled.shape[1])

    callbacks = [
        ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min', verbose=0),
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001, verbose=1)
    ]

    print("🚀 Model eğitiliyor... (Ağırlıklı Eğitim Devrede)")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        sample_weight=sample_weights, # <--- BURASI KRİTİK
        verbose=1
    )

    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest MSE: {loss:.4f}, MAE: {mae:.4f}")

    model.save(MODEL_FILE)
    np.save(SCALER_MEAN_FILE, scaler.mean_)
    np.save(SCALER_SCALE_FILE, scaler.scale_)
    with open(FEATURE_NAMES_FILE, "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    # --- PERFORMANS RAPORU ---
    print("\n" + "="*40)
    print("📊 MODEL PERFORMANS ANALİZİ (V3 - Weighted)")
    print("="*40)

    y_pred = model.predict(X_test_scaled, verbose=0)
    
    r2 = r2_score(y_test, y_pred)
    print(f"📈 R2 Score: {r2:.3f}")
    
    THRESHOLD = 0.30 # Eşiği biraz daha gerçekçi yaptım
    
    y_test_class = (y_test >= THRESHOLD).astype(int)
    y_pred_class = (y_pred >= THRESHOLD).astype(int)
    
    print(f"\n🏆 Sınıflandırma Başarısı (Eşik: {THRESHOLD} G+A/90)")
    print("-" * 40)
    print(classification_report(y_test_class, y_pred_class, target_names=["Düşük Katkı", "Yüksek Katkı"], zero_division=0))
    
    cm = confusion_matrix(y_test_class, y_pred_class)
    try:
        print(f"✅ Doğru Tespit Edilen Yıldızlar (TP): {cm[1,1]}")
        print(f"❌ Gözden Kaçan Yıldızlar (FN): {cm[1,0]}")
        print(f"⚠️ Yıldız Sanılan Balonlar (FP): {cm[0,1]}")
    except: pass
    print("="*40 + "\n")

if __name__ == "__main__":
    main()