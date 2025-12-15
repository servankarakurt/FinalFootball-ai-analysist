import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# --- AYARLAR ---
DATA_FILE = "data/processed/super_lig_final_veri_scored.csv"
MODEL_DIR = "models"
PIPELINE_FILE = os.path.join(MODEL_DIR, "futbol_pipeline.pkl")

def load_and_clean_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Veri dosyası yok: {path}")
        
    df = pd.read_csv(path)
    
    # Hedef: G+A_p90 (Skor Katkısı)
    y = df["G+A_p90"].fillna(0.0)
    
    # LEAKAGE (Kopya) SÜTUNLARI TEMİZLE
    # Sonucu direkt ele verenleri çıkarıyoruz
    leak_cols = [
        "Gls_p90", "Ast_p90", "G+A_p90", 
        "asist_p90_sentetik", "xA_p90_sentetik", "xG_p90_sentetik", 
        "Gls", "Ast", "G+A",
        "Hucum_Skoru", "Defans_Skoru", "OyunKurucu_Skoru", "Overall_Skoru", "Kaleci_Skoru"
    ]
    # Kimlik bilgilerini çıkar
    id_cols = [
        "Player", "Team", "Pos", "MevkiGroup", 
        "FM_Mevki", "FM_Rol", "FM_Rol_Base", "FM_Rol_Gorev",
        "League", "Similarity", "Top5Percentile_Overall_Skoru", 
        "PosGroup", "TeamSheet", "Is_Attacker", "Is_Defender"
    ]
    
    # Sadece sayısal sütunları seç ve yasaklıları düş
    X = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in leak_cols + id_cols if c in X.columns]
    X = X.drop(columns=cols_to_drop)
    
    return X, y

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📦 Veri yükleniyor: {DATA_FILE}")
    
    X, y = load_and_clean_data(DATA_FILE)
    features = X.columns.tolist()
    
    print(f"✅ Özellik Sayısı: {len(features)}")
    print(f"📋 İlk 5 Özellik: {features[:5]}")
    
    # Eğitim / Test Ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- PIPELINE KURULUMU (TEK ÇATI ALTINDA) ---
    # 1. Scaler: Veriyi standartlaştırır
    # 2. Model: XGBoost Regressor (Hızlı ve güçlü)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
    ])
    
    print("🚀 Model eğitiliyor (XGBoost)...")
    pipeline.fit(X_train, y_train)
    
    # --- DEĞERLENDİRME ---
    print("\n📊 Model Performansı:")
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"   📈 R2 Score: {r2:.3f}")
    print(f"   📉 MAE (Ortalama Hata): {mae:.3f}")
    
    # --- KAYIT (EN ÖNEMLİ KISIM) ---
    # Pipeline nesnesine, hangi özellikleri kullandığımızı da "etiket" olarak yapıştırıyoruz.
    # Böylece app.py dosyası features.txt aramasına gerek kalmadan bunu bilecek.
    pipeline.required_features = features
    
    joblib.dump(pipeline, PIPELINE_FILE)
    print(f"\n💾 Pipeline başarıyla kaydedildi: {PIPELINE_FILE}")
    print("   (Bu dosya içinde hem Scaler, hem Model, hem de Feature Listesi var!)")

if __name__ == "__main__":
    main()