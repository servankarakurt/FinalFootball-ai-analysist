import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# --- AYARLAR ---
COMMON_METRICS = [
    "top_calma_p90", "uzaklastirma_p90", "ikili_mucadele_kazanma_yuzde", "hava_topu_kazanma_yuzde",
    "orta_sayisi_p90", "sut_p90", "dribbling_p90", "kilit_pas_p90", "ara_pas_p90", "uzun_pas_p90",
    "pas_isabet_yuzde", "xG_p90_common", "xA_p90_common", "Gls_p90", "Ast_p90", "G+A_p90",
    "kaleci_sut_karsilama_yuzde", "kaleci_kurtaris_sayisi_p90", "kaleci_pas_isabet_yuzde",
    "kaleci_sahipsiz_top_kazanma_p90", "kaleci_yenilen_gol_p90", "sahipsiz_top_kazanma_p90"
]

# --- ELİT OYUNCU EŞİKLERİ (Soft Cap) ---
CAPS = {
    "sut_p90": 4.5,
    "Gls_p90": 0.9,           
    "xG_p90_common": 0.8,
    "Ast_p90": 0.5,
    "xA_p90_common": 0.5,
    "dribbling_p90": 4.0,
    "kilit_pas_p90": 3.0,
    "ara_pas_p90": 0.5,
    "top_calma_p90": 4.5,
    "uzaklastirma_p90": 6.0,
    "sahipsiz_top_kazanma_p90": 8.0,
    "orta_sayisi_p90": 5.0,
    "uzun_pas_p90": 8.0,
    "hava_topu_kazanma_yuzde": 75.0, 
    "ikili_mucadele_kazanma_yuzde": 70.0,
    "pas_isabet_yuzde": 92.0,
    "kaleci_sut_karsilama_yuzde": 82.0,
    "kaleci_kurtaris_sayisi_p90": 4.5
}

def _ensure_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce") # fillna(0) BURADA YAPMIYORUZ, aşağıda akıllı yapacağız
    return df

def _make_common_cols_superlig(df):
    df = df.copy()
    def get_col(col_name):
        if col_name not in df.columns: return pd.Series(np.nan, index=df.index)
        val = df[col_name].astype(str).str.replace(',', '.', regex=False)
        return pd.to_numeric(val, errors="coerce") # NaN kalsın, 0 yapma

    # --- VERİ ÇEKME VE AKILLI DOLDURMA (IMPUTATION) ---
    
    # 1. Fiziksel Veriler (Yoksa %50 kabul et - Ortalama)
    df["hava_topu_kazanma_yuzde"] = get_col("hava_topu_kazanma_yuzde").fillna(50.0)
    df["ikili_mucadele_kazanma_yuzde"] = get_col("ikili_mucadele_kazanma_yuzde").fillna(50.0)
    
    # 2. Pas Verileri (Yoksa %75 kabul et)
    df["pas_isabet_yuzde"] = get_col("pas_isabet_yuzde").fillna(75.0)
    
    # 3. Defansif (Yoksa 0 olabilir, yapacak bir şey yok)
    df["top_calma_p90"] = get_col("top_calma_p90").fillna(0.0)
    df["uzaklastirma_p90"] = get_col("uzaklastirma_p90").fillna(0.0)
    df["sahipsiz_top_kazanma_p90"] = get_col("sahipsiz_top_kazanma_p90").fillna(0.0)
    
    # 4. Ofansif
    df["sut_p90"] = get_col("sut_p90").fillna(0.0)
    df["dribbling_p90"] = get_col("dribbling_p90").fillna(0.0)
    df["orta_sayisi_p90"] = get_col("orta_sayisi_p90").fillna(0.0)
    df["kilit_pas_p90"] = get_col("kilit_pas_p90").fillna(0.0)
    df["ara_pas_p90"] = get_col("ara_pas_p90").fillna(0.0)
    df["uzun_pas_p90"] = get_col("uzun_pas_p90").fillna(0.0)
    
    # 5. Gol ve Asist
    df["Gls_p90"] = get_col("Gls_p90").fillna(0.0)
    df["Ast_p90"] = get_col("Ast_p90").fillna(0.0)
    # Asist sütunu boşsa sentetikten dene
    if df["Ast_p90"].sum() == 0 and "asist_p90_sentetik" in df.columns:
         df["Ast_p90"] = get_col("asist_p90_sentetik").fillna(0.0)
    
    df["G+A_p90"] = get_col("G+A_p90").fillna(df["Gls_p90"] + df["Ast_p90"])

    # 6. xG ve xA (ÇOK ÖNEMLİ: Yoksa Gol'ü kullan)
    df["xG_p90_common"] = get_col("xG_p90_sentetik")
    # Eğer xG boşsa, Gol sayısını xG gibi kullan (Vekalet etsin)
    df["xG_p90_common"] = df["xG_p90_common"].fillna(df["Gls_p90"])
    
    df["xA_p90_common"] = get_col("xA_p90_sentetik")
    df["xA_p90_common"] = df["xA_p90_common"].fillna(df["Ast_p90"])

    # 7. Kaleci (Yoksa Ortalama Değerler)
    df["kaleci_sut_karsilama_yuzde"] = get_col("kaleci_sut_karsilama_yuzde").fillna(70.0)
    df["kaleci_kurtaris_sayisi_p90"] = get_col("kaleci_kurtaris_sayisi_p90").fillna(2.5)
    df["kaleci_yenilen_gol_p90"] = get_col("kaleci_yenilen_gol_p90").fillna(1.5)
    df["kaleci_pas_isabet_yuzde"] = get_col("kaleci_pas_isabet_yuzde").fillna(60.0)
    df["kaleci_sahipsiz_top_kazanma_p90"] = get_col("kaleci_sahipsiz_top_kazanma_p90").fillna(0.0)

    # Metadata
    if "Pos" not in df.columns and "FM_Mevki" in df.columns: df["Pos"] = df["FM_Mevki"]
    if "League" not in df.columns: df["League"] = "Süper Lig"
    if "Age" not in df.columns: df["Age"] = 0
    
    if "MevkiGroup" not in df.columns:
        def assign_mg(row):
            pos = str(row.get("Pos", ""))
            if "GK" in pos: return "GK"
            if "DF" in pos: return "FB" if row.get("orta_sayisi_p90", 0) > 1.0 else "CB"
            if "MF" in pos: return "AM" if (row.get("sut_p90", 0) > 1.5 or row.get("Gls_p90", 0) > 0.2) else "MID"
            if "FW" in pos: return "WING" if row.get("orta_sayisi_p90", 0) > 2.0 else "FWD"
            return "MID"
        df["MevkiGroup"] = df.apply(assign_mg, axis=1)

    return _ensure_numeric(df, COMMON_METRICS)

def _map_fbref_to_superlig(df):
    df = df.copy()
    if "Squad" in df.columns and "Team" not in df.columns: df["Team"] = df["Squad"]
    if "Comp" in df.columns and "League" not in df.columns: df["League"] = df["Comp"]
    if "Age" not in df.columns: df["Age"] = 0
    
    if "League" in df.columns:
        df["League"] = df["League"].astype(str).str.replace(r"^[a-z]{2}\s+", "", regex=True)
        df["League"] = df["League"].replace(["None", "nan", ""], np.nan)

    mapping_path = "data/external/team_league_mapping.csv"
    if os.path.exists(mapping_path):
        try:
            mapping_df = pd.read_csv(mapping_path)
            if "Team" in mapping_df.columns and "League" in mapping_df.columns:
                team_to_league = dict(zip(mapping_df["Team"], mapping_df["League"]))
                df["League"] = df["League"].fillna(df["Team"].map(team_to_league))
        except: pass
    df["League"] = df["League"].fillna("Unknown League")

    def assign_mg(row):
        pos = str(row.get("Pos", ""))
        if "GK" in pos: return "GK"
        if "DF" in pos: return "FB" if row.get("Crs", 0) > 1.0 else "CB"
        if "MF" in pos: return "AM" if (row.get("Sh/90", 0) > 1.5 or row.get("xG", 0) > 0.2) else "MID"
        if "FW" in pos: return "WING" if row.get("Crs", 0) > 2.0 else "FWD"
        return "MID"
    if "MevkiGroup" not in df.columns: df["MevkiGroup"] = df.apply(assign_mg, axis=1)

    def to_float(val): return pd.to_numeric(str(val).replace(',', '.'), errors='coerce')
    nineties = df["90s"].apply(to_float).fillna(1.0) if "90s" in df.columns else 1.0
    
    def get_per_90(c): return df[c].apply(to_float).fillna(0.0) / nineties if c in df.columns else 0.0
    def get_val(c, d=0.0): return df[c].apply(to_float).fillna(d) if c in df.columns else float(d)
    def get_val_p90(c): return get_val(c) / nineties

    df["top_calma_p90"] = get_per_90("Tkl") + get_per_90("Int")
    df["uzaklastirma_p90"] = get_per_90("Clr")
    df["ikili_mucadele_kazanma_yuzde"] = get_val("Won%", 50.0)
    df["hava_topu_kazanma_yuzde"] = get_val("Won%", 50.0)
    df["orta_sayisi_p90"] = get_per_90("Crs")
    df["sut_p90"] = get_val("Sh/90")
    df["dribbling_p90"] = get_per_90("Succ")
    df["kilit_pas_p90"] = get_per_90("KP")
    df["ara_pas_p90"] = get_per_90("TB")
    df["uzun_pas_p90"] = get_per_90("Cmp")
    df["pas_isabet_yuzde"] = get_val("Cmp%", 75.0)
    df["xG_p90_common"] = get_val_p90("xG")
    df["xA_p90_common"] = get_val_p90("xAG")
    df["Gls_p90"] = get_val_p90("Gls")
    df["Ast_p90"] = get_val_p90("Ast")
    df["G+A_p90"] = df["Gls_p90"] + df["Ast_p90"]
    df["kaleci_sut_karsilama_yuzde"] = get_val("Save%")
    df["kaleci_kurtaris_sayisi_p90"] = get_per_90("Saves")
    df["kaleci_yenilen_gol_p90"] = get_val("GA90")
    df["kaleci_pas_isabet_yuzde"] = 0.0 
    df["kaleci_sahipsiz_top_kazanma_p90"] = 0.0 
    df["sahipsiz_top_kazanma_p90"] = 0.0

    return df

def _score_from_norm(df_norm, df_out):
    # Skorlama Kategorileri
    w_def = {"top_calma_p90": 0.4, "uzaklastirma_p90": 0.4, "sahipsiz_top_kazanma_p90": 0.2}
    df_out["Defans_Skoru"] = sum(df_norm[k] * w for k, w in w_def.items() if k in df_norm.columns) * 10

    w_phy = {"ikili_mucadele_kazanma_yuzde": 0.5, "hava_topu_kazanma_yuzde": 0.5}
    df_out["Fizik_Skoru"] = sum(df_norm[k] * w for k, w in w_phy.items() if k in df_norm.columns) * 10

    w_pass = {"pas_isabet_yuzde": 0.6, "uzun_pas_p90": 0.4}
    df_out["Pas_Skoru"] = sum(df_norm[k] * w for k, w in w_pass.items() if k in df_norm.columns) * 10

    w_tech = {"orta_sayisi_p90": 0.20, "kilit_pas_p90": 0.20, "ara_pas_p90": 0.20, "xA_p90_common": 0.20, "Ast_p90": 0.20}
    df_out["Teknik_Skoru"] = sum(df_norm[k] * w for k, w in w_tech.items() if k in df_norm.columns) * 10

    w_att = {"Gls_p90": 0.35, "xG_p90_common": 0.25, "sut_p90": 0.20, "dribbling_p90": 0.20}
    df_out["Hucum_Skoru"] = sum(df_norm[k] * w for k, w in w_att.items() if k in df_norm.columns) * 10

    # Kaleci
    gk_score = 0.0
    if "kaleci_yenilen_gol_p90" in df_norm.columns:
        gk_score += (1.0 - df_norm["kaleci_yenilen_gol_p90"].fillna(1.0).to_numpy()) * 4.0
    if "kaleci_sut_karsilama_yuzde" in df_norm.columns:
        gk_score += df_norm["kaleci_sut_karsilama_yuzde"].fillna(0.0).to_numpy() * 4.0
    if "kaleci_kurtaris_sayisi_p90" in df_norm.columns:
        gk_score += df_norm["kaleci_kurtaris_sayisi_p90"].fillna(0.0).to_numpy() * 2.0
    
    df_out["Kaleci_Skoru"] = 0.0
    if "MevkiGroup" in df_out.columns: 
        df_out.loc[df_out["MevkiGroup"] == "GK", "Kaleci_Skoru"] = gk_score[df_out["MevkiGroup"] == "GK"]

    overall = []
    for i in range(len(df_out)):
        d = df_out.at[i, "Defans_Skoru"]
        ph = df_out.at[i, "Fizik_Skoru"]
        ps = df_out.at[i, "Pas_Skoru"]
        t = df_out.at[i, "Teknik_Skoru"]
        a = df_out.at[i, "Hucum_Skoru"]
        g = df_out.at[i, "Kaleci_Skoru"]
        
        mg = str(df_out.at[i, "MevkiGroup"]) if "MevkiGroup" in df_out.columns else "MID"
        
        if mg == "GK": o = g
        elif mg == "CB": o = 0.40*d + 0.40*ph + 0.20*ps
        elif mg == "FB": o = 0.30*t + 0.25*d + 0.25*ps + 0.20*a
        elif mg == "DM": o = 0.40*d + 0.30*ph + 0.30*ps
        elif mg == "MID": o = 0.25*ps + 0.25*t + 0.25*d + 0.25*ph
        elif mg == "AM": o = 0.50*t + 0.30*a + 0.20*ps
        elif mg == "WING": o = 0.50*a + 0.40*t + 0.10*ph
        elif mg == "FWD": o = 0.60*a + 0.30*ph + 0.10*t
        else: o = 0.2*d + 0.2*ph + 0.2*ps + 0.2*t + 0.2*a
            
        overall.append(o)
        
    df_out["Overall_Skoru"] = overall
    df_out["OyunKurucu_Skoru"] = (df_out["Pas_Skoru"] + df_out["Teknik_Skoru"]) / 2 # Görsel uyumluluk
    return df_out

def build_scored_and_benchmark(superlig_in, top5_in, superlig_out, top5_out, benchmark_out, scaler_out_prefix="models/combined_minmax"):
    print(f"Veriler İşleniyor...")
    try:
        sl = pd.read_csv(superlig_in)
        t5 = pd.read_csv(top5_in)
    except Exception as e:
        print(f"Hata: {e}")
        return

    # Filtreleme (Top 5)
    if "90s" in t5.columns:
        t5_90s = pd.to_numeric(t5["90s"].astype(str).str.replace(',', '.', regex=False), errors="coerce").fillna(0.0)
        t5 = t5[t5_90s >= 3.0].copy()
    
    sl2 = _make_common_cols_superlig(sl).copy()
    t52 = _map_fbref_to_superlig(t5).copy()
    t52 = _ensure_numeric(t52, COMMON_METRICS)
    
    sl2["_source"], t52["_source"] = "superlig", "top5"
    combined = pd.concat([sl2, t52], ignore_index=True)

    # Soft Cap Normalizasyonu
    combined_norm = combined.copy()
    for col in COMMON_METRICS:
        if col in CAPS:
            limit = CAPS[col]
            combined_norm[col] = combined_norm[col] / limit
            combined_norm[col] = combined_norm[col].clip(upper=1.0)
        else:
            max_val = combined_norm[col].quantile(0.99)
            if max_val == 0: max_val = 1.0
            combined_norm[col] = combined_norm[col] / max_val
            combined_norm[col] = combined_norm[col].clip(upper=1.0)

    combined_scored = _score_from_norm(combined_norm, combined.copy())
    sl_scored = combined_scored[combined_scored["_source"] == "superlig"].drop(columns=["_source"])
    t5_scored = combined_scored[combined_scored["_source"] == "top5"].drop(columns=["_source"])

    os.makedirs(os.path.dirname(superlig_out), exist_ok=True)
    sl_scored.to_csv(superlig_out, index=False, encoding="utf-8-sig")
    t5_scored.to_csv(top5_out, index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(benchmark_out, index=False)
    print(f"✅ İŞLEM TAMAMLANDI!")

if __name__ == "__main__":
    top5_input = None
    possible_files = [
        "top5_players_2025_2026_clean.csv",
        "data/external/top5_players_2025_2026_clean.csv",
        "data/external/players_data-2025_2026.csv",
        "data/processed/top5_players_scored.csv"
    ]
    for f in possible_files:
        if os.path.exists(f):
            top5_input = f
            break
            
    if top5_input:
        build_scored_and_benchmark(
            superlig_in="data/processed/super_lig_final_veri_role_enriched_CLEAN.csv",
            top5_in=top5_input,
            superlig_out="data/processed/superlig_scored_benchmark.csv",
            top5_out="data/processed/top5_players_scored.csv",
            benchmark_out="data/processed/top5_benchmarks_percentiles.csv",
            scaler_out_prefix="models/combined_minmax",
        )
    else:
        print("❌ Top 5 dosyası bulunamadı.")