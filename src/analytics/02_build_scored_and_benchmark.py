import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

COMMON_METRICS = [
    "top_calma_p90",
    "uzaklastirma_p90",
    "ikili_mucadele_kazanma_yuzde",
    "hava_topu_kazanma_yuzde",
    "orta_sayisi_p90",
    "sut_p90",
    "dribbling_p90",
    "kilit_pas_p90",
    "ara_pas_p90",
    "uzun_pas_p90",
    "pas_isabet_yuzde",
    "xG_p90_common",
    "xA_p90_common",
    "Gls_p90",
    "Ast_p90",
    "G+A_p90",
    "kaleci_sut_karsilama_yuzde",
    "kaleci_kurtaris_sayisi_p90",
    "kaleci_pas_isabet_yuzde",
    "kaleci_sahipsiz_top_kazanma_p90",
    "kaleci_yenilen_gol_p90",
]

def _compute_percentile(dist: pd.Series, v: float) -> float:
    if len(dist) == 0: return 50.0
    return float((dist <= v).mean() * 100.0)

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _make_common_cols_superlig(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xG_p90_common"] = pd.to_numeric(df.get("xG_p90_sentetik", 0.0), errors="coerce").fillna(0.0)
    df["xA_p90_common"] = pd.to_numeric(df.get("xA_p90_sentetik", 0.0), errors="coerce").fillna(0.0)
    
    if "Pos" not in df.columns and "FM_Mevki" in df.columns:
        df["Pos"] = df["FM_Mevki"]

    if "League" not in df.columns:
        df["League"] = "Süper Lig"
        
    return _ensure_numeric(df, COMMON_METRICS)

def _map_fbref_to_superlig(df: pd.DataFrame) -> pd.DataFrame:
    """
    FBref (Top 5) İngilizce sütunlarını bizim Türkçe metriklerimize dönüştürür.
    Ayrıca Metadata (Team, League, Age) eşleşmelerini yapar.
    """
    df = df.copy()
    
    # --- METADATA DÜZELTMELERİ (Kritik Kısım) ---
  
    if "Squad" in df.columns and "Team" not in df.columns:
        df["Team"] = df["Squad"] 
        
    if "Comp" in df.columns and "League" not in df.columns:
        df["League"] = df["Comp"] 
        
    if "Age" not in df.columns:
        df["Age"] = 0 
        

    if "League" in df.columns:
        df["League"] = df["League"].astype(str).str.replace(r"^[a-z]{2}\s+", "", regex=True)

    def assign_mevki_group(row):
        pos = str(row.get("Pos", ""))
        
        if "GK" in pos: return "GK"
        if "DF" in pos:
            if row.get("Crs", 0) > 1.0: return "FB" 
            return "CB"
        if "MF" in pos:
            if row.get("Sh/90", 0) > 1.5 or row.get("xG", 0) > 0.2: return "AM"
            return "MID"
        if "FW" in pos:
            if row.get("Crs", 0) > 2.0: return "WING"
            return "FWD"
        return "MID"

    if "MevkiGroup" not in df.columns:
        df["MevkiGroup"] = df.apply(assign_mevki_group, axis=1)
    
    if "90s" in df.columns:
        nineties = pd.to_numeric(df["90s"], errors="coerce").fillna(1.0)
    else:
        nineties = 1.0 
    
    def get_per_90(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(0.0) / nineties
        return 0.0

    def get_val(col_name, default_val=0.0):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(default_val)
        return float(default_val)

    def get_val_p90(col_name):
        val = get_val(col_name, 0.0)
        return val / nineties

    df["top_calma_p90"] = get_per_90("Tkl") + get_per_90("Int")
    df["uzaklastirma_p90"] = get_per_90("Clr")
    df["ikili_mucadele_kazanma_yuzde"] = get_val("Won%", 50.0)
    df["hava_topu_kazanma_yuzde"] = get_val("Won%", 50.0)
    
    df["orta_sayisi_p90"] = get_per_90("Crs")
    df["sut_p90"] = get_val("Sh/90", 0.0)
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

    df["kaleci_sut_karsilama_yuzde"] = get_val("Save%", 0.0)
    df["kaleci_kurtaris_sayisi_p90"] = get_per_90("Saves")
    df["kaleci_yenilen_gol_p90"] = get_val("GA90", 0.0)
    
    return df

def _make_common_cols_top5(df: pd.DataFrame) -> pd.DataFrame:
    df = _map_fbref_to_superlig(df)
    return _ensure_numeric(df, COMMON_METRICS)

def _score_from_norm(df_norm: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    # Skorlama Fonksiyonu--Önemli Kısım-- buraya dikkat edelim
    w_def = {"top_calma_p90": 0.25, "uzaklastirma_p90": 0.20, "ikili_mucadele_kazanma_yuzde": 0.30, "hava_topu_kazanma_yuzde": 0.25}
    df_out["Defans_Skoru"] = 0.0
    for k, w in w_def.items():
        if k in df_norm.columns: df_out["Defans_Skoru"] += df_norm[k] * w
    df_out["Defans_Skoru"] *= 10

    w_play = {"kilit_pas_p90": 0.30, "ara_pas_p90": 0.15, "uzun_pas_p90": 0.15, "xA_p90_common": 0.20, "pas_isabet_yuzde": 0.15, "dribbling_p90": 0.05}
    df_out["OyunKurucu_Skoru"] = 0.0
    for k, w in w_play.items():
        if k in df_norm.columns: df_out["OyunKurucu_Skoru"] += df_norm[k] * w
    df_out["OyunKurucu_Skoru"] *= 10

    w_att = {"sut_p90": 0.20, "xG_p90_common": 0.25, "Gls_p90": 0.20, "Ast_p90": 0.10, "xA_p90_common": 0.10, "dribbling_p90": 0.10, "orta_sayisi_p90": 0.05}
    df_out["Hucum_Skoru"] = 0.0
    for k, w in w_att.items():
        if k in df_norm.columns: df_out["Hucum_Skoru"] += df_norm[k] * w
    df_out["Hucum_Skoru"] *= 10

    w_gk = {"kaleci_sut_karsilama_yuzde": 0.35, "kaleci_kurtaris_sayisi_p90": 0.25, "kaleci_pas_isabet_yuzde": 0.20, "kaleci_sahipsiz_top_kazanma_p90": 0.20}
    gk_score = np.zeros(len(df_out))
    for k, w in w_gk.items():
        if k in df_norm.columns: gk_score += df_norm[k].to_numpy() * w
    
    if "kaleci_yenilen_gol_p90" in df_norm.columns:
         gk_score += (1 - df_norm["kaleci_yenilen_gol_p90"].to_numpy()) * 0.25
            
    df_out["Kaleci_Skoru"] = 0.0
    if "MevkiGroup" in df_out.columns:
        is_gk = df_out["MevkiGroup"].astype(str).eq("GK")
        df_out.loc[is_gk, "Kaleci_Skoru"] = gk_score[is_gk] * 10

    # OVERALL
    overall = []
    for i in range(len(df_out)):
        d, p, a, g = df_out.at[i, "Defans_Skoru"], df_out.at[i, "OyunKurucu_Skoru"], df_out.at[i, "Hucum_Skoru"], df_out.at[i, "Kaleci_Skoru"]
        mg = str(df_out.at[i, "MevkiGroup"]) if "MevkiGroup" in df_out.columns else "MID"

        if mg == "GK": o = g
        elif mg == "CB": o = 0.7 * d + 0.2 * p + 0.1 * a
        elif mg == "FB": o = 0.5 * d + 0.3 * p + 0.2 * a
        elif mg in ["DM", "MID"]: o = 0.3 * d + 0.5 * p + 0.2 * a
        elif mg == "AM": o = 0.2 * d + 0.5 * p + 0.3 * a
        elif mg == "WING": o = 0.1 * d + 0.3 * p + 0.6 * a
        elif mg == "FWD": o = 0.05 * d + 0.15 * p + 0.80 * a
        else: o = 0.33 * d + 0.33 * p + 0.34 * a
        overall.append(o)

    df_out["Overall_Skoru"] = overall
    return df_out

def build_scored_and_benchmark(
    superlig_in: str,
    top5_in: str,
    superlig_out: str,
    top5_out: str,
    benchmark_out: str,
    scaler_out_prefix: str = "models/combined_minmax"
):
    print("Veri işleme başlıyor...")
    try:
        sl = pd.read_csv(superlig_in)
        t5 = pd.read_csv(top5_in)
        print(f"Süper Lig: {len(sl)} satır, Top 5: {len(t5)} satır yüklendi.")
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return

    sl2 = _make_common_cols_superlig(sl).copy()
    t52 = _make_common_cols_top5(t5).copy()

    sl2["_source"] = "superlig"
    t52["_source"] = "top5"
    
    combined = pd.concat([sl2, t52], ignore_index=True)

    scaler = MinMaxScaler()
    combined_norm = combined.copy()
    combined_norm[COMMON_METRICS] = combined_norm[COMMON_METRICS].replace([np.inf, -np.inf], 0).fillna(0)
    combined_norm[COMMON_METRICS] = scaler.fit_transform(combined_norm[COMMON_METRICS])

    combined_scored = _score_from_norm(combined_norm, combined.copy())

    sl_scored = combined_scored[combined_scored["_source"] == "superlig"].drop(columns=["_source"])
    t5_scored = combined_scored[combined_scored["_source"] == "top5"].drop(columns=["_source"])

    bench_rows = []
    if "MevkiGroup" in t5_scored.columns:
        for mg in sorted(t5_scored["MevkiGroup"].astype(str).unique()):
            sub = t5_scored[t5_scored["MevkiGroup"].astype(str) == mg]
            if len(sub) < 5: continue
            for metric in ["Defans_Skoru", "OyunKurucu_Skoru", "Hucum_Skoru", "Overall_Skoru", "G+A_p90"]:
                s = pd.to_numeric(sub[metric], errors="coerce").dropna()
                bench_rows.append({
                    "MevkiGroup": mg, "Metric": metric, "count": len(s),
                    "p10": s.quantile(0.10), "p25": s.quantile(0.25), "p50": s.quantile(0.50),
                    "p75": s.quantile(0.75), "p90": s.quantile(0.90), "mean": s.mean(),
                })
    bench = pd.DataFrame(bench_rows)

    if not t5_scored.empty and "MevkiGroup" in sl_scored.columns:
        for metric in ["Defans_Skoru", "OyunKurucu_Skoru", "Hucum_Skoru", "Overall_Skoru", "G+A_p90"]:
            sl_scored[f"Top5Percentile_{metric}"] = np.nan
        for mg in sl_scored["MevkiGroup"].astype(str).unique():
            dist_df = t5_scored[t5_scored["MevkiGroup"].astype(str) == mg]
            if dist_df.empty: continue
            for metric in ["Defans_Skoru", "OyunKurucu_Skoru", "Hucum_Skoru", "Overall_Skoru", "G+A_p90"]:
                dist = pd.to_numeric(dist_df[metric], errors="coerce").dropna()
                if dist.empty: continue
                mask = sl_scored["MevkiGroup"].astype(str) == mg
                sl_scored.loc[mask, f"Top5Percentile_{metric}"] = sl_scored.loc[mask, metric].apply(
                    lambda v: _compute_percentile(dist, float(v))
                )

    os.makedirs(os.path.dirname(superlig_out), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_out_prefix), exist_ok=True)
    
    sl_scored.to_csv(superlig_out, index=False, encoding="utf-8-sig")
    t5_scored.to_csv(top5_out, index=False, encoding="utf-8-sig")
    bench.to_csv(benchmark_out, index=False, encoding="utf-8-sig")

    np.save(scaler_out_prefix + "_min.npy", scaler.data_min_)
    np.save(scaler_out_prefix + "_max.npy", scaler.data_max_)
    with open(scaler_out_prefix + "_features.txt", "w", encoding="utf-8") as f:
        for c in COMMON_METRICS: f.write(c + "\n")

    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"📂 Süper Lig Çıktısı: {superlig_out}")
    print(f"📂 Top 5 Çıktısı: {top5_out}")
    print(f"📂 Benchmark: {benchmark_out}")

if __name__ == "__main__":
    top5_input = "data/raw/big_5_players_stats_2023_2024.csv"
    if not os.path.exists(top5_input):
        top5_input = "data/processed/top5_players_scored.csv"

    build_scored_and_benchmark(
        superlig_in="data/processed/super_lig_final_veri_role_enriched_CLEAN.csv",
        top5_in=top5_input,
        superlig_out="data/processed/superlig_scored_benchmark.csv",
        top5_out="data/processed/top5_players_scored.csv",
        benchmark_out="data/processed/top5_benchmarks_percentiles.csv",
        scaler_out_prefix="models/combined_minmax",
    )