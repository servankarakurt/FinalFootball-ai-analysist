import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    # GK
    "kaleci_sut_karsilama_yuzde",
    "kaleci_kurtaris_sayisi_p90",
    "kaleci_pas_isabet_yuzde",
    "kaleci_sahipsiz_top_kazanma_p90",
    "kaleci_yenilen_gol_p90",
]

def _compute_percentile(dist: pd.Series, v: float) -> float:
    # dist: Top-5 dağılımı
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
    return _ensure_numeric(df, COMMON_METRICS)

def _make_common_cols_top5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xG_p90_common"] = pd.to_numeric(df.get("xG_p90", 0.0), errors="coerce").fillna(0.0)
    df["xA_p90_common"] = pd.to_numeric(df.get("xA_p90", 0.0), errors="coerce").fillna(0.0)
    return _ensure_numeric(df, COMMON_METRICS)

def _score_from_norm(df_norm: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizasyonu (MinMax) aynı scaler ile yapılmış df_norm üzerinden skor üretir.
    Bu sayede Super Lig ve Top-5 skorları aynı ölçekte karşılaştırılabilir olur.
    """

    # DEFANS (kayarak müdahale yoksa zaten 0)
    w_def = {
        "top_calma_p90": 0.25,
        "uzaklastirma_p90": 0.20,
        "ikili_mucadele_kazanma_yuzde": 0.30,
        "hava_topu_kazanma_yuzde": 0.25,
    }

    df_out["Defans_Skoru"] = 0.0
    for k, w in w_def.items():
        df_out["Defans_Skoru"] += df_norm[k] * w

    # OYUN KURUCU
    w_play = {
        "kilit_pas_p90": 0.30,
        "ara_pas_p90": 0.15,
        "uzun_pas_p90": 0.15,
        "xA_p90_common": 0.20,
        "pas_isabet_yuzde": 0.15,
        "dribbling_p90": 0.05,
    }

    df_out["OyunKurucu_Skoru"] = 0.0
    for k, w in w_play.items():
        df_out["OyunKurucu_Skoru"] += df_norm[k] * w

    # HÜCUM
    w_att = {
        "sut_p90": 0.20,
        "xG_p90_common": 0.25,
        "Gls_p90": 0.20,
        "Ast_p90": 0.10,
        "xA_p90_common": 0.10,
        "dribbling_p90": 0.10,
        "orta_sayisi_p90": 0.05,
    }

    df_out["Hucum_Skoru"] = 0.0
    for k, w in w_att.items():
        df_out["Hucum_Skoru"] += df_norm[k] * w

    # KALECİ
    df_out["Kaleci_Skoru"] = 0.0
    w_gk = {
        "kaleci_sut_karsilama_yuzde": 0.35,
        "kaleci_kurtaris_sayisi_p90": 0.25,
        "kaleci_pas_isabet_yuzde": 0.20,
        "kaleci_sahipsiz_top_kazanma_p90": 0.20,
    }
    gk = np.zeros(len(df_out))
    for k, w in w_gk.items():
        gk += df_norm[k].to_numpy() * w
    # Yenilen gol ters
    gk += (1 - df_norm["kaleci_yenilen_gol_p90"].to_numpy()) * 0.25

    is_gk = df_out["MevkiGroup"].astype(str).eq("GK")
    df_out.loc[is_gk, "Kaleci_Skoru"] = gk[is_gk.to_numpy()]

    # OVERALL
    overall = []
    for i in range(len(df_out)):
        d = df_out.at[i, "Defans_Skoru"]
        p = df_out.at[i, "OyunKurucu_Skoru"]
        a = df_out.at[i, "Hucum_Skoru"]
        g = df_out.at[i, "Kaleci_Skoru"]
        mg = str(df_out.at[i, "MevkiGroup"])

        if mg == "GK":
            o = g
        elif mg == "CB":
            o = 0.7 * d + 0.2 * p + 0.1 * a
        elif mg == "FB":
            o = 0.5 * d + 0.3 * p + 0.2 * a
        elif mg in ["DM", "MID"]:
            o = 0.3 * d + 0.5 * p + 0.2 * a
        elif mg == "AM":
            o = 0.2 * d + 0.5 * p + 0.3 * a
        elif mg == "WING":
            o = 0.1 * d + 0.3 * p + 0.6 * a
        elif mg == "FWD":
            o = 0.05 * d + 0.15 * p + 0.80 * a
        else:
            o = 0.33 * d + 0.33 * p + 0.34 * a

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
    """
    1) Super Lig (sentetik) + Top5 (FBref) -> ortak metrik uzayında MinMax scaler fit
    2) İki taraf için aynı normalize ile skor üret
    3) Top5 dağılımlarını (percentile tablosu) çıkar
    4) Super Lig oyuncularına Top5 percentile etiketleri ekle
    5) scaler'ı kaydet (Streamlit similarity için aynı uzayı kullanacağız)
    """
    sl = pd.read_csv(superlig_in)
    t5 = pd.read_csv(top5_in)

    # Ortak kolonlar
    sl2 = _make_common_cols_superlig(sl).copy()
    t52 = _make_common_cols_top5(t5).copy()

    sl2["_source"] = "superlig"
    t52["_source"] = "top5"
    combined = pd.concat([sl2, t52], ignore_index=True)

    scaler = MinMaxScaler()
    combined_norm = combined.copy()
    combined_norm[COMMON_METRICS] = scaler.fit_transform(combined[COMMON_METRICS])

    # Skor üret
    combined_scored = _score_from_norm(combined_norm, combined.copy())

    # Ayrıştır
    sl_scored = combined_scored[combined_scored["_source"] == "superlig"].drop(columns=["_source"])
    t5_scored = combined_scored[combined_scored["_source"] == "top5"].drop(columns=["_source"])

    # Top5 benchmark (quantile table)
    bench_rows = []
    for mg in sorted(t5_scored["MevkiGroup"].astype(str).unique()):
        sub = t5_scored[t5_scored["MevkiGroup"].astype(str) == mg]
        for metric in ["Defans_Skoru", "OyunKurucu_Skoru", "Hucum_Skoru", "Overall_Skoru", "G+A_p90"]:
            s = pd.to_numeric(sub[metric], errors="coerce").dropna()
            bench_rows.append({
                "MevkiGroup": mg,
                "Metric": metric,
                "count": len(s),
                "p10": s.quantile(0.10),
                "p25": s.quantile(0.25),
                "p50": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "p90": s.quantile(0.90),
                "mean": s.mean(),
            })
    bench = pd.DataFrame(bench_rows)

    # Super Lig percentile
    for metric in ["Defans_Skoru", "OyunKurucu_Skoru", "Hucum_Skoru", "Overall_Skoru", "G+A_p90"]:
        sl_scored[f"Top5Percentile_{metric}"] = np.nan

    for mg in sl_scored["MevkiGroup"].astype(str).unique():
        dist_df = t5_scored[t5_scored["MevkiGroup"].astype(str) == mg]
        if dist_df.empty:
            continue
        for metric in ["Defans_Skoru", "OyunKurucu_Skoru", "Hucum_Skoru", "Overall_Skoru", "G+A_p90"]:
            dist = pd.to_numeric(dist_df[metric], errors="coerce").dropna()
            if dist.empty:
                continue
            mask = sl_scored["MevkiGroup"].astype(str) == mg
            sl_scored.loc[mask, f"Top5Percentile_{metric}"] = sl_scored.loc[mask, metric].apply(
                lambda v: _compute_percentile(dist, float(v))
            )

    # Kaydet
    sl_scored.to_csv(superlig_out, index=False, encoding="utf-8-sig")
    t5_scored.to_csv(top5_out, index=False, encoding="utf-8-sig")
    bench.to_csv(benchmark_out, index=False, encoding="utf-8-sig")

    # scaler kaydet
    np.save(scaler_out_prefix + "_min.npy", scaler.data_min_)
    np.save(scaler_out_prefix + "_max.npy", scaler.data_max_)
    with open(scaler_out_prefix + "_features.txt", "w", encoding="utf-8") as f:
        for c in COMMON_METRICS:
            f.write(c + "\n")

    print("OK (SuperLig scored+bench):", superlig_out)
    print("OK (Top5 scored):", top5_out)
    print("OK (Bench):", benchmark_out)
    print("OK (Scaler):", scaler_out_prefix + "_min.npy / _max.npy / _features.txt")


if __name__ == "__main__":
    build_scored_and_benchmark(
        superlig_in="data/processed/super_lig_final_veri_role_enriched_CLEAN.csv",
        top5_in="data/processed/top5_players_2025_2026_clean.csv",
        superlig_out="data/processed/superlig_scored_benchmark.csv",
        top5_out="data/processed/top5_players_scored.csv",
        benchmark_out="data/processed/top5_benchmarks_percentiles.csv",
        scaler_out_prefix="models/combined_minmax",
    )