import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Eksik olanları 0 ile dolduralım (normalize ederken sorun çıkmasın)
    metric_cols = [
        "top_calma_p90",
        "kayarak_mudahale_p90",
        "uzaklastirma_p90",
        "ikili_mucadele_kazanma_yuzde",
        "hava_topu_kazanma_yuzde",
        "orta_sayisi_p90",
        "sut_p90",
        "asist_p90_sentetik",
        "xA_p90_sentetik",
        "dribbling_p90",
        "kilit_pas_p90",
        "ara_pas_p90",
        "uzun_pas_p90",
        "sahipsiz_top_kazanma_p90",
        "hava_topu_mucadele_p90",
        "xG_p90_sentetik",
        "pas_isabet_yuzde",
        "kaleci_sut_karsilama_yuzde",
        "kaleci_kurtaris_sayisi_p90",
        "kaleci_pas_isabet_yuzde",
        "kaleci_sahipsiz_top_kazanma_p90",
        "kaleci_yenilen_gol_p90",
        # Gerçek golleri de hücum skoruna katalım
        "Gls_p90",
        "Ast_p90",
        "G+A_p90",
    ]
    for c in metric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Normalize edilecek metriklerden sadece mevcut olanları al
    present_metrics = [c for c in metric_cols if c in df.columns]

    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[present_metrics] = scaler.fit_transform(df[present_metrics])

    # 1) DEFANS SKORU
    # - Stoperler: top çalma, kayarak müdahale, uzaklaştırma, ikili & hava kazanma
    # - Bekler: defans metriklerine ek olarak ikili mücadele, sahipsiz top
    w_def = {
        "top_calma_p90": 0.25,
        "kayarak_mudahale_p90": 0.15,
        "uzaklastirma_p90": 0.20,
        "ikili_mucadele_kazanma_yuzde": 0.20,
        "hava_topu_kazanma_yuzde": 0.20,
    }

    def def_score(row):
        s = 0.0
        for k, w in w_def.items():
            if k in df_norm.columns:
                s += df_norm.loc[row.name, k] * w
        return s

    df["Defans_Skoru"] = df.apply(def_score, axis=1)

    # 2) OYUN KURUCU SKORU
    # - Kilit pas, ara pas, uzun pas, xA, pas isabet, dribbling
    w_play = {
        "kilit_pas_p90": 0.25,
        "ara_pas_p90": 0.15,
        "uzun_pas_p90": 0.15,
        "xA_p90_sentetik": 0.20,
        "pas_isabet_yuzde": 0.15,
        "dribbling_p90": 0.10,
    }

    def playmaker_score(row):
        s = 0.0
        for k, w in w_play.items():
            if k in df_norm.columns:
                s += df_norm.loc[row.name, k] * w
        return s

    df["OyunKurucu_Skoru"] = df.apply(playmaker_score, axis=1)

    # 3) HÜCUM SKORU
    # - Şut, xG, gol, asist, xA, dribbling, ortalar
    w_att = {
        "sut_p90": 0.20,
        "xG_p90_sentetik": 0.20,
        "Gls_p90": 0.20,
        "asist_p90_sentetik": 0.15,
        "xA_p90_sentetik": 0.10,
        "dribbling_p90": 0.10,
        "orta_sayisi_p90": 0.05,
    }

    def attack_score(row):
        s = 0.0
        for k, w in w_att.items():
            if k in df_norm.columns:
                s += df_norm.loc[row.name, k] * w
        return s

    df["Hucum_Skoru"] = df.apply(attack_score, axis=1)

    # 4) KALECİ SKORU
    # - Şut karşılama + kurtarış + pas isabet + sahipsiz top (pozitif)
    # - Yenilen gol (negatif)
    w_gk_pos = {
        "kaleci_sut_karsilama_yuzde": 0.35,
        "kaleci_kurtaris_sayisi_p90": 0.25,
        "kaleci_pas_isabet_yuzde": 0.20,
        "kaleci_sahipsiz_top_kazanma_p90": 0.20,
    }

    def gk_score(row):
        if row.get("MevkiGroup", "") != "GK":
            return 0.0

        s_pos = 0.0
        for k, w in w_gk_pos.items():
            if k in df_norm.columns:
                s_pos += df_norm.loc[row.name, k] * w

        # Yenilen golü ters yönden ekle (az gol yiyen daha iyi)
        if "kaleci_yenilen_gol_p90" in df_norm.columns:
            s_neg = df_norm.loc[row.name, "kaleci_yenilen_gol_p90"]
            # 1 - norm_value ile ters çevirelim
            s_pos += (1 - s_neg) * 0.25

        return s_pos

    df["Kaleci_Skoru"] = df.apply(gk_score, axis=1)

    # 5) Overall skor (mevkiye göre ağırlıklı kombinasyon)
    def overall_score(row):
        d = row["Defans_Skoru"]
        p = row["OyunKurucu_Skoru"]
        a = row["Hucum_Skoru"]
        g = row["Kaleci_Skoru"]
        mg = row.get("MevkiGroup", "")

        if mg == "GK":
            return g
        if mg == "CB":
            return 0.7*d + 0.2*p + 0.1*a
        if mg == "FB":
            return 0.5*d + 0.3*p + 0.2*a
        if mg in ["DM", "MID"]:
            return 0.3*d + 0.5*p + 0.2*a
        if mg == "AM":
            return 0.2*d + 0.5*p + 0.3*a
        if mg == "WING":
            return 0.1*d + 0.3*p + 0.6*a
        if mg == "FWD":
            return 0.05*d + 0.15*p + 0.80*a

        # default
        return 0.33*d + 0.33*p + 0.34*a

    df["Overall_Skoru"] = df.apply(overall_score, axis=1)

    return df

if __name__ == "__main__":
    # Temizlenmiş veri setini yükle
    input_file = "data/processed/super_lig_final_veri_role_enriched_CLEAN.csv"
    output_file = "data/processed/super_lig_final_veri_scored.csv"

    df = pd.read_csv(input_file)
    df_scored = add_composite_scores(df)
    df_scored.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Skorları eklenmiş veri seti kaydedildi:", output_file)
    print("Satır x Kolon:", df_scored.shape)
    print(
        df_scored[
            [
                "Player",
                "Team",
                "FM_Mevki",
                "FM_Rol_Base",
                "Defans_Skoru",
                "OyunKurucu_Skoru",
                "Hucum_Skoru",
                "Kaleci_Skoru",
                "Overall_Skoru",   # ← bunu ekledim
            ]
        ].head()
    )