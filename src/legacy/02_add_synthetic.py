import pandas as pd
import numpy as np

# ---------------------------------------------------
# Yardımcı: FM_Mevki bilgisinden ana mevki grubu
# ---------------------------------------------------
def mevki_group(m):
    if not isinstance(m, str):
        return "MID"
    m_low = m.lower()

    if "kaleci" in m_low:
        return "GK"
    if "stoper" in m_low:
        return "CB"
    if "bek" in m_low:
        return "FB"
    if "ön libero" in m_low or "defansif orta saha" in m_low:
        return "DM"
    if "merkez orta saha" in m_low or "iki yönlü orta saha" in m_low or "mezzala" in m_low:
        return "CM"
    if "on numara" in m_low or "ofansif orta saha" in m_low:
        return "AM"
    if "kanat" in m_low:
        return "WING"
    if "santrafor" in m_low or "forvet" in m_low:
        return "FWD"
    return "MID"


# ---------------------------------------------------
# Rol-farkında sentetik istatistik üretici
# ---------------------------------------------------
def add_synthetic_features_role_aware(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    df = df.copy()

    if "FM_Mevki" not in df.columns or "FM_Rol" not in df.columns:
        raise ValueError("Bu fonksiyonu kullanmak için 'FM_Mevki' ve 'FM_Rol' kolonlarına ihtiyacımız var.")

    # Ana mevki grubu
    df["MevkiGroup"] = df["FM_Mevki"].apply(mevki_group)

    # FM_Rol_Base: "Pas Dağıtan Stoper (Savunma)" -> "Pas Dağıtan Stoper"
    df["FM_Rol_Base"] = df["FM_Rol"].fillna("").str.split("(").str[0].str.strip()

    # Açılacak sentetik kolonlar
    new_cols = [
        # defans / genel
        "top_calma_p90",
        "kayarak_mudahale_p90",
        "uzaklastirma_p90",
        "ikili_mucadele_kazanma_yuzde",
        "hava_topu_kazanma_yuzde",

        # hücum / pas / oyun kurulum
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

        # kaleci
        "kaleci_sut_karsilama_yuzde",
        "kaleci_kurtaris_sayisi_p90",
        "kaleci_pas_isabet_yuzde",
        "kaleci_sahipsiz_top_kazanma_p90",
        "kaleci_yenilen_gol_p90",
    ]

    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    # ========= 1) KALECİLER =========
    gk = df["MevkiGroup"] == "GK"
    if gk.sum() > 0:
        is_sweeper = gk & df["FM_Rol_Base"].str.contains("Libero Kaleci", na=False)
        is_normal = gk & ~is_sweeper

        # Geleneksel kaleci
        df.loc[is_normal, "kaleci_sut_karsilama_yuzde"] = np.random.uniform(55, 75, is_normal.sum())
        df.loc[is_normal, "kaleci_kurtaris_sayisi_p90"] = np.random.uniform(2, 5, is_normal.sum())
        df.loc[is_normal, "kaleci_pas_isabet_yuzde"] = np.random.uniform(60, 80, is_normal.sum())
        df.loc[is_normal, "kaleci_sahipsiz_top_kazanma_p90"] = np.random.uniform(0.2, 0.8, is_normal.sum())
        df.loc[is_normal, "kaleci_yenilen_gol_p90"] = np.random.uniform(0.5, 2.0, is_normal.sum())

        # Libero kaleci
        df.loc[is_sweeper, "kaleci_sut_karsilama_yuzde"] = np.random.uniform(55, 78, is_sweeper.sum())
        df.loc[is_sweeper, "kaleci_kurtaris_sayisi_p90"] = np.random.uniform(2, 6, is_sweeper.sum())
        df.loc[is_sweeper, "kaleci_pas_isabet_yuzde"] = np.random.uniform(70, 88, is_sweeper.sum())
        df.loc[is_sweeper, "kaleci_sahipsiz_top_kazanma_p90"] = np.random.uniform(0.5, 2.0, is_sweeper.sum())
        df.loc[is_sweeper, "kaleci_yenilen_gol_p90"] = np.random.uniform(0.3, 1.8, is_sweeper.sum())

    # ========= 2) STOPERLER (CB) =========
    cb = df["MevkiGroup"] == "CB"
    if cb.sum() > 0:
        is_passer = cb & df["FM_Rol_Base"].str.contains("Pas", na=False)  # Pas Dağıtan Stoper, Pasör Stoper vs.
        is_no_nonsense = cb & df["FM_Rol_Base"].str.contains("Standart Stoper|Çakılı Stoper|No-nonsense", na=False)
        is_other_cb = cb & ~(is_passer | is_no_nonsense)

        # Çakılı / Standart stoper
        df.loc[is_no_nonsense, "top_calma_p90"] = np.random.uniform(1.5, 3.5, is_no_nonsense.sum())
        df.loc[is_no_nonsense, "kayarak_mudahale_p90"] = np.random.uniform(0.7, 1.8, is_no_nonsense.sum())
        df.loc[is_no_nonsense, "uzaklastirma_p90"] = np.random.uniform(4, 12, is_no_nonsense.sum())
        df.loc[is_no_nonsense, "ikili_mucadele_kazanma_yuzde"] = np.random.uniform(50, 75, is_no_nonsense.sum())
        df.loc[is_no_nonsense, "hava_topu_kazanma_yuzde"] = np.random.uniform(50, 80, is_no_nonsense.sum())
        df.loc[is_no_nonsense, "uzun_pas_p90"] = np.random.uniform(1, 4, is_no_nonsense.sum())
        df.loc[is_no_nonsense, "pas_isabet_yuzde"] = np.random.uniform(75, 86, is_no_nonsense.sum())

        # Pasör stoper tarzı
        df.loc[is_passer, "top_calma_p90"] = np.random.uniform(1.0, 2.8, is_passer.sum())
        df.loc[is_passer, "kayarak_mudahale_p90"] = np.random.uniform(0.4, 1.3, is_passer.sum())
        df.loc[is_passer, "uzaklastirma_p90"] = np.random.uniform(3, 9, is_passer.sum())
        df.loc[is_passer, "ikili_mucadele_kazanma_yuzde"] = np.random.uniform(48, 72, is_passer.sum())
        df.loc[is_passer, "hava_topu_kazanma_yuzde"] = np.random.uniform(45, 75, is_passer.sum())
        df.loc[is_passer, "uzun_pas_p90"] = np.random.uniform(4, 10, is_passer.sum())
        df.loc[is_passer, "pas_isabet_yuzde"] = np.random.uniform(80, 92, is_passer.sum())

        # Diğer stoperler
        df.loc[is_other_cb, "top_calma_p90"] = np.random.uniform(1.0, 3.0, is_other_cb.sum())
        df.loc[is_other_cb, "kayarak_mudahale_p90"] = np.random.uniform(0.5, 1.5, is_other_cb.sum())
        df.loc[is_other_cb, "uzaklastirma_p90"] = np.random.uniform(3, 10, is_other_cb.sum())
        df.loc[is_other_cb, "ikili_mucadele_kazanma_yuzde"] = np.random.uniform(48, 72, is_other_cb.sum())
        df.loc[is_other_cb, "hava_topu_kazanma_yuzde"] = np.random.uniform(45, 78, is_other_cb.sum())
        df.loc[is_other_cb, "uzun_pas_p90"] = np.random.uniform(2, 7, is_other_cb.sum())
        df.loc[is_other_cb, "pas_isabet_yuzde"] = np.random.uniform(78, 90, is_other_cb.sum())

    # ========= 3) BEKLER (FB) =========
    fb = df["MevkiGroup"] == "FB"
    if fb.sum() > 0:
        base = fb
        df.loc[base, "top_calma_p90"] = np.random.uniform(0.8, 2.5, base.sum())
        df.loc[base, "ikili_mucadele_kazanma_yuzde"] = np.random.uniform(40, 65, base.sum())

        is_wing_back = fb & df["FM_Rol_Base"].str.contains("Kanat Bek|Ofansif Bek|İki Yönlü Bek", na=False)
        is_inverted = fb & df["FM_Rol_Base"].str.contains("Sahte Bek|Sigorta Bek", na=False)
        is_no_fb = fb & df["FM_Rol_Base"].str.contains("Çakılı Bek", na=False)
        is_std = fb & ~(is_wing_back | is_inverted | is_no_fb)

        # Kanat/iki yönlü bek
        df.loc[is_wing_back, "orta_sayisi_p90"] = np.random.uniform(3, 9, is_wing_back.sum())
        df.loc[is_wing_back, "dribbling_p90"] = np.random.uniform(1, 4, is_wing_back.sum())
        df.loc[is_wing_back, "pas_isabet_yuzde"] = np.random.uniform(75, 88, is_wing_back.sum())

        # Sahte / sigorta bek
        df.loc[is_inverted, "orta_sayisi_p90"] = np.random.uniform(1, 4, is_inverted.sum())
        df.loc[is_inverted, "kilit_pas_p90"] = np.random.uniform(0.5, 2.0, is_inverted.sum())
        df.loc[is_inverted, "pas_isabet_yuzde"] = np.random.uniform(80, 90, is_inverted.sum())

        # Çakılı bek
        df.loc[is_no_fb, "orta_sayisi_p90"] = np.random.uniform(0.3, 2.0, is_no_fb.sum())
        df.loc[is_no_fb, "pas_isabet_yuzde"] = np.random.uniform(70, 84, is_no_fb.sum())

        # Standart bek
        df.loc[is_std, "orta_sayisi_p90"] = np.random.uniform(1, 5, is_std.sum())
        df.loc[is_std, "pas_isabet_yuzde"] = np.random.uniform(74, 88, is_std.sum())

    # ========= 4) ORTA SAHALAR (DM/CM/AM/MID) =========
    mid = df["MevkiGroup"].isin(["DM", "CM", "AM", "MID"])
    if mid.sum() > 0:
        base = mid
        df.loc[base, "sahipsiz_top_kazanma_p90"] = np.random.uniform(3, 9, base.sum())
        df.loc[base, "dribbling_p90"] = np.random.uniform(0.8, 3.5, base.sum())
        df.loc[base, "sut_p90"] = np.random.uniform(0.4, 2.5, base.sum())
        df.loc[base, "asist_p90_sentetik"] = np.random.uniform(0.0, 0.25, base.sum())
        df.loc[base, "xA_p90_sentetik"] = np.random.uniform(0.0, 0.35, base.sum())

        is_regista = mid & df["FM_Rol_Base"].str.contains("Regista|Derin Oyun Kurucu|Gelişmiş Oyun Kurucu|Gezgin Oyun Kurucu", na=False)
        is_b2b = mid & df["FM_Rol_Base"].str.contains("İki Yönlü Orta Saha|Box to Box", na=False)
        is_ball_winner = mid & df["FM_Rol_Base"].str.contains("Savaşçı Orta Saha|Top kapma", na=False)
        is_anchor = mid & df["FM_Rol_Base"].str.contains("Ön Libero|Anchor", na=False)
        is_mezzala = mid & df["FM_Rol_Base"].str.contains("Mezzala", na=False)
        is_other_mid = mid & ~(is_regista | is_b2b | is_ball_winner | is_anchor | is_mezzala)

        df.loc[is_regista, "kilit_pas_p90"] = np.random.uniform(2, 6, is_regista.sum())
        df.loc[is_regista, "ara_pas_p90"] = np.random.uniform(1.0, 3.0, is_regista.sum())
        df.loc[is_regista, "uzun_pas_p90"] = np.random.uniform(4, 10, is_regista.sum())
        df.loc[is_regista, "pas_isabet_yuzde"] = np.random.uniform(82, 93, is_regista.sum())

        df.loc[is_b2b, "kilit_pas_p90"] = np.random.uniform(1, 4, is_b2b.sum())
        df.loc[is_b2b, "ara_pas_p90"] = np.random.uniform(0.6, 2.0, is_b2b.sum())
        df.loc[is_b2b, "uzun_pas_p90"] = np.random.uniform(3, 8, is_b2b.sum())
        df.loc[is_b2b, "top_calma_p90"] = np.random.uniform(1.2, 3.5, is_b2b.sum())

        df.loc[is_ball_winner, "top_calma_p90"] = np.random.uniform(1.8, 4.5, is_ball_winner.sum())
        df.loc[is_ball_winner, "kayarak_mudahale_p90"] = np.random.uniform(0.8, 2.0, is_ball_winner.sum())
        df.loc[is_ball_winner, "sahipsiz_top_kazanma_p90"] = np.random.uniform(4, 11, is_ball_winner.sum())

        df.loc[is_anchor, "top_calma_p90"] = np.random.uniform(1.5, 3.5, is_anchor.sum())
        df.loc[is_anchor, "uzun_pas_p90"] = np.random.uniform(2, 7, is_anchor.sum())

        df.loc[is_mezzala, "dribbling_p90"] = np.random.uniform(2, 5, is_mezzala.sum())
        df.loc[is_mezzala, "sut_p90"] = np.random.uniform(1, 3, is_mezzala.sum())

        df.loc[is_other_mid, "kilit_pas_p90"] = np.random.uniform(1, 4, is_other_mid.sum())
        df.loc[is_other_mid, "ara_pas_p90"] = np.random.uniform(0.5, 2.0, is_other_mid.sum())
        df.loc[is_other_mid, "uzun_pas_p90"] = np.random.uniform(2, 7, is_other_mid.sum())
        df.loc[is_other_mid, "pas_isabet_yuzde"] = np.random.uniform(78, 90, is_other_mid.sum())

    # ========= 5) KANATLAR =========
    wing = df["MevkiGroup"] == "WING"
    if wing.sum() > 0:
        base = wing
        df.loc[base, "orta_sayisi_p90"] = np.random.uniform(2, 9, base.sum())
        df.loc[base, "dribbling_p90"] = np.random.uniform(2, 7, base.sum())
        df.loc[base, "sut_p90"] = np.random.uniform(1, 4, base.sum())
        df.loc[base, "asist_p90_sentetik"] = np.random.uniform(0, 0.4, base.sum())
        df.loc[base, "xA_p90_sentetik"] = np.random.uniform(0, 0.5, base.sum())

        is_inside = wing & df["FM_Rol_Base"].str.contains("Kanat Forvet|Ters Ayaklı Kanat|Inside forward", na=False)
        is_def_w = wing & df["FM_Rol_Base"].str.contains("Defansif Kanat", na=False)
        is_wpm = wing & df["FM_Rol_Base"].str.contains("Kanat Oyun Kurucu|Wide playmaker", na=False)
        is_wtm = wing & df["FM_Rol_Base"].str.contains("Hedef Kanat Oyuncusu|Wide Target Man", na=False)

        df.loc[is_inside, "sut_p90"] = np.random.uniform(2, 5, is_inside.sum())
        df.loc[is_inside, "xG_p90_sentetik"] = np.random.uniform(0.2, 0.9, is_inside.sum())

        df.loc[is_def_w, "top_calma_p90"] = np.random.uniform(1.5, 3.5, is_def_w.sum())
        df.loc[is_def_w, "ikili_mucadele_kazanma_yuzde"] = np.random.uniform(45, 70, is_def_w.sum())

        df.loc[is_wpm, "kilit_pas_p90"] = np.random.uniform(1.5, 4.0, is_wpm.sum())
        df.loc[is_wpm, "xA_p90_sentetik"] = np.random.uniform(0.2, 0.6, is_wpm.sum())

        df.loc[is_wtm, "hava_topu_mucadele_p90"] = np.random.uniform(3, 8, is_wtm.sum())

    # ========= 6) FORVETLER =========
    fwd = df["MevkiGroup"] == "FWD"
    if fwd.sum() > 0:
        base = fwd
        df.loc[base, "sut_p90"] = np.random.uniform(2, 6, base.sum())
        df.loc[base, "xG_p90_sentetik"] = np.random.uniform(0.2, 1.0, base.sum())
        df.loc[base, "hava_topu_mucadele_p90"] = np.random.uniform(1, 7, base.sum())

        is_poacher = fwd & df["FM_Rol_Base"].str.contains("Fırsatçı Golcü|Poacher", na=False)
        is_target = fwd & df["FM_Rol_Base"].str.contains("Pivot Santrafor|Target Man", na=False)
        is_pf = fwd & df["FM_Rol_Base"].str.contains("Çalışkan Forvet|Pressing Forward", na=False)
        is_complete = fwd & df["FM_Rol_Base"].str.contains("Komple Forvet|Complete Forward", na=False)

        df.loc[is_poacher, "sut_p90"] = np.random.uniform(3, 7, is_poacher.sum())
        df.loc[is_poacher, "xG_p90_sentetik"] = np.random.uniform(0.4, 1.2, is_poacher.sum())

        df.loc[is_target, "hava_topu_mucadele_p90"] = np.random.uniform(4, 10, is_target.sum())
        df.loc[is_target, "sahipsiz_top_kazanma_p90"] = np.random.uniform(1, 4, is_target.sum())

        df.loc[is_pf, "top_calma_p90"] = np.random.uniform(0.8, 2.5, is_pf.sum())
        df.loc[is_pf, "sahipsiz_top_kazanma_p90"] = np.random.uniform(2, 6, is_pf.sum())

        df.loc[is_complete, "dribbling_p90"] = np.random.uniform(1.5, 4.0, is_complete.sum())
        df.loc[is_complete, "kilit_pas_p90"] = np.random.uniform(0.8, 2.5, is_complete.sum())

    return df


INPUT_FILE = "data/interim/super_lig_final_veri_with_roles.csv"
OUTPUT_FILE = "data/interim/super_lig_final_veri_role_enriched.csv"

if __name__ == "__main__":
    df_roles = pd.read_csv(INPUT_FILE)
    df_enriched = add_synthetic_features_role_aware(df_roles, random_state=42)
    df_enriched.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Yeni veri kaydedildi:", OUTPUT_FILE)
    print("Satır x Kolon:", df_enriched.shape)
    print(df_enriched[["Player", "Team", "FM_Mevki", "FM_Rol", "MevkiGroup"]].head())