import pandas as pd
import numpy as np
import re

def map_mevki_group_fbref(pos: str) -> str:
    """
    FBref Pos alanı örn: "DF,MF" / "FW,MF" / "GK"
    Projedeki MevkiGroup sınıflarına kaba ama pratik bir map uygular.
    """
    pos = str(pos)
    parts = [p.strip() for p in pos.split(",") if p.strip()]

    if "GK" in parts:
        return "GK"
    if "DF" in parts and "MF" in parts:
        return "FB"
    if "MF" in parts and "FW" in parts:
        return "WING"
    if parts and parts[0] == "DF":
        return "CB"
    if parts and parts[0] == "MF":
        return "MID"
    if parts and parts[0] == "FW":
        return "FWD"
    return "MID"


def ingest_top5_players(path_in: str, path_out: str) -> pd.DataFrame:
    """
    Top-5 lig datasetini (players_data-2025_2026.csv) standardize eder.
    Çıktı: Team/League/Pos_Simple/MevkiGroup + temel per90 metrikleri + bazı proxy kolonlar.
    """
    df = pd.read_csv(path_in)

    # Kolon standartlaştırma
    if "Squad" in df.columns:
        df = df.rename(columns={"Squad": "Team"})
    if "Comp" in df.columns:
        df = df.rename(columns={"Comp": "League"})

    # Pos alanları
    df["Pos"] = df["Pos"].astype(str)
    df["Pos_Simple"] = df["Pos"].str.split(",").str[0]
    df["MevkiGroup"] = df["Pos"].apply(map_mevki_group_fbref)

    # numeric safe
    numeric_candidates = [
        "90s","Gls","Ast","xG","xAG","Sh","SoT","Crs",
        "Tkl","Int","Clr","KP","TB","PrgP","PrgC",
        "Cmp%","Won%","Tkl%","Saves","Save%","GA90"
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["90s"] = df["90s"].fillna(0.0)
    denom = df["90s"].replace(0, np.nan)

    def per90(src: str, dst: str):
        if src in df.columns:
            df[dst] = (df[src] / denom).fillna(0.0)
        else:
            df[dst] = 0.0

    # Hücum
    per90("Gls", "Gls_p90")
    per90("Ast", "Ast_p90")
    df["G+A_p90"] = ((df.get("Gls", 0).fillna(0) + df.get("Ast", 0).fillna(0)) / denom).fillna(0.0)
    per90("xG", "xG_p90")
    per90("xAG", "xA_p90")  # FBref xAG -> xA proxy
    per90("Sh", "sut_p90")
    per90("SoT", "sot_p90")
    per90("Crs", "orta_sayisi_p90")

    # Savunma / oyun kurucu proxy
    per90("Tkl", "top_calma_p90")           # tackles proxy
    per90("Int", "interception_p90")        # ekstra bilgi
    per90("Clr", "uzaklastirma_p90")
    per90("KP", "kilit_pas_p90")
    per90("TB", "ara_pas_p90")              # through balls proxy
    per90("PrgP", "uzun_pas_p90")           # progressive passes proxy
    per90("PrgC", "dribbling_p90")          # progressive carries proxy

    # yüzdeler
    df["pas_isabet_yuzde"] = df["Cmp%"].fillna(0.0) if "Cmp%" in df.columns else 0.0
    df["hava_topu_kazanma_yuzde"] = df["Won%"].fillna(0.0) if "Won%" in df.columns else 0.0
    df["ikili_mucadele_kazanma_yuzde"] = df["Tkl%"].fillna(0.0) if "Tkl%" in df.columns else 0.0

    # Kaleci proxy
    df["kaleci_sut_karsilama_yuzde"] = df["Save%"].fillna(0.0) if "Save%" in df.columns else 0.0
    df["kaleci_kurtaris_sayisi_p90"] = (df["Saves"] / denom).fillna(0.0) if "Saves" in df.columns else 0.0
    df["kaleci_yenilen_gol_p90"] = df["GA90"].fillna(0.0) if "GA90" in df.columns else 0.0
    df["kaleci_pas_isabet_yuzde"] = 0.0
    df["kaleci_sahipsiz_top_kazanma_p90"] = 0.0

    # Bozuk /90 kolon adı varsa normalize et
    if "/90" in df.columns:
        df = df.rename(columns={"/90": "unknown_per90_metric"})

    df.to_csv(path_out, index=False, encoding="utf-8-sig")
    print("OK (Top5 ingest):", path_out, df.shape)
    return df


if __name__ == "__main__":
    ingest_top5_players(
        "data/external/players_data-2025_2026.csv",
        "data/processed/top5_players_2025_2026_clean.csv"
    )