import pandas as pd
import numpy as np

INPUT_FILE = "data/interim/super_lig_final_veri_role_enriched.csv"
OUTPUT_FILE = "data/processed/super_lig_final_veri_role_enriched_CLEAN.csv"


# ----------------------------
# Yardımcı fonksiyonlar
# ----------------------------

def mevki_group(mevki, current_group):
    """
    FM_Mevki bilgisinden MevkiGroup'u türet.
    current_group: mevcut MevkiGroup (boş değilse onu koruyabiliriz).
    """
    m = str(mevki).lower()
    if "kaleci" in m:
        return "GK"
    if "stoper" in m:
        return "CB"
    if "bek" in m:
        return "FB"
    if "kanat" in m:
        return "WING"
    if "santrafor" in m or "forvet" in m:
        return "FWD"
    if "numara" in m:
        return "AM"
    if "orta" in m:
        return "MID"

    # Eğer FM_Mevki işe yaramadıysa, mevcut grubu koru
    if isinstance(current_group, str) and current_group.strip() != "":
        return current_group
    return "MID"


def main():
    print("Veri yükleniyor:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    # ----------------------------
    # 1) Takım isimlerini düzelt
    # ----------------------------
    team_fix = {
        "Galatasayar": "Galatasaray",
        "fenerbahce": "Fenerbahçe",
        "kasimpasa": "Kasımpaşa",
        "Basaksehir": "Başakşehir",
    }
    if "Team" in df.columns:
        df["Team"] = df["Team"].replace(team_fix)
    else:
        print("Uyarı: 'Team' kolonu bulunamadı.")

    # ----------------------------
    # 2) Pos alanındaki virgüllü değerleri sadeleştir
    #    Örn: 'MF,FW' -> 'FW'
    # ----------------------------
    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].astype(str).apply(lambda x: x.split(",")[-1].strip())
    else:
        print("Uyarı: 'Pos' kolonu bulunamadı.")

    # ----------------------------
    # 3) MevkiGroup'u FM_Mevki'den yeniden üret
    # ----------------------------
    if "FM_Mevki" in df.columns:
        current_group = df["MevkiGroup"] if "MevkiGroup" in df.columns else ""
        df["MevkiGroup"] = [
            mevki_group(m, g) for m, g in zip(df["FM_Mevki"], current_group)
        ]
    else:
        print("Uyarı: 'FM_Mevki' kolonu bulunamadı, MevkiGroup güncellenmedi.")

    # ----------------------------
    # 4) Eksik FM_Rol için default rol ata
    # ----------------------------
    default_roles = {
        "CB": "Standart Stoper",
        "FB": "Standart Bek",
        "WING": "Kanat Oyuncusu",
        "MID": "Merkez Orta Saha",
        "AM": "On Numara",
        "FWD": "Yaratıcı Forvet",
        "GK": "Kaleci",
    }

    if "FM_Rol" not in df.columns:
        df["FM_Rol"] = np.nan

    def fill_role(row):
        if pd.notna(row["FM_Rol"]) and str(row["FM_Rol"]).strip() != "":
            return row["FM_Rol"]
        mg = row.get("MevkiGroup", "")
        return default_roles.get(mg, "Bilinmeyen")

    df["FM_Rol"] = df.apply(fill_role, axis=1)

    # FM_Rol_Base kolonunu tazele
    df["FM_Rol_Base"] = df["FM_Rol"].astype(str).str.split("(").str[0].str.strip()

    # ----------------------------
    # 5) Sayısal kolonlardaki uç değerleri kırp (winsorize 5–95%)
    # ----------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Eğer tüm değerler NaN ise atla
        if df[col].dropna().empty:
            continue
        low, high = df[col].quantile([0.05, 0.95])
        # low/high NaN olabilir, o zaman atla
        if pd.isna(low) or pd.isna(high):
            continue
        df[col] = df[col].clip(lower=low, upper=high)

    # ----------------------------
    # 6) Kaydet
    # ----------------------------
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("Temiz veri seti kaydedildi:", OUTPUT_FILE)
    print("Satır x Kolon:", df.shape)


if __name__ == "__main__":
    main()
