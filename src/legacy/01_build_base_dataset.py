import pandas as pd
import numpy as np

# 1) Ana veri
df_main = pd.read_csv("super_lig_final_veri.csv")

# Takım isimlerini sheet isimlerine map et
TEAM_SHEET_MAP = {
    'Alanya': 'Alanya',
    'Antalya': 'Antalya',
    'Basaksehir': 'Başakşehir',
    'Besiktas': 'Beşiktaş',
    'Eyupspor': 'Eyüpspor',
    'Fatihkaragumruk': 'Fatih Karagümrük',
    'fenerbahce': 'Fenerbahçe',
    'Galatasayar': 'Galatasaray',
    'Gaziantep': 'Gaziantep',
    'genclerbirligi': 'Gençlerbirliği',
    'Göztepe': 'Göztepe',
    'kasimpasa': 'Kasımpaşa',
    'kayserispor': 'Kayserispor',
    'kocaelispor': 'Kocaelispor',
    'Konyaspor': 'Konyaspor',
    'Rizespor': 'Rizespor',
    'Samsunspor': 'Samsunspor',
    'Trabzonspor': 'Trabzonspor'
}

df_main["TeamSheet"] = df_main["Team"].map(TEAM_SHEET_MAP)

# 2) Roller dosyasını okuyalım
roles_xls = pd.ExcelFile("Oyuncu rolleri.xlsx")
all_roles = []

for sheet in roles_xls.sheet_names:
    df_r = pd.read_excel(roles_xls, sheet_name=sheet)

    # Sadece gerçek oyuncular: Mevki dolu, grup başlıklarını at
    df_r = df_r[df_r["Mevki"].notna()].copy()
    df_r = df_r[~df_r["Oyuncu Adı"].isin([
        "Kaleciler",
        "Defans Oyuncuları",
        "Orta Saha Oyuncuları",
        "Kanat Oyuncuları",
        "Forvetler"
    ])]

    df_r["TeamSheet"] = sheet
    all_roles.append(df_r[["Oyuncu Adı", "Mevki", "En Uygun Rol (Rehbere Göre)", "TeamSheet"]])

roles_df = pd.concat(all_roles, ignore_index=True)

# 3) Player + TeamSheet üzerinden merge
merged = df_main.merge(
    roles_df,
    left_on=["Player", "TeamSheet"],
    right_on=["Oyuncu Adı", "TeamSheet"],
    how="left"
)

# Yardımcı kolonları düzenleyelim
merged["FM_Mevki"] = merged["Mevki"]
merged["FM_Rol"] = merged["En Uygun Rol (Rehbere Göre)"]

# Ana isimleri bırakıp gereksizleri silelim
merged = merged.drop(columns=["Oyuncu Adı", "Mevki", "En Uygun Rol (Rehbere Göre)"])

# FM rolünü base ve görev olarak ikiye ayır (ör: "Pasör Stoper (Savunma)")
merged["FM_Rol_Base"] = merged["FM_Rol"].str.split("(").str[0].str.strip()
merged["FM_Rol_Gorev"] = merged["FM_Rol"].str.extract(r"\((.*?)\)").iloc[:, 0]

merged.to_csv("super_lig_final_veri_with_roles.csv", index=False, encoding="utf-8-sig")

print("Kayıt sayısı:", len(merged))
print(merged[["Player", "Team", "Pos", "FM_Mevki", "FM_Rol"]].head(20))
