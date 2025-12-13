import pandas as pd

df = pd.read_csv("super_lig_final_veri_role_enriched.csv")
print(df.shape)
print(df.columns)

# Örnek: sadece stoperlerden birkaçını inceleyelim
cb = df[df["MevkiGroup"] == "CB"][[
    "Player", "Team", "FM_Rol_Base",
    "top_calma_p90", "kayarak_mudahale_p90",
    "uzaklastirma_p90", "uzun_pas_p90",
    "pas_isabet_yuzde"
]].head(10)
print(cb)
