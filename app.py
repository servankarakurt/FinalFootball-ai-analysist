import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Futbol AI Scout", page_icon="⚽", layout="wide")

# --- CSS ---
st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Yapay Zeka Destekli Futbol Analiz Sistemi")
st.markdown("**2025-2026 Süper Lig: Hücum vs Savunma Analizi**")

# --- DOSYA YOLLARI ---
DATA_FILE = "data/processed/super_lig_final_veri_scored.csv"
MODEL_FILE = "models/futbol_ann_ga.h5"
SCALER_MEAN_FILE = "models/scaler_mean.npy"
SCALER_SCALE_FILE = "models/scaler_scale.npy"
FEATURE_NAMES_FILE = "models/feature_names.txt"


# --- VERİ & MODEL YÜKLEME ---
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv(DATA_FILE)

    # Eski UI kolon adlarına uyum
    if "Takim" not in df.columns and "Team" in df.columns:
        df["Takim"] = df["Team"]

    if "Pos_Simple" not in df.columns:
        if "Pos" in df.columns:
            df["Pos_Simple"] = df["Pos"].astype(str).str.split(",").str[0]
        else:
            df["Pos_Simple"] = "MF"

    if "Rol" not in df.columns:
        if "FM_Rol_Base" in df.columns:
            df["Rol"] = df["FM_Rol_Base"]
        else:
            df["Rol"] = ""

    if "MevkiGroup" not in df.columns:
        # fallback: hiç yoksa MF yap
        df["MevkiGroup"] = "MID"

    # Model + scaler
    model = keras.models.load_model(MODEL_FILE, compile=False)
    mean_ = np.load(SCALER_MEAN_FILE)
    scale_ = np.load(SCALER_SCALE_FILE)
    with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
        feature_names = [line.strip() for line in f.readlines()]

    return df, model, mean_, scale_, feature_names


def prepare_features(row: pd.Series, feature_names, mean_, scale_):
    """Tek bir oyuncu Series'inden ANN feature vektörü hazırla."""
    x = np.array([row.get(col, 0.0) for col in feature_names], dtype=float)
    x_scaled = (x - mean_) / scale_
    return x_scaled.reshape(1, -1)


df, model, mean_, scale_, feature_names = load_data_and_model()

# --- KENAR ÇUBUĞU ---
st.sidebar.header("🕵️‍♂️ Oyuncu Seçimi")

mode = st.sidebar.radio(
    "🎛️ Karşılaştırma Modu",
    ["Serbest Matchup (Hücum vs Rakip)", "Aynı Mevki Kıyas (MevkiGroup)"],
    index=0,
)

takimlar = sorted(df["Takim"].astype(str).unique())

# --- Mod'a göre seçimler ---
if mode == "Serbest Matchup (Hücum vs Rakip)":
    st.sidebar.subheader("1. Oyuncu (Hücum)")
    t1 = st.sidebar.selectbox("Takım", takimlar, index=0, key="t1_free")
    o1_list = df[df["Takim"] == t1]["Player"].tolist()
    o1_name = st.sidebar.selectbox("Oyuncu", o1_list, key="o1_free")

    st.sidebar.subheader("2. Oyuncu (Rakip)")
    t2 = st.sidebar.selectbox("Takım ", takimlar, index=min(1, len(takimlar) - 1), key="t2_free")
    o2_list = df[df["Takim"] == t2]["Player"].tolist()
    if not o2_list:
        o2_list = df["Player"].tolist()
    o2_name = st.sidebar.selectbox("Oyuncu ", o2_list, key="o2_free")

    selected_mg = None

else:
    st.sidebar.subheader("Aynı Mevki Kıyas Ayarları")

    mg_list_pref = ["GK", "CB", "FB", "MID", "AM", "WING", "FWD"]
    available_mg = [m for m in mg_list_pref if m in df["MevkiGroup"].astype(str).unique()]
    if not available_mg:
        available_mg = sorted(df["MevkiGroup"].astype(str).unique().tolist())

    selected_mg = st.sidebar.selectbox("MevkiGroup", available_mg, key="mg_same")

    df_mg = df[df["MevkiGroup"].astype(str) == str(selected_mg)].copy()

    # Top 10 Overall (opsiyonel özellik)
    st.sidebar.markdown("### 🔝 Top 10 (Overall)")
    top10 = (
        df_mg.sort_values("Overall_Skoru", ascending=False)
        .loc[:, ["Player", "Takim", "Overall_Skoru", "Hucum_Skoru", "Defans_Skoru"]]
        .head(10)
    )
    st.sidebar.dataframe(top10, use_container_width=True, height=280)

    st.sidebar.subheader("1. Oyuncu")
    t1 = st.sidebar.selectbox("Takım", sorted(df_mg["Takim"].astype(str).unique()), key="t1_same")
    p1_list = df_mg[df_mg["Takim"] == t1]["Player"].tolist()
    o1_name = st.sidebar.selectbox("Oyuncu", p1_list, key="o1_same")

    st.sidebar.subheader("2. Oyuncu")
    t2_candidates = sorted(df_mg["Takim"].astype(str).unique())
    default_idx = 0 if len(t2_candidates) == 1 else 1
    t2 = st.sidebar.selectbox("Takım ", t2_candidates, index=min(default_idx, len(t2_candidates) - 1), key="t2_same")
    p2_list = df_mg[df_mg["Takim"] == t2]["Player"].tolist()
    o2_name = st.sidebar.selectbox("Oyuncu ", p2_list, key="o2_same")

# --- Satırları çek ---
p1 = df[df["Player"] == o1_name].iloc[0]
p2 = df[df["Player"] == o2_name].iloc[0]

# --- Kart başlıkları (opsiyonel iyileştirme) ---
label1 = "🔵 Hücum Oyuncusu" if mode.startswith("Serbest") else "🔵 Oyuncu 1"
label2 = "🔴 Rakip Oyuncu" if mode.startswith("Serbest") else "🔴 Oyuncu 2 (Aynı Mevki)"

# --- OYUNCU KARTLARI ---
c1, c2, c3 = st.columns([1, 0.2, 1])

with c1:
    st.info(f"{label1}: {p1['Player']}")
    st.caption(f"{p1['Takim']} | {p1['Pos_Simple']} | {p1['Rol']}")
    col_a, col_b = st.columns(2)
    col_a.metric("Hücum Skoru", f"{p1['Hucum_Skoru']:.3f}")
    col_b.metric("Overall Skor", f"{p1['Overall_Skoru']:.3f}")
    col_c, col_d = st.columns(2)
    col_c.metric("Oyun Kurucu Skoru", f"{p1['OyunKurucu_Skoru']:.3f}")
    col_d.metric("G+A / 90", f"{p1['G+A_p90']:.2f}")

with c2:
    st.markdown("<h2 style='text-align: center; margin-top: 20px;'>VS</h2>", unsafe_allow_html=True)

with c3:
    st.error(f"{label2}: {p2['Player']}")
    st.caption(f"{p2['Takim']} | {p2['Pos_Simple']} | {p2['Rol']}")
    col_a, col_b = st.columns(2)
    col_a.metric("Defans Skoru", f"{p2['Defans_Skoru']:.3f}")
    col_b.metric("Overall Skor", f"{p2['Overall_Skoru']:.3f}")
    col_c, col_d = st.columns(2)
    col_c.metric("Oyun Kurucu Skoru", f"{p2['OyunKurucu_Skoru']:.3f}")
    col_d.metric("G+A / 90", f"{p2['G+A_p90']:.2f}")

# --- ANALİZ BUTONU ---
if st.button("🔥 EŞLEŞMEYİ ANALİZ ET"):
    st.divider()

    # ANN tahmini (iki oyuncu için ayrı)
    x1 = prepare_features(p1, feature_names, mean_, scale_)
    x2 = prepare_features(p2, feature_names, mean_, scale_)
    y1_pred = float(model.predict(x1, verbose=0)[0, 0])
    y2_pred = float(model.predict(x2, verbose=0)[0, 0])

    # Basit avantaj skoru (p1 lehine)
    diff = y1_pred - y2_pred
    advantage = 50 + diff * 40
    advantage = max(0.0, min(100.0, advantage))

    st.subheader("🧠 Yapay Zeka Kararı")
    st.progress(int(advantage))

    col_res1, col_res2 = st.columns([3, 1])

    with col_res1:
        if advantage > 55:
            st.success(f"🔥 **{p1['Player']} daha avantajlı görünüyor!** (%{advantage:.1f})")
            st.write("Model, 1. oyuncunun beklenen gol+asist katkısını rakibine göre daha yüksek tahmin ediyor.")
        elif advantage < 45:
            st.error(f"🧱 **{p2['Player']} daha güçlü görünüyor!** (%{100-advantage:.1f})")
            st.write("Model, 2. oyuncunun katkısını daha yüksek görüyor; eşleşmede rakibin öne çıkma ihtimali daha fazla.")
        else:
            st.warning(f"⚖️ **Çok dengeli bir eşleşme!** (%{advantage:.1f})")
            st.write("Beklenen katkılar yakın; maç günü formu ve taktik belirleyici olur.")

        st.markdown(
            f"- {p1['Player']} tahmini G+A_p90: **{y1_pred:.2f}** (gerçek: {p1['G+A_p90']:.2f})  \n"
            f"- {p2['Player']} tahmini G+A_p90: **{y2_pred:.2f}** (gerçek: {p2['G+A_p90']:.2f})"
        )

        if selected_mg is not None:
            st.info(f"Bu kıyas **MevkiGroup = {selected_mg}** içinde yapılıyor.")

    with col_res2:
        # Radar / Polar Grafik: 4 ana skor
        categories = ["Defans", "Oyun Kurucu", "Hücum", "Overall"]
        v1 = [p1["Defans_Skoru"], p1["OyunKurucu_Skoru"], p1["Hucum_Skoru"], p1["Overall_Skoru"]]
        v2 = [p2["Defans_Skoru"], p2["OyunKurucu_Skoru"], p2["Hucum_Skoru"], p2["Overall_Skoru"]]

        v1 += v1[:1]
        v2 += v2[:1]
        cats_closed = categories + categories[:1]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=v1, theta=cats_closed, fill="toself", name=p1["Player"]))
        fig.add_trace(go.Scatterpolar(r=v2, theta=cats_closed, fill="toself", name=p2["Player"]))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)
