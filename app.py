import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import plotly.graph_objects as go

# ---------- SAYFA ----------
st.set_page_config(page_title="Futbol AI Scout", page_icon="⚽", layout="wide")

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
st.markdown("**2025-2026 Süper Lig: Hücum vs Savunma Analizi + Top-5 Benchmark/Scout**")

# ---------- PATHS ----------
SUPERLIG_FILE = "data/processed/superlig_scored_benchmark.csv"
TOP5_FILE     = "data/processed/top5_players_scored.csv"
BENCH_FILE    = "data/processed/top5_benchmarks_percentiles.csv"

MODEL_FILE = "models/futbol_ann_ga.h5"
SCALER_MIN_FILE = "models/combined_minmax_min.npy"
SCALER_MAX_FILE = "models/combined_minmax_max.npy"
SCALER_FEATS_FILE = "models/combined_minmax_features.txt"

# ---------- LOAD ----------
@st.cache_resource
def load_everything():
    sl = pd.read_csv(SUPERLIG_FILE)
    t5 = pd.read_csv(TOP5_FILE)
    bench = pd.read_csv(BENCH_FILE)

    model = keras.models.load_model(MODEL_FILE, compile=False)

    data_min = np.load(SCALER_MIN_FILE)
    data_max = np.load(SCALER_MAX_FILE)
    with open(SCALER_FEATS_FILE, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f.readlines()]

    # eski UI uyumluluğu
    if "Takim" not in sl.columns and "Team" in sl.columns:
        sl["Takim"] = sl["Team"]
    if "Takim" not in t5.columns and "Team" in t5.columns:
        t5["Takim"] = t5["Team"]

    if "Pos_Simple" not in sl.columns:
        sl["Pos_Simple"] = sl["Pos"].astype(str).str.split(",").str[0]
    if "Pos_Simple" not in t5.columns:
        t5["Pos_Simple"] = t5["Pos"].astype(str).str.split(",").str[0]

    if "Rol" not in sl.columns:
        sl["Rol"] = sl.get("FM_Rol_Base", "")

    return sl, t5, bench, model, feats, data_min, data_max

sl, t5, bench, model, feats, data_min, data_max = load_everything()

def minmax_transform(X: np.ndarray) -> np.ndarray:
    # X: (n, d)
    denom = (data_max - data_min)
    denom = np.where(denom == 0, 1.0, denom)
    return (X - data_min) / denom

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,), b: (n,d)
    a = a.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a_norm @ b_norm.T).ravel()

def radar_scores(p1: pd.Series, p2: pd.Series, title1: str, title2: str):
    categories = ["Defans", "Oyun Kurucu", "Hücum", "Overall"]
    v1 = [p1["Defans_Skoru"], p1["OyunKurucu_Skoru"], p1["Hucum_Skoru"], p1["Overall_Skoru"]]
    v2 = [p2["Defans_Skoru"], p2["OyunKurucu_Skoru"], p2["Hucum_Skoru"], p2["Overall_Skoru"]]

    v1 += v1[:1]
    v2 += v2[:1]
    cats = categories + categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=v1, theta=cats, fill="toself", name=title1))
    fig.add_trace(go.Scatterpolar(r=v2, theta=cats, fill="toself", name=title2))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

# ---------- TABS ----------
tab1, tab2 = st.tabs(["1) Matchup Analizi", "2) Top-5 Benchmark & Similarity Scout"])

# ==========================
# TAB 1: MATCHUP
# ==========================
with tab1:
    st.sidebar.header("🕵️‍♂️ Oyuncu Seçimi (Matchup)")

    mode = st.sidebar.radio(
        "🎛️ Karşılaştırma Modu",
        ["Serbest Matchup (Hücum vs Rakip)", "Aynı Mevki Kıyas (MevkiGroup)"],
        index=0,
        key="mode_tab1"
    )

    takimlar = sorted(sl["Takim"].astype(str).unique())

    if mode == "Serbest Matchup (Hücum vs Rakip)":
        st.sidebar.subheader("1. Oyuncu (Hücum)")
        t1 = st.sidebar.selectbox("Takım", takimlar, index=0, key="t1_free")
        o1_list = sl[sl["Takim"] == t1]["Player"].tolist()
        o1_name = st.sidebar.selectbox("Oyuncu", o1_list, key="o1_free")

        st.sidebar.subheader("2. Oyuncu (Rakip)")
        t2 = st.sidebar.selectbox("Takım ", takimlar, index=min(1, len(takimlar) - 1), key="t2_free")
        o2_list = sl[sl["Takim"] == t2]["Player"].tolist()
        if not o2_list:
            o2_list = sl["Player"].tolist()
        o2_name = st.sidebar.selectbox("Oyuncu ", o2_list, key="o2_free")

        selected_mg = None
    else:
        st.sidebar.subheader("Aynı Mevki Kıyas Ayarları")

        mg_list_pref = ["GK", "CB", "FB", "MID", "AM", "WING", "FWD"]
        available_mg = [m for m in mg_list_pref if m in sl["MevkiGroup"].astype(str).unique()]
        if not available_mg:
            available_mg = sorted(sl["MevkiGroup"].astype(str).unique().tolist())

        selected_mg = st.sidebar.selectbox("MevkiGroup", available_mg, key="mg_same")

        sl_mg = sl[sl["MevkiGroup"].astype(str) == str(selected_mg)].copy()

        st.sidebar.markdown("### 🔝 Top 10 (Overall)")
        top10 = (
            sl_mg.sort_values("Overall_Skoru", ascending=False)
            .loc[:, ["Player", "Takim", "Overall_Skoru", "Hucum_Skoru", "Defans_Skoru"]]
            .head(10)
        )
        st.sidebar.dataframe(top10, use_container_width=True, height=280)

        st.sidebar.subheader("1. Oyuncu")
        t1 = st.sidebar.selectbox("Takım", sorted(sl_mg["Takim"].astype(str).unique()), key="t1_same")
        p1_list = sl_mg[sl_mg["Takim"] == t1]["Player"].tolist()
        o1_name = st.sidebar.selectbox("Oyuncu", p1_list, key="o1_same")

        st.sidebar.subheader("2. Oyuncu")
        t2_candidates = sorted(sl_mg["Takim"].astype(str).unique())
        default_idx = 0 if len(t2_candidates) == 1 else 1
        t2 = st.sidebar.selectbox("Takım ", t2_candidates, index=min(default_idx, len(t2_candidates) - 1), key="t2_same")
        p2_list = sl_mg[sl_mg["Takim"] == t2]["Player"].tolist()
        o2_name = st.sidebar.selectbox("Oyuncu ", p2_list, key="o2_same")

    p1 = sl[sl["Player"] == o1_name].iloc[0]
    p2 = sl[sl["Player"] == o2_name].iloc[0]

    label1 = "🔵 Hücum Oyuncusu" if mode.startswith("Serbest") else "🔵 Oyuncu 1"
    label2 = "🔴 Rakip Oyuncu" if mode.startswith("Serbest") else "🔴 Oyuncu 2 (Aynı Mevki)"

    c1, c2, c3 = st.columns([1, 0.2, 1])

    with c1:
        st.info(f"{label1}: {p1['Player']}")
        st.caption(f"{p1['Takim']} | {p1['Pos_Simple']} | {p1.get('Rol','')}")
        col_a, col_b = st.columns(2)
        col_a.metric("Hücum Skoru", f"{p1['Hucum_Skoru']:.3f}")
        col_b.metric("Overall Skor", f"{p1['Overall_Skoru']:.3f}")

        col_c, col_d = st.columns(2)
        col_c.metric("Top-5 Overall %", f"{p1.get('Top5Percentile_Overall_Skoru', np.nan):.1f}")
        col_d.metric("G+A / 90", f"{p1['G+A_p90']:.2f}")

    with c2:
        st.markdown("<h2 style='text-align: center; margin-top: 20px;'>VS</h2>", unsafe_allow_html=True)

    with c3:
        st.error(f"{label2}: {p2['Player']}")
        st.caption(f"{p2['Takim']} | {p2['Pos_Simple']} | {p2.get('Rol','')}")
        col_a, col_b = st.columns(2)
        col_a.metric("Defans Skoru", f"{p2['Defans_Skoru']:.3f}")
        col_b.metric("Overall Skor", f"{p2['Overall_Skoru']:.3f}")

        col_c, col_d = st.columns(2)
        col_c.metric("Top-5 Overall %", f"{p2.get('Top5Percentile_Overall_Skoru', np.nan):.1f}")
        col_d.metric("G+A / 90", f"{p2['G+A_p90']:.2f}")

    if st.button("🔥 EŞLEŞMEYİ ANALİZ ET", key="btn_analyze_tab1"):
        st.divider()

        # ANN input: scaler/feature list ile hazırlanır
        x1 = np.array([float(p1.get(f, 0.0)) for f in feats], dtype=float).reshape(1, -1)
        x2 = np.array([float(p2.get(f, 0.0)) for f in feats], dtype=float).reshape(1, -1)

        # ANN modeli bu projede tek vektör -> G+A tahmini olarak eğitildi varsayımıyla
        y1 = float(model.predict(x1, verbose=0)[0, 0])
        y2 = float(model.predict(x2, verbose=0)[0, 0])

        diff = y1 - y2
        advantage = 50 + diff * 40
        advantage = max(0.0, min(100.0, advantage))

        st.subheader("🧠 Yapay Zeka Kararı")
        st.progress(int(advantage))

        col_res1, col_res2 = st.columns([3, 1])

        with col_res1:
            if advantage > 55:
                st.success(f"🔥 **{p1['Player']} daha avantajlı görünüyor!** (%{advantage:.1f})")
            elif advantage < 45:
                st.error(f"🧱 **{p2['Player']} daha güçlü görünüyor!** (%{100-advantage:.1f})")
            else:
                st.warning(f"⚖️ **Çok dengeli!** (%{advantage:.1f})")

            st.markdown(
                f"- {p1['Player']} tahmini G+A_p90: **{y1:.2f}** (gerçek: {p1['G+A_p90']:.2f})  \n"
                f"- {p2['Player']} tahmini G+A_p90: **{y2:.2f}** (gerçek: {p2['G+A_p90']:.2f})"
            )

            if selected_mg is not None:
                st.info(f"Bu kıyas **MevkiGroup = {selected_mg}** içinde yapılıyor.")

        with col_res2:
            fig = radar_scores(p1, p2, p1["Player"], p2["Player"])
            st.plotly_chart(fig, use_container_width=True)

# ==========================
# TAB 2: BENCHMARK + SIMILARITY
# ==========================
with tab2:
    st.subheader("📊 Top-5 Benchmark & Similarity Scout")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Süper Lig oyuncusu seç")
        team_pick = st.selectbox("Takım", sorted(sl["Takim"].astype(str).unique()), key="team_tab2")
        p_list = sl[sl["Takim"] == team_pick]["Player"].tolist()
        player_pick = st.selectbox("Oyuncu", p_list, key="player_tab2")
        p = sl[sl["Player"] == player_pick].iloc[0]

        st.markdown("### Top-5 percentile (aynı MevkiGroup)")
        cols = st.columns(3)
        cols[0].metric("Overall %", f"{p.get('Top5Percentile_Overall_Skoru', np.nan):.1f}")
        cols[1].metric("Hücum %", f"{p.get('Top5Percentile_Hucum_Skoru', np.nan):.1f}")
        cols[2].metric("Defans %", f"{p.get('Top5Percentile_Defans_Skoru', np.nan):.1f}")

        cols2 = st.columns(2)
        cols2[0].metric("Oyun Kurucu %", f"{p.get('Top5Percentile_OyunKurucu_Skoru', np.nan):.1f}")
        cols2[1].metric("G+A_p90 %", f"{p.get('Top5Percentile_G+A_p90', np.nan):.1f}")

        st.caption(f"MevkiGroup: {p['MevkiGroup']} | Pozisyon: {p.get('Pos_Simple','')} | Rol: {p.get('Rol','')}")

    with right:
        st.markdown("### 🔎 Avrupa benzerleri (Top-5)")
        mg = str(p["MevkiGroup"])

        # Similarity için ortak feature uzayı: feats listesi (combined_minmax_features)
        # Not: Bu feats, MinMax fit edilirken kullanılan COMMON_METRICS'tir.
        p_vec = np.array([float(p.get(f, 0.0)) for f in feats], dtype=float)

        t5_sub = t5[t5["MevkiGroup"].astype(str) == mg].copy()
        if t5_sub.empty:
            st.warning("Bu MevkiGroup için Top-5 tarafında kayıt bulunamadı.")
        else:
            mat = np.stack([np.array([float(row.get(f, 0.0)) for f in feats], dtype=float) for _, row in t5_sub.iterrows()])
            sims = cosine_sim(p_vec, mat)
            t5_sub["Similarity"] = sims

            out = t5_sub.sort_values("Similarity", ascending=False).head(10)
            st.dataframe(
                out[["Player", "Team", "League", "Pos", "Overall_Skoru", "G+A_p90", "Similarity"]],
                use_container_width=True,
                height=380
            )

            # Radar: ilk benzer oyuncu ile
            best = out.iloc[0]
            st.markdown(f"#### Radar: {p['Player']} vs {best['Player']}")
            fig2 = radar_scores(p, best, p["Player"], best["Player"])
            st.plotly_chart(fig2, use_container_width=True)