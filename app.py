import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
import plotly.graph_objects as go
import os

# ---------- SAYFA AYARLARI ----------
st.set_page_config(page_title="Futbol AI Scout Pro", page_icon="⚽", layout="wide")

st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        background-color: #00CC66;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Futbol AI Scout & Matchup Analisti")
st.markdown("**Gelişmiş Yapay Zeka Destekli Oyuncu Karşılaştırma ve Scouting Sistemi**")

# ---------- DOSYA YOLLARI (PATHS) ----------
# Bu yolların senin proje klasörünle birebir aynı olduğundan emin ol
SUPERLIG_FILE = "data/processed/superlig_scored_benchmark.csv"
TOP5_FILE     = "data/processed/top5_players_scored.csv"
BENCH_FILE    = "data/processed/top5_benchmarks_percentiles.csv"

# Model dosya isimlerini düzelttik
MODEL_FILE = "models/futbol_ann_ga.h5"
SCALER_MEAN_FILE = "models/scaler_mean.npy"
SCALER_SCALE_FILE = "models/scaler_scale.npy"
SCALER_FEATS_FILE = "models/feature_names.txt"

# ---------- YÜKLEME FONKSİYONLARI ----------
@st.cache_resource
def load_everything():
    # Veri setlerini yükle
    if not os.path.exists(SUPERLIG_FILE):
        st.error(f"Dosya bulunamadı: {SUPERLIG_FILE}. Lütfen veri işleme adımlarını tamamlayın.")
        return None, None, None, None, None, None, None

    sl = pd.read_csv(SUPERLIG_FILE)
    
    # Top5 dosyası yoksa boş bir DataFrame oluştur (Hata vermemesi için)
    if os.path.exists(TOP5_FILE):
        t5 = pd.read_csv(TOP5_FILE)
    else:
        t5 = pd.DataFrame() 

    if os.path.exists(BENCH_FILE):
        bench = pd.read_csv(BENCH_FILE)
    else:
        bench = pd.DataFrame()

    # Modeli ve Scaler'ı yükle
    if not os.path.exists(MODEL_FILE):
        st.error("Model dosyası bulunamadı. Lütfen önce '05_train_ann.py' dosyasını çalıştırın.")
        return None, None, None, None, None, None, None

    model = keras.models.load_model(MODEL_FILE, compile=False)
    data_mean = np.load(SCALER_MEAN_FILE)
    data_scale = np.load(SCALER_SCALE_FILE)
    
    with open(SCALER_FEATS_FILE, "r", encoding="utf-8") as f:
        feats = [line.strip() for line in f.readlines()]

    # UI uyumluluğu (Eski koddan kalan temizlikler)
    for df in [sl, t5]:
        if not df.empty:
            if "Takim" not in df.columns and "Team" in df.columns:
                df["Takim"] = df["Team"]
            if "Pos_Simple" not in df.columns and "Pos" in df.columns:
                df["Pos_Simple"] = df["Pos"].astype(str).str.split(",").str[0]
            if "Rol" not in df.columns:
                df["Rol"] = df.get("FM_Rol_Base", "")

    return sl, t5, bench, model, feats, data_mean, data_scale

# Verileri Yükle
loaded_data = load_everything()
if loaded_data[0] is None:
    st.stop() # Veri yoksa durdur

sl, t5, bench, model, feats, data_mean, data_scale = loaded_data

# ---------- YARDIMCI FONKSİYONLAR ----------

def standard_transform(X: np.ndarray) -> np.ndarray:
    # StandardScaler Formülü: (X - mean) / scale
    # Veri eksikse (NaN) 0 kabul ediyoruz
    X = np.nan_to_num(X, nan=0.0)
    return (X - data_mean) / data_scale

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,) vektör (bizim oyuncu)
    # b: (n,d) matris (diğer oyuncular)
    a = a.reshape(1, -1)
    # Norm hesapla (sıfıra bölme hatasını önlemek için +1e-9 ekle)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    
    # Cosine Similarity = (A . B) / (|A| * |B|)
    dot_product = a @ b.T
    similarity = dot_product / (a_norm @ b_norm.T)
    return similarity.ravel()

def create_radar_chart(p1, p2, label1, label2, comparison_type="Genel"):
    categories = ["Defans", "Oyun Kurucu", "Hücum", "Overall"]
    
    # Verileri 0-1 arasına normalize etmek yerine direkt skorları kullanıyoruz (Zaten 0-10 aralığında scale edilmişti önceki adımlarda)
    # Görselleştirmek için 10 üzerinden normalizasyon varsayıyoruz.
    
    v1 = [p1.get("Defans_Skoru",0), p1.get("OyunKurucu_Skoru",0), p1.get("Hucum_Skoru",0), p1.get("Overall_Skoru",0)]
    v2 = [p2.get("Defans_Skoru",0), p2.get("OyunKurucu_Skoru",0), p2.get("Hucum_Skoru",0), p2.get("Overall_Skoru",0)]

    # Radar kapanması için başa dön
    v1 += v1[:1]
    v2 += v2[:1]
    cats = categories + categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=v1, theta=cats, fill="toself", name=label1, line_color='blue'))
    fig.add_trace(go.Scatterpolar(r=v2, theta=cats, fill="toself", name=label2, line_color='red'))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])), # Skorlar genelde 0-10 arası
        showlegend=True,
        title=f"{comparison_type} Karşılaştırması"
    )
    return fig

# ---------- SEKMELER ----------
tab1, tab2 = st.tabs(["⚔️ Matchup & Kıyaslama", "🌍 Scouting & Benzer Oyuncular"])

# ==========================
# TAB 1: MATCHUP VE KIYASLAMA
# ==========================
with tab1:
    st.sidebar.header("Oyuncu Seçimi")
    
    # KULLANICI İSTEĞİ: Mod Seçimi
    mode = st.sidebar.radio(
        "Analiz Modu Seçin:",
        ["Aynı Mevki Kıyaslama (Örn: Stoper vs Stoper)", "Matchup Analizi (Hücumcu vs Savunmacı)"]
    )
    
    takimlar = sorted(sl["Takim"].astype(str).unique())
    
    # --- OYUNCU SEÇİM KUTULARI ---
    st.sidebar.subheader("1. Oyuncu (Bizim Takım)")
    t1 = st.sidebar.selectbox("Takım 1", takimlar, index=0, key="t1")
    p1_list = sl[sl["Takim"] == t1]["Player"].tolist()
    p1_name = st.sidebar.selectbox("Oyuncu 1", p1_list, key="p1")
    
    st.sidebar.subheader("2. Oyuncu (Rakip/Kıyas)")
    # Varsayılan olarak farklı bir takım seçilsin
    def_idx = 1 if len(takimlar) > 1 else 0
    t2 = st.sidebar.selectbox("Takım 2", takimlar, index=def_idx, key="t2")
    p2_list = sl[sl["Takim"] == t2]["Player"].tolist()
    p2_name = st.sidebar.selectbox("Oyuncu 2", p2_list, key="p2")

    # Seçilen oyuncuların verilerini al
    player1 = sl[sl["Player"] == p1_name].iloc[0]
    player2 = sl[sl["Player"] == p2_name].iloc[0]

    # --- ANALİZ BUTONU ---
    if st.button("Analizi Başlat", type="primary"):
        st.divider()
        
        col1, col2, col3 = st.columns([1, 0.2, 1])
        
        # OYUNCU 1 KART
        with col1:
            st.subheader(f"🔵 {player1['Player']}")
            st.caption(f"{player1['Takim']} | {player1['Pos_Simple']}")
            st.metric("Overall Skor", f"{player1['Overall_Skoru']:.1f}")
            st.metric("Hücum Skoru", f"{player1['Hucum_Skoru']:.1f}")
            st.metric("Defans Skoru", f"{player1['Defans_Skoru']:.1f}")
            
        with col2:
            st.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
            
        # OYUNCU 2 KART
        with col3:
            st.subheader(f"🔴 {player2['Player']}")
            st.caption(f"{player2['Takim']} | {player2['Pos_Simple']}")
            st.metric("Overall Skor", f"{player2['Overall_Skoru']:.1f}")
            st.metric("Hücum Skoru", f"{player2['Hucum_Skoru']:.1f}")
            st.metric("Defans Skoru", f"{player2['Defans_Skoru']:.1f}")

        st.divider()
        
        # --- MANTIKSAL ANALİZ KISMI (KULLANICI İSTEĞİ) ---
        
        # 1. MODEL TAHMİNİ (G+A Potansiyeli - Data Leakage Olmadan)
        # Veriyi hazırla
        vec1 = np.array([float(player1.get(f, 0.0)) for f in feats]).reshape(1, -1)
        vec2 = np.array([float(player2.get(f, 0.0)) for f in feats]).reshape(1, -1)
        
        # Scale et (StandardScaler ile)
        vec1_scaled = standard_transform(vec1)
        vec2_scaled = standard_transform(vec2)
        
        # Tahmin
        pred1 = float(model.predict(vec1_scaled, verbose=0)[0, 0])
        pred2 = float(model.predict(vec2_scaled, verbose=0)[0, 0])

        # 2. SENARYOYA GÖRE DEĞERLENDİRME
        st.subheader("📝 Yapay Zeka Analiz Raporu")
        
        if mode == "Aynı Mevki Kıyaslama (Örn: Stoper vs Stoper)":
            # AYNI MEVKİ: Direkt Overall veya mevkiye özgü puana bakılır
            st.info("Bu modda oyuncuların kendi mevkilerindeki genel performansları kıyaslanıyor.")
            
            score1 = player1["Overall_Skoru"]
            score2 = player2["Overall_Skoru"]
            
            diff = score1 - score2
            if diff > 0.5:
                st.success(f"**{player1['Player']}** daha komple bir oyuncu profili çiziyor.")
            elif diff < -0.5:
                st.error(f"**{player2['Player']}** bu mevkide daha üstün istatistiklere sahip.")
            else:
                st.warning("İki oyuncu da çok benzer seviyede.")
                
            # Radar Grafiği (Genel)
            fig = create_radar_chart(player1, player2, player1['Player'], player2['Player'], "Mevki Performans")
            st.plotly_chart(fig, use_container_width=True)

        else:
            # MATCHUP: Hücum vs Defans
            st.info("Bu modda Oyuncu 1'in Hücum gücü ile Oyuncu 2'nin Savunma gücü çarpıştırılıyor.")
            
            # Matchup Skoru Hesaplama
            # P1 Hücum - P2 Defans
            att_power = player1["Hucum_Skoru"]
            def_power = player2["Defans_Skoru"]
            
            matchup_diff = att_power - def_power
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.write(f"🔵 {player1['Player']} Hücum Gücü: **{att_power:.1f}**")
                st.write(f"🔴 {player2['Player']} Defans Gücü: **{def_power:.1f}**")
                
                if matchup_diff > 1.5:
                    st.success(f"🔥 **{player1['Player']}** bu eşleşmede rakibine büyük üstünlük kurabilir!")
                elif matchup_diff < -1.5:
                    st.error(f"🧱 **{player2['Player']}** savunmada duvar örüyor, geçmek çok zor.")
                else:
                    st.warning("⚖️ **Dengeli bir eşleşme.** Günlük form belirleyici olur.")

                st.markdown("---")
                st.caption(f"Yapay Zeka Tahmini (Ofansif Katkı Potansiyeli):")
                st.caption(f"{player1['Player']}: {pred1:.2f} G+A/90 Beklentisi")
                
            with col_res2:
                 # Matchup Bar
                 fig_bar = go.Figure()
                 fig_bar.add_trace(go.Bar(
                     x=[att_power, def_power],
                     y=[f"{player1['Player']} (Hücum)", f"{player2['Player']} (Defans)"],
                     orientation='h',
                     marker_color=['blue', 'red']
                 ))
                 fig_bar.update_layout(title="Matchup Güç Dengesi", xaxis_range=[0, 10])
                 st.plotly_chart(fig_bar, use_container_width=True)

# ==========================
# TAB 2: SCOUTING (BENZER OYUNCU)
# ==========================
with tab2:
    st.header("🌍 Global Scouting Ağı (Top 5 Lig)")
    st.markdown("Süper Lig'deki bir oyuncunun Avrupa'nın 5 büyük ligindeki istatistiksel ikizlerini bulun.")
    
    # Süper Lig'den oyuncu seç (Tab 1'den bağımsız olabilir)
    col_scout1, col_scout2 = st.columns([1, 2])
    
    with col_scout1:
        team_scout = st.selectbox("Takım Seç", takimlar, key="scout_team")
        p_scout_list = sl[sl["Takim"] == team_scout]["Player"].tolist()
        player_scout_name = st.selectbox("Oyuncu Seç", p_scout_list, key="scout_player")
        
        target_player = sl[sl["Player"] == player_scout_name].iloc[0]
        
        st.markdown("### Seçilen Oyuncu")
        st.write(f"**{target_player['Player']}**")
        st.write(f"Mevki: {target_player['Pos_Simple']}")
        st.metric("Süper Lig Overall", f"{target_player['Overall_Skoru']:.1f}")

    with col_scout2:
        if t5.empty:
            st.warning("Top 5 lig verisi (top5_players_scored.csv) bulunamadı.")
        else:
            if st.button("🔎 Benzer Oyuncuları Tara"):
                with st.spinner("Avrupa veritabanı taranıyor..."):
                    # 1. Seçilen oyuncunun feature vektörünü çıkar
                    target_vec = np.array([float(target_player.get(f, 0.0)) for f in feats])
                    
                    # 2. Sadece AYNI MEVKİDEKİ oyuncularla kıyasla (Daha doğru sonuç için)
                    target_pos_group = str(target_player["MevkiGroup"])
                    t5_filtered = t5[t5["MevkiGroup"].astype(str) == target_pos_group].copy()
                    
                    if t5_filtered.empty:
                        st.warning("Avrupa'da bu mevkide oyuncu bulunamadı, tüm veritabanı taranıyor.")
                        t5_filtered = t5.copy()
                    
                    # 3. Avrupa'daki oyuncuların feature matrisini hazırla
                    # (Burada yavaşlık olmaması için normalde önceden hesaplanmalı ama şimdilik anlık yapıyoruz)
                    t5_matrix = []
                    valid_indices = []
                    
                    for idx, row in t5_filtered.iterrows():
                        vec = [float(row.get(f, 0.0)) for f in feats]
                        t5_matrix.append(vec)
                        valid_indices.append(idx)
                        
                    t5_matrix = np.array(t5_matrix)
                    
                    # 4. Cosine Similarity Hesapla
                    scores = cosine_sim(target_vec, t5_matrix)
                    
                    # 5. Sonuçları DataFrame'e ekle
                    t5_filtered["Similarity"] = scores * 100 # Yüzde cinsinden
                    
                    # En benzer 10 oyuncuyu getir
                    top_matches = t5_filtered.sort_values("Similarity", ascending=False).head(10)
                    
                    st.success("Tarama Tamamlandı!")
                    st.dataframe(
                        top_matches[["Player", "Team", "League", "Age", "Similarity", "Overall_Skoru"]].style.format({"Similarity": "{:.1f}%", "Overall_Skoru": "{:.1f}"}),
                        use_container_width=True
                    )
                    
                    # En benzer oyuncu ile Radar kıyaslaması
                    best_match = top_matches.iloc[0]
                    st.subheader(f"En Yakın Eşleşme: {best_match['Player']} ({best_match['Team']})")
                    
                    fig_radar = create_radar_chart(target_player, best_match, target_player['Player'], best_match['Player'], "Scouting")
                    st.plotly_chart(fig_radar, use_container_width=True)