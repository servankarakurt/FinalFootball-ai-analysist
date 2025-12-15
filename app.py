import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from streamlit_option_menu import option_menu 

# ---------- SAYFA AYARLARI ----------
st.set_page_config(page_title="ScoutPro AI", page_icon="⚽", layout="wide")

# Modern CSS Tasarımı
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #1d3557; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #457b9d;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1d3557;
    }
    .top5-header {
        background-color: #e63946;
        color: white;
        padding: 10px;
        border-radius: 5px 5px 0 0;
        text-align: center;
        font-weight: bold;
    }
    .ai-report-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1565c0;
        padding: 20px;
        border-radius: 8px;
        font-family: 'Verdana', sans-serif;
        font-size: 15px;
        color: #0d47a1;
        line-height: 1.6;
    }
    .vs-badge {
        background-color: #d32f2f;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        margin-right: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- DOSYA YOLLARI ----------
SUPERLIG_FILE = "data/processed/superlig_scored_benchmark.csv"
TOP5_FILE     = "data/processed/top5_players_scored.csv"
PIPELINE_FILE = "models/futbol_pipeline.pkl" 

# ---------- YÜKLEME FONKSİYONLARI ----------
@st.cache_resource
def load_data_and_model():
    if not os.path.exists(SUPERLIG_FILE):
        st.error(f"Veri dosyası bulunamadı: {SUPERLIG_FILE}")
        return None, None, None, None

    sl = pd.read_csv(SUPERLIG_FILE)
    
    if os.path.exists(TOP5_FILE):
        t5 = pd.read_csv(TOP5_FILE)
    else:
        t5 = pd.DataFrame()

    if not os.path.exists(PIPELINE_FILE):
        pipeline = None
        feats = []
    else:
        try:
            pipeline = joblib.load(PIPELINE_FILE)
            feats = getattr(pipeline, "required_features", [])
        except:
            pipeline = None
            feats = []

    # UI Uyumluluğu
    for df in [sl, t5]:
        if not df.empty:
            if "Takim" not in df.columns and "Team" in df.columns:
                df["Takim"] = df["Team"]
            if "Pos_Simple" not in df.columns and "Pos" in df.columns:
                df["Pos_Simple"] = df["Pos"].astype(str).str.split(",").str[0]
            if "Player" in df.columns:
                df["TM_Link"] = "https://www.transfermarkt.com.tr/schnellsuche/ergebnis/schnellsuche?query=" + df["Player"].astype(str).str.replace(" ", "+")

    return sl, t5, pipeline, feats

loaded = load_data_and_model()
if loaded[0] is None:
    st.stop()

sl, t5, pipeline, feats = loaded

# ---------- YARDIMCI FONKSİYONLAR ----------
def get_prediction(player_row):
    if pipeline is None: return 0.0
    try:
        input_data = {f: player_row.get(f, 0.0) for f in feats}
        input_df = pd.DataFrame([input_data])
        return float(pipeline.predict(input_df)[0])
    except:
        return 0.0

def cosine_sim(a, b):
    a = a.reshape(1, -1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-9
    result = (a @ b.T) / (a_norm @ b_norm.T)
    return result.flatten()

def create_radar(p1, p2, label1, label2, title):
    cats = ["Defans", "Pas", "Fizik", "Teknik", "Hücum"]
    
    def get_val(p, c):
        key_map = {"Defans": "Defans_Skoru", "Pas": "Pas_Skoru", "Fizik": "Fizik_Skoru", "Teknik": "Teknik_Skoru", "Hücum": "Hucum_Skoru"}
        val = p.get(key_map[c], 0.0)
        return val

    v1 = [get_val(p1, c) for c in cats]
    v2 = [get_val(p2, c) for c in cats]

    v1 += v1[:1]; v2 += v2[:1]; cats += cats[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=label1, line_color='#1d3557'))
    fig.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name=label2, line_color='#e63946'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])), 
        title=dict(text=title, x=0.5),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

# --- 🧠 YENİ: VS MODU KIYASLAMA MOTORU ---
def generate_comparison_report(p1, p2, mode):
    name1, name2 = p1['Player'], p2['Player']
    team1, team2 = p1['Team'], p2['Team']
    
    # Skorları Al
    def get_s(p, k): return float(p.get(k, 0))
    
    phy1, phy2 = get_s(p1, 'Fizik_Skoru'), get_s(p2, 'Fizik_Skoru')
    att1, att2 = get_s(p1, 'Hucum_Skoru'), get_s(p2, 'Hucum_Skoru')
    def1, def2 = get_s(p1, 'Defans_Skoru'), get_s(p2, 'Defans_Skoru')
    tech1, tech2 = get_s(p1, 'Teknik_Skoru'), get_s(p2, 'Teknik_Skoru')
    pas1, pas2 = get_s(p1, 'Pas_Skoru'), get_s(p2, 'Pas_Skoru')
    ovr1, ovr2 = get_s(p1, 'Overall_Skoru'), get_s(p2, 'Overall_Skoru')

    report = f"🤖 **ScoutPro VS Analizi: {name1} vs {name2}**\n\n"
    
    # 1. FİZİKSEL MÜCADELE (En Önemli Kısım)
    diff_phy = phy1 - phy2
    if abs(diff_phy) > 1.5:
        better = name1 if diff_phy > 0 else name2
        worse = name2 if diff_phy > 0 else name1
        report += f"💪 **Fiziksel Hakimiyet:** {better} ({max(phy1, phy2):.1f}), {worse} karşısında ({min(phy1, phy2):.1f}) bariz bir fiziksel üstünlüğe sahip. İkili mücadelelerde rakibini ezecektir.\n"
    elif abs(diff_phy) > 0.5:
        better = name1 if diff_phy > 0 else name2
        report += f"⚖️ **Fizik:** {better} fiziksel olarak bir adım önde. Kıran kırana bir mücadele olacak.\n"
    else:
        report += "🤝 **Fizik:** İki oyuncu da fiziksel olarak birbirine denk. Maçın kaderini teknik detaylar belirleyecek.\n"

    # 2. SENARYO ANALİZİ (Moda Göre Değişir)
    if mode == "Hücum vs Defans":
        # Biri Hücumcu, Biri Defansçı varsayımı
        # Kimin ne olduğunu tahmin edelim (Basitçe MevkiGroup'tan veya skorlardan)
        is_p1_att = att1 > def1
        attacker = name1 if is_p1_att else name2
        defender = name2 if is_p1_att else name1
        att_score = att1 if is_p1_att else att2
        def_score = def2 if is_p1_att else def1
        
        diff_matchup = att_score - def_score
        
        if diff_matchup > 1.0:
            report += f"🔥 **Kritik Eşleşme:** {attacker}, hücum yetenekleriyle ({att_score:.1f}), {defender}'in savunma hattını ({def_score:.1f}) delip geçebilir. Savunmanın ekstra yardıma ihtiyacı olacak.\n"
        elif diff_matchup < -1.0:
            report += f"🧱 **Kritik Eşleşme:** {defender}, savunma disipliniyle ({def_score:.1f}), {attacker}'i sahadan silebilir. Hücum oyuncusu için zor bir maç olacak.\n"
        else:
            report += f"⚔️ **Kritik Eşleşme:** {attacker} ile {defender} arasındaki düello nefes kesecek. Anlık hatalar sonucu belirler.\n"

    else: # Aynı Mevki Kıyaslama (Örn: İki Forvet veya İki Stoper)
        # Teknik ve Pas Farkı
        diff_tech = tech1 - tech2
        if abs(diff_tech) > 1.5:
            better = name1 if diff_tech > 0 else name2
            report += f"🎨 **Teknik Kapasite:** {better}, top tekniği ve yaratıcılık konusunda çok daha yetenekli.\n"
        
        # Oyun Zekası (Pas Skoru üzerinden yorum)
        if max(pas1, pas2) > 7.5:
            better = name1 if pas1 > pas2 else name2
            report += f"🧠 **Oyun Aklı:** {better}, sahadaki duruşu ve pas dağıtımıyla takımını yöneten isim.\n"

    # 3. GENEL SONUÇ (Overall Kıyaslama)
    diff_ovr = ovr1 - ovr2
    if diff_ovr > 1.0:
        report += f"\n🏆 **Sonuç:** **{name1}**, rakibine göre çok daha komple bir oyuncu. Bu eşleşmenin favorisi net bir şekilde o."
    elif diff_ovr < -1.0:
        report += f"\n🏆 **Sonuç:** **{name2}**, rakibine göre çok daha komple bir oyuncu. Bu eşleşmenin favorisi net bir şekilde o."
    else:
        report += f"\n⚖️ **Sonuç:** İki oyuncu da birbirine çok yakın seviyede. Günlük form durumları belirleyici olur."

    return report

# ---------- SOL MENÜ ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/53/53283.png", width=80) 
    st.title("ScoutPro AI")
    
    selected = option_menu(
        menu_title="Menü",
        options=["Matchup Analizi", "Global Scouting", "Hakkında"],
        icons=["swords", "globe", "info-circle"], 
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#457b9d"},
        }
    )
    st.markdown("---")
    st.info("💡 **İpucu:** Scouting bölümünde Transfermarkt linklerine tıklayarak oyuncu profillerini inceleyebilirsiniz.")

# ========================================================
# SAYFA 1: MATCHUP ANALİZİ
# ========================================================
if selected == "Matchup Analizi":
    st.subheader("⚔️ Oyuncu Karşılaştırma & Matchup")
    
    col_settings, col_visual = st.columns([1, 2])
    
    with col_settings:
        st.markdown("### 🛠️ Ayarlar")
        mode = st.radio("Analiz Tipi:", ["Aynı Mevki Kıyaslama", "Hücum vs Defans"], horizontal=True)
        
        selected_pos_filter = "Tümü"
        if mode == "Aynı Mevki Kıyaslama":
            st.markdown("---")
            if "MevkiGroup" in sl.columns:
                pos_options = sorted(sl["MevkiGroup"].astype(str).unique())
                selected_pos_filter = st.selectbox("📌 Hangi Mevki?", ["Tümü"] + pos_options)

        if selected_pos_filter != "Tümü":
            sl_filtered = sl[sl["MevkiGroup"] == selected_pos_filter]
        else:
            sl_filtered = sl

        teams = sorted(sl_filtered["Takim"].astype(str).unique())
        
        st.markdown("**1. Oyuncu**")
        t1 = st.selectbox("Takım", teams, key="t1")
        p1_list = sl_filtered[sl_filtered["Takim"] == t1]["Player"].tolist()
        p1_name = st.selectbox("Oyuncu", p1_list, key="p1")
        
        st.markdown("**2. Oyuncu**")
        t2 = st.selectbox("Takım", teams, index=min(1, len(teams)-1), key="t2")
        p2_list = sl_filtered[sl_filtered["Takim"] == t2]["Player"].tolist() or ["Veri Yok"]
        p2_name = st.selectbox("Oyuncu", p2_list, key="p2")
        
        analyze_btn = st.button("🔥 Analizi Başlat")

    with col_visual:
        if mode == "Aynı Mevki Kıyaslama" and selected_pos_filter != "Tümü":
            st.markdown(f"<div class='top5-header'>🏆 Süper Lig - En İyi 5 {selected_pos_filter} (Overall)</div>", unsafe_allow_html=True)
            top5_df = sl_filtered.sort_values("Overall_Skoru", ascending=False).head(5)
            st.dataframe(
                top5_df[["Player", "Team", "Overall_Skoru", "TM_Link"]],
                column_config={
                    "Overall_Skoru": st.column_config.ProgressColumn("Puan", format="%.1f", min_value=0, max_value=10),
                    "TM_Link": st.column_config.LinkColumn("Profil", display_text="🔗")
                },
                hide_index=True, use_container_width=True
            )
            st.markdown("---")

        if analyze_btn and p1_name and p2_name:
            p1 = sl[sl["Player"] == p1_name].iloc[0]
            p2 = sl[sl["Player"] == p2_name].iloc[0]
            
            c1, c2, c3 = st.columns([1, 0.2, 1])
            with c1:
                st.metric(label=p1['Player'], value=f"{p1['Overall_Skoru']:.1f}", delta="Overall")
                st.caption(f"{p1['Takim']} | {p1['Pos_Simple']}")
            with c2:
                st.markdown("<h2 style='text-align: center; color: #e63946;'>VS</h2>", unsafe_allow_html=True)
            with c3:
                st.metric(label=p2['Player'], value=f"{p2['Overall_Skoru']:.1f}", delta="Overall")
                st.caption(f"{p2['Takim']} | {p2['Pos_Simple']}")
            
            st.plotly_chart(create_radar(p1, p2, p1['Player'], p2['Player'], mode), use_container_width=True)
            
            # --- YAPAY ZEKA VS RAPORU ---
            st.markdown("### 📝 Yapay Zeka Karşılaştırma Raporu")
            comparison_text = generate_comparison_report(p1, p2, mode)
            st.markdown(f"<div class='ai-report-box'>{comparison_text}</div>", unsafe_allow_html=True)
            # ---------------------------

        else:
            if mode == "Aynı Mevki Kıyaslama" and selected_pos_filter == "Tümü":
                st.info("👈 Lütfen sol menüden bir **Mevki** seçiniz.")
            else:
                st.info("Analiz için sol taraftan oyuncu seçip butona basınız.")

# ========================================================
# SAYFA 2: GLOBAL SCOUTING
# ========================================================
elif selected == "Global Scouting":
    st.subheader("🌍 Avrupa Liglerinde Benzer Oyuncu Bul")
    
    col_scout_inp, col_scout_out = st.columns([1, 2])
    
    with col_scout_inp:
        st.markdown("### 🎯 Hedef Oyuncu")
        teams = sorted(sl["Takim"].astype(str).unique())
        scout_team = st.selectbox("Takım", teams, key="s_team")
        scout_player = st.selectbox("Oyuncu", sl[sl["Takim"] == scout_team]["Player"], key="s_player")
        
        find_btn = st.button("🔎 Benzerleri Tara")
        
        if scout_player:
            target = sl[sl["Player"] == scout_player].iloc[0]
            st.markdown("---")
            st.write(f"**Mevki:** {target.get('Pos_Simple', '-')}")
            # YAŞ KALDIRILDI
            st.progress(float(target['Overall_Skoru'])/10, text=f"Overall: {target['Overall_Skoru']:.1f}")

    with col_scout_out:
        if find_btn:
            mg = str(target.get("MevkiGroup", "MID"))
            pool = t5[t5["MevkiGroup"].astype(str) == mg].copy() if "MevkiGroup" in t5.columns else t5.copy()
            
            if pool.empty:
                st.warning("Bu mevkide Avrupa verisi bulunamadı.")
            else:
                sim_feats = feats if feats else ["Overall_Skoru", "Hucum_Skoru", "Defans_Skoru", "Fizik_Skoru", "Teknik_Skoru"]
                target_vec = np.array([float(target.get(f, 0)) for f in sim_feats])
                pool_mat = pool[sim_feats].fillna(0).values
                for f in sim_feats: 
                    if f not in pool.columns: pool[f] = 0.0 
                pool_mat = pool[sim_feats].fillna(0).values

                sims = cosine_sim(target_vec, pool_mat)
                pool["Similarity"] = sims * 100
                
                results = pool.sort_values("Similarity", ascending=False).head(10)
                
                st.success(f"Avrupa veritabanında {len(pool)} {mg} oyuncusu tarandı, en benzer 10 oyuncu:")
                
                cols_to_show = ["Player", "Team", "League", "Age", "Similarity", "TM_Link"]
                final_cols = [c for c in cols_to_show if c in results.columns]
                
                st.dataframe(
                    results[final_cols],
                    column_config={
                        "TM_Link": st.column_config.LinkColumn("Transfermarkt", display_text="Profili Gör 🔗"),
                        "Similarity": st.column_config.ProgressColumn("Benzerlik", format="%.1f%%", min_value=0, max_value=100),
                        "Age": st.column_config.NumberColumn("Yaş", format="%d")
                    },
                    hide_index=True, use_container_width=True
                )
                
                best = results.iloc[0]
                st.markdown("#### 🌟 En İyi Eşleşme")
                st.plotly_chart(create_radar(target, best, target['Player'], best['Player'], "Scout Radar"), use_container_width=True)

# ========================================================
# SAYFA 3: HAKKINDA
# ========================================================
elif selected == "Hakkında":
    st.subheader("ℹ️ Proje Hakkında")
    st.markdown("""
    Bu proje, **Süper Lig** oyuncularını yapay zeka ve istatistiksel analiz yöntemleriyle inceleyip, 
    Avrupa'nın 5 büyük ligindeki benzer oyuncularla eşleştirir.
    """)