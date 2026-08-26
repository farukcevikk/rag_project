import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
FEEDBACK_LOG_PATH = PROJECT_DIR / "log" / "user_feedback_logs.jsonl"

st.set_page_config(page_title="RAG Analitik Paneli", page_icon="📊", layout="wide")
st.title("📊 RAG Sistem Analitik Paneli")
st.markdown("Bu panel, operatörlerin asistanı kullanırken bıraktığı geri bildirimleri (👍/👎) analiz eder.")

# --- VERİ OKUMA İŞLEMİ ---
# @st.cache_data
def load_feedback_data():
    if not FEEDBACK_LOG_PATH.exists():
        return pd.DataFrame()
    
    data = []
    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
                
    df = pd.DataFrame(data)
    if not df.empty:
        # Zaman sütununu datetime objesine çevir
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_feedback_data()

# --- EĞER VERİ YOKSA UYARI VER ---
if df.empty:
    st.info("Henüz loglanmış bir kullanıcı geri bildirimi (user_feedback_logs.jsonl) bulunmuyor. Ana sayfada birkaç soru sorup 👍/👎 butonlarına tıklayarak veri oluşturabilirsiniz.")
else:
    # --- 1. ÜST BİLGİ KARTLARI (KPI) ---
    col1, col2, col3 = st.columns(3)
    
    toplam_soru = len(df)
    olumlu_yanit = len(df[df['is_positive'] == True])
    memnuniyet_orani = (olumlu_yanit / toplam_soru) * 100 if toplam_soru > 0 else 0
    
    col1.metric("Toplam Geri Bildirim", toplam_soru)
    col2.metric("Olumlu Yanıt (👍)", olumlu_yanit)
    col3.metric("Kullanıcı Memnuniyet Oranı", f"%{memnuniyet_orani:.1f}")
    
    st.markdown("---")
    
    # --- 2. GRAFİKLER ---
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Memnuniyet Dağılımı")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        # Pandas ile pratik pie chart
        df['is_positive'].map({True: 'Olumlu', False: 'Olumsuz'}).value_counts().plot.pie(
            autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], 
            ax=ax1, 
            ylabel=''
        )
        st.pyplot(fig1)

    with col_chart2:
        st.subheader("Son Geri Bildirimler")
        # En son gelen 5 geri bildirimi tablo olarak göster
        son_islemler = df.sort_values(by='timestamp', ascending=False).head(5)
        
        # Sadece göstermek istediğimiz sütunları seçip formatlıyoruz
        gosterim_df = son_islemler[['timestamp', 'question', 'is_positive']].copy()
        gosterim_df['Durum'] = gosterim_df['is_positive'].map({True: '👍', False: '👎'})
        gosterim_df = gosterim_df.drop(columns=['is_positive'])
        gosterim_df.columns = ['Tarih', 'Soru', 'Durum']
        
        st.dataframe(gosterim_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # --- 3. KÖK NEDEN ANALİZİ İÇİN BAŞARISIZ SORULAR ---
    st.subheader("🔧 İyileştirme Gereken Sorular (Olumsuz Geri Bildirimler)")
    olumsuz_df = df[df['is_positive'] == False]
    
    if not olumsuz_df.empty:
        st.dataframe(
            olumsuz_df[['timestamp', 'question', 'answer']], 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.success("Harika! Hiç olumsuz geri bildirim alınmamış.")
