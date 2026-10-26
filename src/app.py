import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# 'src' paketinin bulunabilmesi için kök dizini (parent_dir) yola ekliyoruz
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.rules.clinical_rules import ClinicalRuleEngine
from src.utils.logger import ClinicalLogger
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Klinik Karar Destek Sistemi", layout="wide")

# Numaratörlerdeki (number_input) artı/eksi butonlarını gizleyen ve 
# sadece klavyeden girişe zorlayan özel CSS kodu
st.markdown("""
<style>
[data-testid="stNumberInputContainer"] button {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Klinik Karar Destek Sistemi (CDSS) - Erken Uyarı Dashboard'u")

# Session State Initialization
default_vals = {
    'hr': 80, 'spo2': 96, 'rr': 16, 'temp': 36.8, 'sbp': 120, 'dbp': 80,
    'o2_flow': 0.0, 'mobility': 2, 'sepsis_score': 0.1, 'baseline_risk': 0.2, 'age': 45,
    'threshold': 0.5
}
for k, v in default_vals.items():
    if k not in st.session_state: st.session_state[k] = v

# --- Senaryo Fonksiyonları ---
def set_scenario(scenario_name):
    if scenario_name == "Normal":
        st.session_state.update({'hr': 75, 'spo2': 98, 'rr': 14, 'temp': 36.6, 'sbp': 120, 'dbp': 80})
    elif scenario_name == "Erken Sepsis":
        st.session_state.update({'hr': 105, 'spo2': 95, 'rr': 24, 'temp': 38.5, 'sbp': 105, 'dbp': 65, 'sepsis_score': 0.6})
    elif scenario_name == "Septik Şok":
        st.session_state.update({'hr': 135, 'spo2': 91, 'rr': 28, 'temp': 39.2, 'sbp': 85, 'dbp': 50, 'sepsis_score': 0.9})
    elif scenario_name == "Hipoksi":
        st.session_state.update({'hr': 110, 'spo2': 88, 'rr': 26, 'temp': 37.0, 'sbp': 130, 'dbp': 85})

@st.cache_resource
def load_resources():
    model_path = "models/xgb_calibrated.joblib"
    scaler_path = "models/scaler.joblib"
    base_model_path = "models/xgb_base.joblib"
    
    model, scaler, base_model = None, None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        if os.path.exists(base_model_path):
            base_model = joblib.load(base_model_path)
            
    return model, scaler, base_model

model, scaler, base_model = load_resources()
logger = ClinicalLogger()

if model is None:
    st.error("Model bulunamadı. Lütfen önce `src/models/train_xgb.py` betiğini çalıştırın.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("ℹ️ Sistem Hakkında")
st.sidebar.info("""
**Nasıl Kullanılır?**
1. Aşağıdaki **Test Senaryoları** butonlarına tıklayarak örnek hasta profillerini hızlıca yükleyebilirsiniz.
2. Sağdaki **Vital Bulguları Manuel Ayarla** bölümünden hastanın anlık değerlerini girerek sistemin tepkisini görebilirsiniz.
3. Her bir değerin ne anlama geldiğini görmek için yanındaki **(?)** işaretine fare ile gelin.
4. "Açıklanabilir Yapay Zeka" sekmesinden yapay zekanın karar verirken nelere dikkat ettiğini görebilirsiniz.
""")

st.sidebar.header("🧪 Test Senaryoları")
st.sidebar.button("🟢 Stabil Hasta", on_click=set_scenario, args=("Normal",), use_container_width=True)
st.sidebar.button("🟡 Erken Sepsis", on_click=set_scenario, args=("Erken Sepsis",), use_container_width=True)
st.sidebar.button("🔴 Septik Şok", on_click=set_scenario, args=("Septik Şok",), use_container_width=True)
st.sidebar.button("🔵 Akut Hipoksi", on_click=set_scenario, args=("Hipoksi",), use_container_width=True)
st.sidebar.divider()
st.sidebar.markdown("### Karar Eşiği (Threshold)")
st.session_state.threshold = st.sidebar.slider("ML Tahmin Eşiği", 0.1, 0.9, st.session_state.threshold, 0.05)

# --- TABS ---
tab_klinik, tab_xai, tab_performans = st.tabs(["🩺 Klinik Dashboard", "🧠 Açıklanabilir Yapay Zeka (XAI)", "📊 Model Performansı & Eğitim"])

with tab_klinik:
    status_placeholder = st.empty()
    
    # Vitals Input
    with st.expander("Vital Bulguları Manuel Ayarla (Detayları Gör)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Kardiyovasküler**")
            hr = st.number_input("Nabız (HR)", 40, 180, int(st.session_state.hr), key='hr', help="Kalp atım hızı. Normali 60-100 arasıdır.")
            sbp = st.number_input("Sistolik Kan Basıncı (SBP)", 70, 200, int(st.session_state.sbp), key='sbp', help="Büyük tansiyon. Normali 90-120 mmHg arasıdır.")
            dbp = st.number_input("Diastolik Kan Basıncı (DBP)", 40, 120, int(st.session_state.dbp), key='dbp', help="Küçük tansiyon. Normali 60-80 mmHg arasıdır.")
        with col2:
            st.markdown("**Solunum & Ateş**")
            spo2 = st.number_input("SpO2 (%)", 70, 100, int(st.session_state.spo2), key='spo2', help="Kandaki oksijen doygunluğu. %95 ve üzeri normal kabul edilir.")
            rr = st.number_input("Solunum Sayısı (RR)", 8, 40, int(st.session_state.rr), key='rr', help="Dakikadaki nefes alış sayısı. Normali 12-20 arasıdır.")
            temp = st.number_input("Vücut Sıcaklığı (°C)", 34.0, 42.0, float(st.session_state.temp), 0.1, key='temp', help="Vücut ısısı. 36.5 - 37.5 arası normaldir.")
            o2_flow = st.number_input("Oksijen Akışı (L/dk)", 0.0, 15.0, float(st.session_state.o2_flow), 0.5, key='o2_flow', help="Hastaya dışarıdan verilen oksijen desteği miktarı.")
        with col3:
            st.markdown("**Klinik Skorlar & Diğer**")
            age = st.number_input("Yaş", 18, 100, int(st.session_state.age), key='age', help="Hastanın yaşı. İleri yaş risk faktörüdür.")
            mobility = st.selectbox("Mobilite Skoru", [0, 1, 2, 3, 4], index=int(st.session_state.mobility), key='mobility', help="Hareket kabiliyeti. 0: Tam hareketli, 4: Yatağa bağımlı/bilinç kapalı.")
            sepsis_score = st.number_input("Sepsis Risk Skoru", 0.0, 1.0, float(st.session_state.sepsis_score), 0.01, key='sepsis_score', help="Laboratuvar verilerinden hesaplanan temel sepsis riski.")
            baseline_risk = st.number_input("Temel Risk Skoru", 0.0, 1.0, float(st.session_state.baseline_risk), 0.01, key='baseline_risk', help="Hastanın kronik hastalıklarına bağlı temel risk katsayısı.")

    # Prepare Data
    input_dict = {'hr':hr, 'spo2':spo2, 'rr':rr, 'temp':temp, 'sbp':sbp, 'dbp':dbp, 
                  'o2_flow':o2_flow, 'mobility':mobility, 'sepsis_score':sepsis_score, 
                  'baseline_risk':baseline_risk, 'age':age}
                  
    features = ['hr', 'spo2', 'rr', 'temp', 'sbp', 'dbp', 'o2_flow', 'mobility', 'sepsis_score', 'baseline_risk', 'age']
    input_data = np.array([[input_dict[f] for f in features]])
    
    # Model Prediction (Using base_model to avoid sklearn XGBoost Pickling bugs)
    input_scaled = scaler.transform(input_data)
    prob = base_model.predict_proba(input_scaled)[0, 1]
    
    # Rule Engine Evaluation
    severity_level, severity_label, alert_reason = ClinicalRuleEngine.evaluate_severity(
        prob, hr, rr, temp, sbp, spo2, threshold=st.session_state.threshold
    )
    
    patterns = ClinicalRuleEngine.analyze_patterns(hr, rr, temp, sbp, spo2)
    news2 = ClinicalRuleEngine.calculate_news2(hr, rr, temp, sbp, spo2, o2_flow)
    qsofa = ClinicalRuleEngine.calculate_qsofa(rr, sbp)
    sirs = ClinicalRuleEngine.calculate_sirs(temp, hr, rr)
    
    # Logging
    logger.log_prediction(input_dict, prob, severity_level, severity_label, alert_reason)

    # UI Presentation
    colors = {0: "green", 1: "goldenrod", 2: "darkorange", 3: "red"}
    bg_colors = {0: "#e6ffe6", 1: "#fff8e6", 2: "#ffebe6", 3: "#ffe6e6"}
    
    if severity_level == 3:
        st.markdown("""
        <style>
        .stApp {
            background-color: #fff0f0 !important;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.8; }
            100% { opacity: 1; }
        }
        .critical-alert {
            animation: blink 1.5s linear infinite;
            padding: 20px; 
            border-radius: 10px; 
            background-color: #cc0000; 
            color: white !important;
            border-left: 10px solid #800000; 
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 0 15px rgba(204, 0, 0, 0.5);
        }
        .critical-alert h2, .critical-alert p, .critical-alert h1 {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        status_html = f"""
        <div class="critical-alert">
            <h1 style="margin-top: 0; font-size: 2.5rem;">🚨 KRİTİK ALARM 🚨</h1>
            <h2 style="margin-top: 0;">Durum: {severity_label}</h2>
            <p style="font-size: 1.2rem;"><strong>Neden:</strong> {alert_reason}</p>
            <p><strong>ML Risk Olasılığı:</strong> %{prob*100:.1f}</p>
        </div>
        """
    else:
        status_html = f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {bg_colors[severity_level]}; border-left: 10px solid {colors[severity_level]}; margin-bottom: 20px;">
            <h2 style="color: {colors[severity_level]}; margin-top: 0;">Durum: {severity_label}</h2>
            <p><strong>Neden:</strong> {alert_reason}</p>
            <p><strong>ML Risk Olasılığı:</strong> %{prob*100:.1f}</p>
        </div>
        """
        
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    
    col_scores, col_patterns = st.columns(2)
    with col_scores:
        st.subheader("Hesaplanan Klinik Skorlar")
        st.write(f"- **NEWS2 Skoru:** {news2} (>=5 Kritik)  \n  *Hastanın genel klinik bozulma riskini ölçen ulusal erken uyarı skoru.*")
        st.write(f"- **qSOFA Skoru:** {qsofa} (>=2 Yüksek Risk)  \n  *Sepsis (kan zehirlenmesi) şüphesi olan hastalarda mortalite riskini gösterir.*")
        st.write(f"- **SIRS Skoru:** {sirs} (>=2 Sistemik Enflamasyon)  \n  *Vücudun enfeksiyona veya travmaya verdiği sistemik tepkiyi ölçer.*")
        
    with col_patterns:
        st.subheader("Kompleks Etkileşim (Pattern) Analizi")
        if patterns:
            for p in patterns:
                st.error(f"⚠️ {p}")
        else:
            st.success("Bilinen kritik pattern tespit edilmedi.")
            
    # Trend Analysis Simulation
    st.divider()
    st.subheader("📈 Son 24 Saatlik Trend Analizi (Simülasyon)")
    # Generate dummy time series based on current value to show trend capability
    np.random.seed(42)
    hours = list(range(-24, 1))
    trend_hr = [hr + np.random.normal(0, 5) for _ in hours[:-1]] + [hr]
    trend_spo2 = [min(100, spo2 + np.random.normal(0, 1)) for _ in hours[:-1]] + [spo2]
    trend_sbp = [sbp + np.random.normal(0, 8) for _ in hours[:-1]] + [sbp]
    
    df_trend = pd.DataFrame({'Saat': hours, 'Nabız': trend_hr, 'SpO2': trend_spo2, 'Tansiyon(SBP)': trend_sbp})
    fig_trend = px.line(df_trend, x='Saat', y=['Nabız', 'SpO2', 'Tansiyon(SBP)'], title="Vital Bulgu Trendi")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab_xai:
    st.header("Model Açıklanabilirliği (Feature Importance)")
    st.write("Aşağıdaki grafik, modelin genel olarak karar verirken hangi vital bulgulara daha çok önem verdiğini gösterir. (XGBoost Native Feature Importance)")
    
    if base_model is not None:
        importances = base_model.feature_importances_
        df_imp = pd.DataFrame({'Özellik': features, 'Önem Derecesi': importances})
        df_imp = df_imp.sort_values('Önem Derecesi', ascending=True)
        
        fig_imp = px.bar(df_imp, x='Önem Derecesi', y='Özellik', orientation='h', 
                         title='Vital Bulguların Klinik Karardaki Etkisi')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("Bu özelliklerin anlık değerleri klinik kural motoru tarafından da değerlendirilmektedir.")
    else:
        st.warning("Model yüklenemediği için açıklanabilirlik grafiği çizilemiyor.")

with tab_performans:
    st.header("Model Eğitimi ve Performans Metrikleri")
    
    try:
        with open("results/xgb_metrics.json", "r") as f:
            metrics = json.load(f)
            
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        col_m2.metric("Precision", f"{metrics['precision']:.3f}")
        col_m3.metric("Recall", f"{metrics['recall']:.3f}")
        col_m4.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        
    except FileNotFoundError:
        st.info("Metrikler henüz oluşturulmamış. Modeli eğitin.")
        
    st.subheader("Threshold Optimizasyonu")
    st.write(f"Şu anki Threshold: **{st.session_state.threshold}**")
    st.info("Eşiği sol taraftaki menüden değiştirebilirsiniz. Eşiğin değişmesi sadece ML tabanlı risk seviyelerini etkiler, Hard Clinical Rule'ları değiştirmez.")
