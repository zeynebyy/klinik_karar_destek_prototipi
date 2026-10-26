# Hasta Durum Kötüleşmesi Alarm Sistemi (RL-PPO)

## Mevcut Uygulama

Proje, PPO ile pekiştirmeli öğrenme deneylerini ve XGBoost tabanlı bir Streamlit karar destek arayüzünü içerir. Mevcut arayüz tahminleri `xgb_base.joblib` modeli ve klinik kural motoruyla üretir; PPO ajanı arayüze bağlı değildir. Arayüzdeki 24 saatlik trend grafiği simülasyondur. Klinik skorlar mevcut girdilerle sınırlı hesaplanır. Bu çalışma bir araştırma/eğitim prototipidir.

Hasta verileri, yerel tahmin veritabanı, eğitim günlükleri ve eğitilmiş model dosyaları depoya dahil edilmez. Aşağıdaki komutları bu README dosyasının bulunduğu `patient_rl_real` klasöründe çalıştırın.

### Streamlit arayüzünü hazırlama

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Kullanım hakkınız olan veri dosyasını `data/hospital_deterioration_hourly_panel.csv.csv` konumuna yerleştirin. Beklenen sütunlar `src/prepare_real_data.py` dosyasındaki `column_mapping` içinde tanımlıdır. Ardından:

```bash
python src/prepare_real_data.py
python src/models/train_xgb.py
streamlit run src/app.py
```

Veri hazırlama işlemi `models/scaler.joblib`, XGBoost eğitimi ise `models/xgb_calibrated.joblib` ve `models/xgb_base.joblib` dosyalarını oluşturur. Aşağıdaki PPO eğitim ve test adımları ayrı deney akışıdır.

`results/xgb_metrics.json` kalibre edilmiş modelin kayıtlı test sonuçlarını içerir. Arayüz temel XGBoost modelini kullandığından bu ölçümler arayüzdeki tahminlerin doğrudan değerlendirmesi değildir.

Bu proje, gerçek hastane verilerini kullanarak hastaların klinik durumundaki kötüleşmeyi (deterioration) önceden tahmin eden ve alarm üreten bir **Pekiştirmeli Öğrenme (Reinforcement Learning)** ajanı geliştirmeyi amaçlar.

## 🚀 Proje Amacı
Klasik sınıflandırma modellerinden farklı olarak, bu sistem bir "ajan" (agent) gibi çalışır. Hastanın vital bulgularını zaman serisi olarak izler ve her adımda bir aksiyon (Alarm Ver / Verme) alır. Amaç:
- Kritik hastaları kaçırmamak (Yüksek Duyarlılık/Recall)
- Gereksiz alarmları (Alarm Yorgunluğu) azaltmak
- Uzun vadeli hasta güvenliğini optimize etmek

## 🛠️ Mimari ve Teknolojiler
- **Algoritma**: Proximal Policy Optimization (PPO) - Stable-Baselines3
- **Ortam**: OpenAI Gym / Gymnasium tabanlı özel `PatientEnv`
- **Veri Seti**: Kaggle Hospital Deterioration Dataset
- **Arayüz**: Streamlit
- **Takip**: TensorBoard

## 📊 Ödül Sistemi (Reward Engineering)
Ajanın doğru davranışı öğrenmesi için şu ödül yapısı kullanılmıştır:
- **True Positive (Doğru Alarm)**: +10
- **False Positive (Gereksiz Alarm)**: -5
- **False Negative (Kaçırılan Kritik Durum)**: -20 (En yüksek ceza)
- **True Negative (Doğru Sessizlik)**: +2

## 📁 Proje Yapısı
```text
patient_rl_real/
│
├── data/           # Orijinal ve işlenmiş veriler
├── logs/           # TensorBoard logları
├── models/         # Eğitilmiş model ve scaler
├── results/        # Performans grafikleri ve EDA
├── src/
│   ├── prepare_real_data.py  # Veri önişleme
│   ├── patient_env.py        # RL ortamı
│   ├── train_rl.py           # PPO eğitimi
│   ├── test_model.py         # Test ve değerlendirme
│   └── app.py                # Streamlit arayüzü
└── requirements.txt
```

## 💻 Nasıl Çalıştırılır?

1. **Gereksinimleri Yükleyin**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Veriyi Hazırlayın**:
   ```bash
   python src/prepare_real_data.py
   ```

3. **Modeli Eğitin**:
   ```bash
   python src/train_rl.py
   ```

4. **Modeli Test Edin**:
   ```bash
   python src/test_model.py
   ```

5. **Arayüzü Başlatın**:
   ```bash
   streamlit run src/app.py
   ```

6. **TensorBoard İzleme**:
   ```bash
   tensorboard --logdir logs/
   ```

## 📝 Akademik Rapor
Bu çalışma, RL'in sağlık sektöründeki "Karar Destek Sistemleri" için kullanımını örnekler. Geleneksel modellerin aksine, RL ajanları ardışık karar verme (sequential decision making) süreçlerinde toplam ödülü maksimize etmeye odaklandıkları için, klinik alarm yorgunluğunu azaltma konusunda daha esnek bir yapı sunarlar.
