import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

# Klasörleri oluştur
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def prepare_data(file_path):
    print(f"Veri yükleniyor: {file_path}")
    df = pd.read_csv(file_path)
    
    # İstenen kolonları seç ve isimlendir
    # Orijinal: heart_rate, respiratory_rate, spo2_pct, temperature_c, systolic_bp, diastolic_bp, oxygen_flow, mobility_score, sepsis_risk_score, baseline_risk_score, age, deterioration_next_12h
    column_mapping = {
        'heart_rate': 'hr',
        'respiratory_rate': 'rr',
        'spo2_pct': 'spo2',
        'temperature_c': 'temp',
        'systolic_bp': 'systolic_bp',
        'diastolic_bp': 'diastolic_bp',
        'oxygen_flow': 'oxygen_flow',
        'mobility_score': 'mobility_score',
        'sepsis_risk_score': 'sepsis_risk_score',
        'baseline_risk_score': 'baseline_risk_score',
        'age': 'age',
        'deterioration_next_12h': 'event',
        'patient_id': 'patient_id',
        'hour_from_admission': 'hour'
    }
    
    selected_cols = list(column_mapping.keys())
    df = df[selected_cols]
    df = df.rename(columns=column_mapping)
    
    # Eksik verileri temizle
    print("Eksik veriler temizleniyor...")
    df = df.dropna()
    
    # Analiz ve Grafikler
    print("EDA grafikleri oluşturuluyor...")
    
    # 1. Sınıf Dağılımı
    plt.figure(figsize=(8, 6))
    sns.countplot(x='event', data=df, palette='viridis')
    plt.title('Sınıf Dağılımı (Kötüleşme Olayı)')
    plt.xlabel('Olay (0: Yok, 1: Var)')
    plt.ylabel('Sayı')
    plt.savefig('results/class_distribution.png')
    plt.close()
    
    # 2. Korelasyon Matrisi
    plt.figure(figsize=(12, 10))
    corr = df.drop(['patient_id', 'hour'], axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Özellik Korelasyon Heatmap')
    plt.savefig('results/correlation_heatmap.png')
    plt.close()
    
    # Veriyi Kaydet (Orijinal haliyle ama temizlenmiş)
    df.to_csv("data/real_patient_data.csv", index=False)
    
    # Normalizasyon (Sadece özellikler için)
    features = ['hr', 'spo2', 'rr', 'temp', 'systolic_bp', 'diastolic_bp', 'oxygen_flow', 'mobility_score', 'sepsis_risk_score', 'baseline_risk_score', 'age']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Train/Test Split (Hastalar bazında ayırmak daha doğru olur ama basitlik için direkt ayırıyoruz)
    # Gerçek bir RL projesinde test hastaları eğitimde hiç görülmemelidir.
    unique_patients = df['patient_id'].unique()
    train_ids, test_ids = train_test_split(unique_patients, test_size=0.2, random_state=42)
    
    df_train = df[df['patient_id'].isin(train_ids)]
    df_test = df[df['patient_id'].isin(test_ids)]
    
    df_test.to_csv("data/real_patient_data_test.csv", index=False)
    
    # Dengelenmiş Veri Seti Oluşturma (Sadece Eğitim Seti Üzerinde)
    print("Eğitim seti dengeleniyor...")
    df_majority = df_train[df_train.event == 0]
    df_minority = df_train[df_train.event == 1]
    
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=42)
    
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_balanced.to_csv("data/real_patient_data_balanced.csv", index=False)
    
    print("Veri hazırlama tamamlandı.")
    print(f"Normal veri boyutu: {len(df)}")
    print(f"Eğitim (Dengelenmiş) boyutu: {len(df_balanced)}")
    print(f"Test boyutu: {len(df_test)}")

if __name__ == "__main__":
    raw_data_path = "data/hospital_deterioration_hourly_panel.csv.csv"
    prepare_data(raw_data_path)