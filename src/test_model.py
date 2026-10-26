import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
from stable_baselines3 import PPO

# Proje kök dizinini ekle
sys.path.append(os.getcwd())
from src.patient_env import PatientEnv

def test():
    # 1. Test verisini yükle
    print("Test verisi yükleniyor...")
    df_test = pd.read_csv("data/real_patient_data_test.csv")
    
    # 2. Modeli yükle
    model_path = "models/ppo_patient_real.zip"
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}. Lütfen önce eğitimi tamamlayın.")
        return
    
    model = PPO.load(model_path)
    
    # 3. Test Environment'ı
    env = PatientEnv(df_test)
    
    # 4. Tahminleri Topla
    y_true = []
    y_pred = []
    rewards = []
    
    print("Test süreci başlıyor...")
    num_episodes = 100 # Hızlı test için 100 hasta
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            # Gerçek olayı al (env içinden gizlice alıyoruz metrik için)
            event = env.patient_data.loc[env.current_step, 'event']
            
            next_obs, reward, done, truncated, info = env.step(action)
            
            y_true.append(event)
            y_pred.append(action)
            rewards.append(reward)
            
            obs = next_obs
            
    # 5. Metrikleri Hesapla
    print("\n--- Test Sonuçları ---")
    print(classification_report(y_true, y_pred))
    
    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC-AUC Skoru: {auc:.4f}")
    
    # 6. Grafikler
    os.makedirs("results", exist_ok=True)
    
    # Karışıklık Matrisi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.png')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'PPO (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('results/roc_curve.png')
    
    # Ödül Dağılımı
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, color='skyblue', edgecolor='black')
    plt.title('Ödül Dağılımı')
    plt.savefig('results/reward_distribution.png')
    
    print("Grafikler 'results/' klasörüne kaydedildi.")

if __name__ == "__main__":
    test()