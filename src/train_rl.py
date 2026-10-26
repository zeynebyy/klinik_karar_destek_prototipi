import os
import sys
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

# Proje kök dizinini ekle
sys.path.append(os.getcwd())
from src.patient_env import PatientEnv

# Klasörleri oluştur
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def train():
    # 1. Veriyi yükle (Dengelenmiş veri seti)
    print("Veri yükleniyor...")
    df_balanced = pd.read_csv("data/real_patient_data_balanced.csv")
    
    # 2. Environment'ı oluştur
    print("Environment hazrlanyor...")
    env = PatientEnv(df_balanced)
    env = Monitor(env) # Metrikleri takip etmek iin
    
    # 3. Model Parametreleri
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[128, 128])) # Opsiyonel: Mimari
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.00005,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./logs/",
        device="auto" # GPU varsa kullanr
    )
    
    # 4. Callback'ler
    # Model kaydetme (her 10.000 admda)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./models/checkpoints/',
        name_prefix='ppo_patient'
    )
    
    # Erken durdurma ve en iyi modeli kaydetme
    # Not: Gerek projede eval_env ayr bir test setinden olumaldr
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_callback = EvalCallback(
        env, 
        callback_after_eval=stop_callback, 
        best_model_save_path='./models/best_model/',
        log_path='./logs/eval/', 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )
    
    # 5. Eitimi Balat
    print("Eitim balyor...")
    total_timesteps = 100000 # Gsterim iin 100k, gerek eitimde daha fazla olabilir
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="PPO_Patient_Run"
    )
    
    # 6. Final Modeli Kaydet
    model.save("models/ppo_patient_real")
    print("Eitim tamamland ve model kaydedildi.")

if __name__ == "__main__":
    train()