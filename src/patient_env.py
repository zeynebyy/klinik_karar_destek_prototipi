import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PatientEnv(gym.Env):
    """
    Hasta Durum Kötüleşmesi Alarm Sistemi için Özel Environment
    """
    def __init__(self, df, reward_config=None):
        super(PatientEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.patient_ids = self.df['patient_id'].unique()
        self.current_patient_idx = 0
        
        # Özellikler (States)
        self.feature_cols = ['hr', 'spo2', 'rr', 'temp', 'systolic_bp', 'diastolic_bp', 
                            'oxygen_flow', 'mobility_score', 'sepsis_risk_score', 
                            'baseline_risk_score', 'age']
        
        # Action Space: 0 = Alarm verme, 1 = Alarm ver
        self.action_space = spaces.Discrete(2)
        
        # Observation Space: Vital bulgular
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols),), dtype=np.float32
        )
        
        # Reward Yapılandırması
        self.reward_config = reward_config or {
            'tp': 10,  # True Positive
            'fp': -5,  # False Positive
            'fn': -20, # False Negative (Kritik hata)
            'tn': 2    # True Negative
        }
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Rastgele bir hasta seç veya sıradaki hastaya geç
        # Eğitimde rastgele seçim daha iyi olabilir
        self.current_patient_id = np.random.choice(self.patient_ids)
        self.patient_data = self.df[self.df['patient_id'] == self.current_patient_id].reset_index(drop=True)
        
        self.current_step = 0
        self.max_steps = len(self.patient_data)
        
        state = self.patient_data.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        
        return state, {}

    def step(self, action):
        # Mevcut durumdaki gerçek olay
        event = self.patient_data.loc[self.current_step, 'event']
        
        # Ödül Hesaplama
        reward = 0
        if action == 1:
            if event == 1:
                reward = self.reward_config['tp']
            else:
                reward = self.reward_config['fp']
        else:
            if event == 1:
                reward = self.reward_config['fn']
            else:
                reward = self.reward_config['tn']
        
        # Bir sonraki adıma geç
        self.current_step += 1
        
        # Episode bitti mi?
        done = self.current_step >= self.max_steps - 1
        truncated = False
        
        if not done:
            next_state = self.patient_data.loc[self.current_step, self.feature_cols].values.astype(np.float32)
        else:
            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        return next_state, reward, done, truncated, {}

    def render(self):
        pass