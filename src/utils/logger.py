import sqlite3
import pandas as pd
from datetime import datetime
import os

class ClinicalLogger:
    def __init__(self, db_path="data/clinical_logs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                hr REAL,
                spo2 REAL,
                rr REAL,
                temp REAL,
                sbp REAL,
                dbp REAL,
                o2_flow REAL,
                mobility INTEGER,
                sepsis_score REAL,
                baseline_risk REAL,
                age INTEGER,
                prediction_prob REAL,
                severity_level INTEGER,
                severity_label TEXT,
                alert_reason TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_prediction(self, inputs, prob, severity_level, severity_label, alert_reason):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        hr = float(inputs.get('hr', 0))
        spo2 = float(inputs.get('spo2', 0))
        rr = float(inputs.get('rr', 0))
        temp = float(inputs.get('temp', 0))
        sbp = float(inputs.get('sbp', 0))
        dbp = float(inputs.get('dbp', 0))
        o2_flow = float(inputs.get('o2_flow', 0))
        mobility = int(inputs.get('mobility', 0))
        sepsis_score = float(inputs.get('sepsis_score', 0))
        baseline_risk = float(inputs.get('baseline_risk', 0))
        age = int(inputs.get('age', 0))

        cursor.execute('''
            INSERT INTO predictions (
                timestamp, hr, spo2, rr, temp, sbp, dbp, o2_flow, mobility, 
                sepsis_score, baseline_risk, age, prediction_prob, severity_level, severity_label, alert_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, hr, spo2, rr, temp, sbp, dbp, o2_flow, mobility,
            sepsis_score, baseline_risk, age, float(prob), int(severity_level), str(severity_label), str(alert_reason)
        ))
        conn.commit()
        conn.close()

    def get_recent_logs(self, limit=100):
        if not os.path.exists(self.db_path):
            return pd.DataFrame()
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}", conn)
        conn.close()
        return df
