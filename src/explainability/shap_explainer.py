import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class SHAPExplainer:
    def __init__(self, model_path="models/xgb_base.joblib"):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.explainer = shap.TreeExplainer(self.model)
            
    def get_shap_values(self, input_df):
        if self.explainer is None:
            return None
        shap_values = self.explainer(input_df)
        return shap_values
        
    def plot_waterfall(self, shap_values, max_display=10):
        # shap_values is expected to be a single instance Explaination object
        fig, ax = plt.subplots(figsize=(8, 6))
        # shap.plots.waterfall typically draws on the current matplotlib figure. 
        # We need to catch it.
        shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
        plt.tight_layout()
        return fig
