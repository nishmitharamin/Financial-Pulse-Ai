import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(data):
    # 1. AI Logic: Isolation Forest (Finds complex patterns)
    model = IsolationForest(contamination=0.01, random_state=42)
    inputs = data[['Close', 'Volume']].values
    data['AI_Score'] = model.fit_predict(inputs)
    
    # 2. Math Logic: Z-Score (Finds sudden price spikes)
    # window=20 means it looks at the last 20 hours/days of data
    window = 20
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    
    # Formula for Z-Score: (Value - Average) / Standard Deviation
    data['Z_Score'] = (data['Close'] - rolling_mean) / rolling_std
    
    # HYBRID RULE: If AI detects it (-1) OR price is 3x the standard deviation (>3)
    data['Is_Anomaly'] = np.where((data['AI_Score'] == -1) | (data['Z_Score'].abs() > 3), 'Yes', 'No')
    
    return data