import pandas as pd
import numpy as np
from src.engine import detect_anomalies

def test_anomaly_output():
    # Create fake data
    df = pd.DataFrame({
        'Close': np.random.rand(100),
        'Volume': np.random.rand(100)
    })
    result = detect_anomalies(df)
    assert 'Is_Anomaly' in result.columns
    assert result['Is_Anomaly'].isin(['Yes', 'No']).all()