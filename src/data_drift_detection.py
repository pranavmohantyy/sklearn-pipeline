import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

class DataDriftDetector:
    def __init__(self, baseline_data):
        self.baseline_data = baseline_data
        self.scaler = StandardScaler().fit(baseline_data)

    def check_drift(self, new_data):
        new_data_scaled = self.scaler.transform(new_data)
        baseline_scaled = self.scaler.transform(self.baseline_data)
        drift_results = {}
        for column in new_data.columns:
            stat, p_value = ks_2samp(baseline_scaled[:, new_data.columns.get_loc(column)], new_data_scaled[:, new_data.columns.get_loc(column)])
            drift_results[column] = p_value < 0.05
        return drift_results

if __name__ == '__main__':
    baseline_data = pd.read_csv('path_to_baseline_data.csv')
    detector = DataDriftDetector(baseline_data)
    new_data = pd.read_csv('path_to_new_data.csv')
    drift = detector.check_drift(new_data)
    print(drift)