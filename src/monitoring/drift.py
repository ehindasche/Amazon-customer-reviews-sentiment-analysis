#! D:\Python\myvenv\Scripts\python.exe

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def detect_drift():
    """Detect data drift between training and production data"""
    reference = pd.read_csv("data/processed/train.csv")
    current = pd.read_csv("data/processed/test.csv")
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("monitoring/drift_report.html")

if __name__ == "__main__":
    detect_drift()
