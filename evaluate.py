import argparse,pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from power_ad import PowerAnomalyDetector
p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--label',default='label'); a=p.parse_args(); d=pd.read_csv(a.csv); y=d[a.label].astype(int); r=PowerAnomalyDetector().predict(d.drop(columns=[a.label],errors='ignore')); two=y.nunique()>1; print({'f1':f1_score(y,r.is_anomaly,zero_division=0),'roc_auc':roc_auc_score(y,r.anomaly_score) if two else None,'pr_auc':average_precision_score(y,r.anomaly_score) if two else None,'alerts':int(r.is_anomaly.sum()),'positives':int(y.sum())})
