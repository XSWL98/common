import argparse,pandas as pd
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score
from power_ad import PowerAnomalyDetector
p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--label',default='label'); a=p.parse_args(); d=pd.read_csv(a.csv); r=PowerAnomalyDetector().predict(d); y=d[a.label].astype(int); print({'f1':f1_score(y,r.is_anomaly),'roc_auc':roc_auc_score(y,r.anomaly_score),'pr_auc':average_precision_score(y,r.anomaly_score)})
