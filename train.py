"""Train a reconstruction model on healthy CSV data (optional torch dependency)."""
import argparse, json, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
import joblib
def main():
 p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--out',default='artifacts'); a=p.parse_args(); d=pd.read_csv(a.csv); x=d.select_dtypes('number').interpolate().bfill().ffill(); sc=RobustScaler().fit(x); model=MLPRegressor(hidden_layer_sizes=(128,64,128),max_iter=200,random_state=7,early_stopping=True).fit(sc.transform(x),sc.transform(x)); import os; os.makedirs(a.out,exist_ok=True); joblib.dump({'model':model,'scaler':sc,'columns':x.columns.tolist()},a.out+'/reconstruction.joblib'); json.dump({'features':x.columns.tolist()},open(a.out+'/metadata.json','w'))
if __name__=='__main__': main()
