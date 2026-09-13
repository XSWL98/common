"""Industrial multivariate power anomaly engine: robust baseline + change-point rules."""
import numpy as np, pandas as pd

class PowerAnomalyDetector:
 def __init__(self, window=30, z_threshold=4.0, change_threshold=12.0, min_duration=2): self.window=window; self.z_threshold=z_threshold; self.change_threshold=change_threshold; self.min_duration=min_duration
 def predict(self, frame):
  df=frame.copy(); cols=df.select_dtypes(include=np.number).columns.tolist()
  if not cols: raise ValueError('至少需要一个数值测点列')
  x=df[cols].astype(float).interpolate(limit_direction='both'); med=x.rolling(self.window,min_periods=max(5,self.window//3)).median(); mad=(x-med).abs().rolling(self.window,min_periods=max(5,self.window//3)).median(); scale=(1.4826*mad).replace(0,np.nan).fillna(x.std().replace(0,1)); score=((x-med).abs()/scale).max(axis=1); delta=x.pct_change(max(1,self.window//4)).abs().max(axis=1)*100; flag=(score>=self.z_threshold)|(delta>=self.change_threshold); flag=flag.rolling(self.min_duration,min_periods=1).sum()>=self.min_duration; df['anomaly_score']=score.round(3); df['change_pct']=delta.round(3); df['is_anomaly']=flag; df['severity']=np.select([score>=self.z_threshold*2,score>=self.z_threshold,delta>=self.change_threshold],['critical','high','medium'],default='normal'); df['drivers']=x.sub(med).abs().idxmax(axis=1); return df
 def events(self, result):
  a=result[result.is_anomaly].copy();
  if a.empty:return a
  grp=(a.index.to_series().diff().fillna(1)>1).cumsum(); return a.assign(event_id=grp.values).groupby('event_id',as_index=False).agg(start=('timestamp','min'),end=('timestamp','max'),peak_score=('anomaly_score','max'),max_change_pct=('change_pct','max'),severity=('severity','first'),driver=('drivers','first'))
