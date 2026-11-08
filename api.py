from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import numpy as np, pandas as pd
from power_ad import PowerAnomalyDetector
from deep_detector import DeepPowerDetector

app=FastAPI(title='智序工业电力异常检测 API'); detector=PowerAnomalyDetector()

def _json_safe(records):
 """把 NaN/Inf/NumPy 标量/Timestamp 转成合法 JSON 值,避免 500 或非法 JSON。"""
 out=[]
 for rec in records:
  clean={}
  for k,v in rec.items():
   if isinstance(v,np.bool_): v=bool(v)
   elif isinstance(v,np.integer): v=int(v)
   elif isinstance(v,np.floating): v=float(v)
   if isinstance(v,float) and not np.isfinite(v): v=None
   elif isinstance(v,pd.Timestamp): v=v.isoformat()
   elif isinstance(v,(np.datetime64,)): v=pd.Timestamp(v).isoformat()
   clean[k]=v
  out.append(clean)
 return out

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/v1/detect')
def detect(file:UploadFile=File(...),model:str='robust'):
 try: d=pd.read_csv(file.file)
 except Exception as error: raise HTTPException(400,f'CSV 解析失败: {error}')
 numeric=d.select_dtypes(include=np.number).columns.tolist()
 if not numeric: raise HTTPException(400,'CSV 中没有数值测点列')
 if model=='robust':
  if 'timestamp' not in d.columns: raise HTTPException(400,'缺少 timestamp 列,无法生成事件摘要')
  if len(d)<5: raise HTTPException(400,'数据少于 5 行,无法计算滚动基线')
  r=detector.predict(d); events=detector.events(r).to_dict('records')
 elif model in ('patchtst','tranad'):
  path=Path('artifacts')/model
  if not path.exists(): raise HTTPException(404,f'{model} model is not trained')
  try: r=DeepPowerDetector.load(path).predict(d[numeric])
  except ValueError as error: raise HTTPException(400,str(error))
  events=[]
 else: raise HTTPException(400,'model must be robust, patchtst or tranad')
 return {'model':model,'records':_json_safe(r.to_dict('records')),'events':_json_safe(events)}
