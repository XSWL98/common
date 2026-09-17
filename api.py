from fastapi import FastAPI, UploadFile, File, HTTPException
from power_ad import PowerAnomalyDetector
from deep_detector import DeepPowerDetector
from pathlib import Path
import pandas as pd
app=FastAPI(title='智序工业电力异常检测 API'); detector=PowerAnomalyDetector()
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/v1/detect')
async def detect(file:UploadFile=File(...),model:str='robust'):
 d=pd.read_csv(file.file)
 if model=='robust': r=detector.predict(d); events=detector.events(r).to_dict('records')
 elif model in ('patchtst','tranad'):
  path=Path('artifacts')/model
  if not path.exists():raise HTTPException(404,f'{model} model is not trained')
  r=DeepPowerDetector.load(path).predict(d.select_dtypes(include='number')); events=[]
 else:raise HTTPException(400,'model must be robust, patchtst or tranad')
 return {'model':model,'records':r.to_dict('records'),'events':events}
