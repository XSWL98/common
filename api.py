from fastapi import FastAPI, UploadFile, File
from power_ad import PowerAnomalyDetector
import pandas as pd
app=FastAPI(title='智序工业电力异常检测 API'); detector=PowerAnomalyDetector()
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/v1/detect')
async def detect(file:UploadFile=File(...)):
 d=pd.read_csv(file.file); r=detector.predict(d); return {'records':r.to_dict('records'),'events':detector.events(r).to_dict('records')}
