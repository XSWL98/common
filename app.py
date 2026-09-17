import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from power_ad import PowerAnomalyDetector
from pathlib import Path
from deep_detector import DeepPowerDetector,ModelConfig
from features import wavelet_features
st.set_page_config(page_title='智序工业电力异常检测',page_icon='⚡',layout='wide')
@st.cache_data
def demo(n=1440):
 r=np.random.default_rng(7); t=pd.date_range(end=pd.Timestamp.now().floor('min'),periods=n,freq='min'); i=np.arange(n); load=420+90*np.sin(i*2*np.pi/1440)+r.normal(0,8,n); v=380+r.normal(0,.7,n); cur=load/(v*.92)*1000+r.normal(0,4,n); f=50+r.normal(0,.025,n); pf=.92+r.normal(0,.012,n)
 for x,d in [(430,180),(875,-210),(1180,130)]: load[x:x+8]+=d; cur[x:x+8]+=d/.35
 return pd.DataFrame({'timestamp':t,'active_power_kw':load,'voltage_v':v,'current_a':cur,'frequency_hz':f,'power_factor':pf})
def detect(df,cols,w,z,s):
 return PowerAnomalyDetector(w,z,s).predict(df)
st.title('⚡ 智序·工业电力异常检测'); st.caption('多元时序 · 动态阈值 · 突增突降识别')
with st.sidebar:
 model_name=st.selectbox('检测模型',['鲁棒统计基线','PatchTST-AE','TranAD']); use_wavelet=st.toggle('小波多尺度增强',False); up=st.file_uploader('上传 CSV',type=['csv']); w=st.slider('基线窗口（分钟）',5,120,30,5); z=st.slider('鲁棒异常阈值',2.,8.,4.,.5); s=st.slider('突变阈值（%）',2.,50.,12.,1.); train_now=st.button('用当前数据训练深度模型',disabled=model_name=='鲁棒统计基线',use_container_width=True)
df=demo()
if up:
 raw=pd.read_csv(up); tc=next((c for c in raw.columns if c.lower() in ['timestamp','time','datetime','date']),raw.columns[0]); raw[tc]=pd.to_datetime(raw[tc],errors='coerce'); raw=raw.rename(columns={tc:'timestamp'}).dropna(subset=['timestamp']); nc=raw.select_dtypes(include=np.number).columns.tolist(); df=raw[['timestamp']+nc] if nc else df
cols=df.select_dtypes(include=np.number).columns.tolist()
if model_name=='鲁棒统计基线': out=detect(df,cols,w,z,s); out['alert']=out.is_anomaly; out['status']=out.severity
else:
 model_key='patchtst' if model_name=='PatchTST-AE' else 'tranad'; model_path=Path('artifacts')/model_key
 model_input=wavelet_features(df[cols]) if use_wavelet else df[cols]
 if train_now:
  config=ModelConfig(name=model_key,window=64,patch_len=8,d_model=32,nhead=4,layers=1)
  with st.spinner(f'正在训练 {model_name}...'):
   DeepPowerDetector(config).fit(model_input,epochs=3,batch_size=32).save(model_path)
  st.success(f'{model_name} 训练完成，动态阈值已校准。')
 if model_path.exists():
  detector=DeepPowerDetector.load(model_path)
  try: pred=detector.predict(model_input)
  except ValueError as error: st.error(f'当前数据与模型特征不一致：{error}。请重新训练。'); st.stop()
  out=df.copy(); out[['anomaly_score','is_anomaly','severity','drivers']]=pred[['anomaly_score','is_anomaly','severity','drivers']]; out['change_pct']=df[cols].pct_change().abs().max(axis=1).fillna(0)*100; out['alert']=out.is_anomaly; out['status']=out.severity
 else:
  st.warning(f'{model_name} 尚未训练，当前使用鲁棒统计基线。运行 train.py 生成 {model_path}。'); out=detect(df,cols,w,z,s); out['alert']=out.is_anomaly; out['status']=out.severity
al=out[out.alert]
cs=st.columns(4); cs[0].metric('监测点数',len(cols)); cs[1].metric('数据时长',f'{len(out)/60:.1f} 小时'); cs[2].metric('异常记录',len(al)); cs[3].metric('最新风险',out.status.iloc[-1])
t1,t2=st.tabs(['实时监测','告警清单'])
with t1:
 fig=go.Figure(); [fig.add_trace(go.Scatter(x=out.timestamp,y=out[k],name=k)) for k in cols[:6]]; fig.update_layout(height=450,hovermode='x unified'); st.plotly_chart(fig,use_container_width=True); st.dataframe(out.tail(20).iloc[::-1],use_container_width=True,hide_index=True)
with t2:
 view=al[['timestamp']+cols+['anomaly_score','change_pct','status']]; st.dataframe(view,use_container_width=True,hide_index=True); st.download_button('下载告警 CSV',view.to_csv(index=False).encode('utf-8-sig'),'power_alerts.csv','text/csv')
st.caption('支持鲁棒统计、PatchTST-AE、TranAD 与小波多尺度增强；深度模型使用健康数据训练并由重构误差分位数校准阈值。')
