import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from power_ad import PowerAnomalyDetector
st.set_page_config(page_title='智序工业电力异常检测',page_icon='⚡',layout='wide')
@st.cache_data
def demo(n=1440):
 r=np.random.default_rng(7); t=pd.date_range(end=pd.Timestamp.now().floor('min'),periods=n,freq='min'); i=np.arange(n); load=420+90*np.sin(i*2*np.pi/1440)+r.normal(0,8,n); v=380+r.normal(0,.7,n); cur=load/(v*.92)*1000+r.normal(0,4,n); f=50+r.normal(0,.025,n); pf=.92+r.normal(0,.012,n)
 for x,d in [(430,180),(875,-210),(1180,130)]: load[x:x+8]+=d; cur[x:x+8]+=d/.35
 return pd.DataFrame({'timestamp':t,'active_power_kw':load,'voltage_v':v,'current_a':cur,'frequency_hz':f,'power_factor':pf})
def detect(df,cols,w,z,s):
 return PowerAnomalyDetector(w,z,s).predict(df)
 x=df[cols].astype(float); med=x.rolling(w,min_periods=max(5,w//3)).median(); mad=(x-med).abs().rolling(w,min_periods=max(5,w//3)).median(); scale=(1.4826*mad).replace(0,np.nan).fillna(x.std().replace(0,1)); score=((x-med).abs()/scale).max(axis=1); delta=x.pct_change(max(1,w//4)).abs().max(axis=1)*100; o=df.copy(); o['anomaly_score']=score.round(2); o['change_pct']=delta.round(2); o['alert']=(score>=z)|(delta>=s); o['status']=np.select([score>=z*2,score>=z,delta>=s],['严重','高','中'],default='正常'); return o
st.title('⚡ 智序·工业电力异常检测'); st.caption('多元时序 · 动态阈值 · 突增突降识别')
with st.sidebar:
 up=st.file_uploader('上传 CSV',type=['csv']); w=st.slider('基线窗口（分钟）',5,120,30,5); z=st.slider('鲁棒异常阈值',2.,8.,4.,.5); s=st.slider('突变阈值（%）',2.,50.,12.,1.)
df=demo()
if up:
 raw=pd.read_csv(up); tc=next((c for c in raw.columns if c.lower() in ['timestamp','time','datetime','date']),raw.columns[0]); raw[tc]=pd.to_datetime(raw[tc],errors='coerce'); raw=raw.rename(columns={tc:'timestamp'}).dropna(subset=['timestamp']); nc=raw.select_dtypes(include=np.number).columns.tolist(); df=raw[['timestamp']+nc] if nc else df
cols=df.select_dtypes(include=np.number).columns.tolist(); out=detect(df,cols,w,z,s); al=out[out.alert]
cs=st.columns(4); cs[0].metric('监测点数',len(cols)); cs[1].metric('数据时长',f'{len(out)/60:.1f} 小时'); cs[2].metric('异常记录',len(al)); cs[3].metric('最新风险',out.status.iloc[-1])
t1,t2=st.tabs(['实时监测','告警清单'])
with t1:
 fig=go.Figure(); [fig.add_trace(go.Scatter(x=out.timestamp,y=out[k],name=k)) for k in cols[:6]]; fig.update_layout(height=450,hovermode='x unified'); st.plotly_chart(fig,use_container_width=True); st.dataframe(out.tail(20).iloc[::-1],use_container_width=True,hide_index=True)
with t2:
 view=al[['timestamp']+cols+['anomaly_score','change_pct','status']]; st.dataframe(view,use_container_width=True,hide_index=True); st.download_button('下载告警 CSV',view.to_csv(index=False).encode('utf-8-sig'),'power_alerts.csv','text/csv')
st.caption('滚动中位数 + MAD 鲁棒基线与变化率规则；生产可接入 Kafka/OPC-UA，并替换为 TranAD/PatchTST-AE。')
