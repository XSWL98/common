import argparse
import pandas as pd
from deep_detector import DeepPowerDetector,ModelConfig
from features import wavelet_features

def main():
 p=argparse.ArgumentParser(description='Train PatchTST-AE or TranAD on healthy power data'); p.add_argument('--csv',required=True); p.add_argument('--model',choices=['patchtst','tranad'],default='patchtst'); p.add_argument('--out',default='artifacts/model'); p.add_argument('--window',type=int,default=64); p.add_argument('--epochs',type=int,default=10); p.add_argument('--wavelet',action='store_true'); a=p.parse_args()
 frame=pd.read_csv(a.csv).select_dtypes(include='number')
 if a.wavelet: frame=wavelet_features(frame)
 DeepPowerDetector(ModelConfig(name=a.model,window=a.window)).fit(frame,epochs=a.epochs).save(a.out)
 print(f'saved {a.model} to {a.out}')
if __name__=='__main__':main()
