import tempfile
import unittest
import numpy as np
import pandas as pd
from deep_detector import DeepPowerDetector,ModelConfig
from deep_models import PatchTSTAutoencoder,TranAD
from features import wavelet_features

class DeepModelTest(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(4); n=180; i=np.arange(n); self.healthy=pd.DataFrame({'power':400+10*np.sin(i/12)+rng.normal(0,1,n),'current':1100+20*np.sin(i/12)+rng.normal(0,2,n),'voltage':380+rng.normal(0,.2,n)})
 def test_shapes(self):
  import torch; x=torch.randn(2,32,3); self.assertEqual(PatchTSTAutoencoder(3,32,8,16,4,1)(x).shape,x.shape); a,b=TranAD(3,32,16,4,1)(x); self.assertEqual(a.shape,x.shape); self.assertEqual(b.shape,x.shape)
 def test_train_save_load(self):
  for name in ('patchtst','tranad'):
   detector=DeepPowerDetector(ModelConfig(name=name,window=32,patch_len=8,d_model=16,nhead=4,layers=1,threshold_quantile=.98)).fit(self.healthy,epochs=1,batch_size=32)
   test=self.healthy.copy(); test.loc[140:145,['power','current']]+=[160,450]; result=detector.predict(test); self.assertTrue(result.loc[140:150,'is_anomaly'].any())
   with tempfile.TemporaryDirectory() as path:
    detector.save(path); loaded=DeepPowerDetector.load(path); self.assertEqual(len(loaded.predict(test)),len(test))
 def test_wavelet_features(self):
  enhanced=wavelet_features(self.healthy); self.assertGreater(enhanced.shape[1],self.healthy.shape[1]); self.assertFalse(enhanced.isna().any().any())
if __name__=='__main__':unittest.main()
