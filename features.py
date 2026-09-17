import numpy as np
import pandas as pd
import pywt

def wavelet_features(frame, wavelet="db4", level=2):
    """Add aligned trend and high-frequency energy for every numeric sensor."""
    out=frame.copy()
    for col in frame.select_dtypes(include=np.number).columns:
        values=frame[col].interpolate(limit_direction="both").to_numpy(float)
        max_level=pywt.dwt_max_level(len(values),pywt.Wavelet(wavelet).dec_len)
        coeffs=pywt.wavedec(values,wavelet,level=min(level,max_level))
        approx=pywt.waverec([coeffs[0]]+[np.zeros_like(c) for c in coeffs[1:]],wavelet)[:len(values)]
        detail=values-approx
        out[f"{col}__trend"]=approx
        out[f"{col}__hf_energy"]=pd.Series(detail**2).rolling(8,min_periods=1).mean().to_numpy()
    return out
