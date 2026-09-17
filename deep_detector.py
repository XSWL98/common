"""Training, calibrated inference and conservative online adaptation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from deep_models import build_model


def make_windows(values, window, stride=1):
    if len(values) < window:
        raise ValueError(f"至少需要 {window} 条记录")
    return np.stack([values[i:i + window] for i in range(0, len(values) - window + 1, stride)])


@dataclass
class ModelConfig:
    name: str = "patchtst"
    window: int = 64
    patch_len: int = 8
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    threshold_quantile: float = 0.995


class DeepPowerDetector:
    def __init__(self, config=None, device="cpu"):
        self.config = config or ModelConfig()
        self.device = torch.device(device)
        self.model = self.scaler = self.columns = None
        self.threshold = None

    def _model_kwargs(self):
        c = self.config
        kw = dict(window=c.window, d_model=c.d_model, nhead=c.nhead, layers=c.layers)
        if c.name == "patchtst": kw["patch_len"] = c.patch_len
        return kw

    def fit(self, frame, epochs=10, batch_size=32, lr=1e-3):
        x = frame.select_dtypes(include=np.number).interpolate(limit_direction="both")
        if x.empty: raise ValueError("训练数据没有数值测点")
        self.columns = x.columns.tolist(); self.scaler = RobustScaler().fit(x)
        windows = make_windows(self.scaler.transform(x).astype("float32"), self.config.window)
        self.model = build_model(self.config.name, len(self.columns), **self._model_kwargs()).to(self.device)
        loader = DataLoader(TensorDataset(torch.from_numpy(windows)), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()
        for epoch in range(epochs):
            for (batch,) in loader:
                batch = batch.to(self.device); optimizer.zero_grad()
                pred = self.model(batch)
                if isinstance(pred, tuple): loss = .3 * (pred[0] - batch).pow(2).mean() + .7 * (pred[1] - batch).pow(2).mean()
                else: loss = (pred - batch).pow(2).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); optimizer.step()
        errors, _ = self._window_errors(windows)
        self.threshold = float(np.quantile(errors, self.config.threshold_quantile))
        return self

    def _window_errors(self, windows):
        self.model.eval(); all_scores=[]; all_feature=[]
        with torch.no_grad():
            for start in range(0, len(windows), 128):
                batch=torch.from_numpy(windows[start:start+128]).to(self.device); pred=self.model(batch)
                if isinstance(pred, tuple): pred=pred[1]
                err=(pred[:, -1] - batch[:, -1]).pow(2).cpu().numpy(); all_feature.append(err); all_scores.append(err.mean(axis=1))
        return np.concatenate(all_scores), np.concatenate(all_feature)

    def predict(self, frame):
        if self.model is None: raise RuntimeError("模型尚未训练或加载")
        missing=[c for c in self.columns if c not in frame];
        if missing: raise ValueError(f"缺少训练特征: {missing}")
        values=self.scaler.transform(frame[self.columns].interpolate(limit_direction="both")).astype("float32")
        windows=make_windows(values,self.config.window); scores, feature_errors=self._window_errors(windows)
        out=frame.copy(); out["anomaly_score"]=0.0; out["is_anomaly"]=False; out["severity"]="normal"; out["drivers"]="warmup"
        idx=out.index[self.config.window-1:]; out.loc[idx,"anomaly_score"]=scores; ratio=scores/max(self.threshold,1e-12); out.loc[idx,"is_anomaly"]=scores>=self.threshold; out.loc[idx,"severity"]=np.select([ratio>=2,ratio>=1.4,ratio>=1],["critical","high","medium"],default="normal"); out.loc[idx,"drivers"]=[self.columns[i] for i in feature_errors.argmax(axis=1)]; return out

    def partial_fit(self, frame, epochs=1, lr=1e-5):
        """Adapt only on samples currently below threshold to limit anomaly contamination."""
        result=self.predict(frame); healthy=frame.loc[~result.is_anomaly,self.columns]
        if len(healthy)<self.config.window:return self
        windows=make_windows(self.scaler.transform(healthy).astype("float32"),self.config.window); opt=torch.optim.AdamW(self.model.parameters(),lr=lr)
        self.model.train()
        for _ in range(epochs):
            for batch in DataLoader(torch.from_numpy(windows),batch_size=32,shuffle=True):
                batch=batch.to(self.device); opt.zero_grad(); pred=self.model(batch); pred=pred[1] if isinstance(pred,tuple) else pred; loss=(pred-batch).pow(2).mean(); loss.backward(); opt.step()
        return self

    def save(self, directory):
        path=Path(directory); path.mkdir(parents=True,exist_ok=True); torch.save(self.model.state_dict(),path/"model.pt"); import joblib; joblib.dump(self.scaler,path/"scaler.joblib"); (path/"metadata.json").write_text(json.dumps({"config":asdict(self.config),"columns":self.columns,"threshold":self.threshold},ensure_ascii=False,indent=2),encoding="utf-8")

    @classmethod
    def load(cls, directory, device="cpu"):
        path=Path(directory); meta=json.loads((path/"metadata.json").read_text(encoding="utf-8")); obj=cls(ModelConfig(**meta["config"]),device); obj.columns=meta["columns"]; obj.threshold=meta["threshold"]; import joblib; obj.scaler=joblib.load(path/"scaler.joblib"); obj.model=build_model(obj.config.name,len(obj.columns),**obj._model_kwargs()).to(obj.device); obj.model.load_state_dict(torch.load(path/"model.pt",map_location=obj.device,weights_only=True)); obj.model.eval(); return obj
