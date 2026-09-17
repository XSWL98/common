# 智序工业电力异常检测

运行：pip install -r requirements.txt，然后 streamlit run app.py。API：uvicorn api:app --reload

上传 CSV 需包含时间列和数值测点列。系统用滚动中位数+MAD动态基线，检测突增突降。train.py可训练模型，evaluate.py输出F1、ROC-AUC、PR-AUC。

## 深度模型

系统包含 CPU 友好的 PatchTST-AE 和 TranAD 风格两阶段自条件 Transformer。页面可选择模型并用当前数据训练；训练集应尽量使用健康运行片段。阈值按训练重构误差的 99.5% 分位数自动校准。

命令行训练：

    python train.py --csv healthy.csv --model patchtst --out artifacts/patchtst --epochs 10 --wavelet
    python train.py --csv healthy.csv --model tranad --out artifacts/tranad --epochs 10

小波增强会为每个测点生成低频趋势和高频能量特征。在线更新接口只使用当前判定为健康的数据，以减少异常样本污染模型。

运行测试：python -m unittest -v test_system.py test_deep_models.py
