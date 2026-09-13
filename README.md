# 智序工业电力异常检测

运行：pip install -r requirements.txt，然后 streamlit run app.py。API：uvicorn api:app --reload

上传 CSV 需包含时间列和数值测点列。系统用滚动中位数+MAD动态基线，检测突增突降。train.py可训练模型，evaluate.py输出F1、ROC-AUC、PR-AUC。
