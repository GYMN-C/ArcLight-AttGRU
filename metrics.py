import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def _agg_by_record(df: pd.DataFrame):
    """
    将 test_preds.csv 聚合到录音段级：
    - y_true: 取众数（若每段标签一致则等价）
    - y_prob: 取均值（或你可以改为 max/median）
    - y_pred: 由均值概率再阈值化（默认0.5）
    """
    grouped = df.groupby('record_id', as_index=False).agg(
        y_true=('y_true', lambda x: int(round(np.mean(x)))),
        y_prob=('y_prob', 'mean')
    )
    grouped['y_pred'] = (grouped['y_prob'] >= 0.5).astype(int)
    return grouped

def compute_metrics_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    if not {'record_id','y_true','y_prob','y_pred'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: record_id, y_true, y_prob, y_pred")
    # 录音段级聚合
    g = _agg_by_record(df)
    y_true = g['y_true'].values
    y_pred = g['y_pred'].values
    y_prob = g['y_prob'].values

    metrics = {
        'ACC': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan'),
        'n_records': len(g)
    }
    return metrics, g
