# تدريب الموديل المرشح (عتبة 0.58) من الصفر.  Usage:
#   python train_model.py [--csv ../data/aggtrades_1m_long.csv] [--train-end 2026-08-31T23:59] [--out ../model]
# ينتج: BTC_30m_model_058.joblib  و  BTC_30m_model_058.npz  (نسخة لا تحتاج sklearn)
import os, sys, argparse, joblib, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from common import build_features_from_csv
from sklearn.ensemble import HistGradientBoostingClassifier as H

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(here, "..", "data", "aggtrades_1m_long.csv"))
ap.add_argument("--train-end", default=None, help="آخر وقت UTC يدخل في التدريب (افتراضي: كل البيانات)")
ap.add_argument("--out", default=os.path.join(here, "..", "model"))
a = ap.parse_args()

PARAMS = dict(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=2.0, min_samples_leaf=300,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=42)

df, F = build_features_from_csv(a.csv); close = df["close"].astype(float)
ok = (df["ts"].shift(-30) - df["ts"]) == pd.Timedelta(minutes=30)          # شمعة بعد 30 دقيقة موجودة فعلاً
y = (close.shift(-30) > close).where(ok)                                    # الهدف: السعر بعد 30 دقيقة أعلى من الآن؟
pk = F.copy(); pk["ts"] = df["ts"]; pk["y"] = y
pk = pk.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
end = pd.Timestamp(a.train_end, tz="UTC") if a.train_end else pk["ts"].max()
m = pk["ts"] <= end - pd.Timedelta(minutes=31)                              # فاصل 31 دقيقة حتى لا يرى الموديل المستقبل
feats = list(F.columns)
model = H(**PARAMS).fit(pk.loc[m, feats], pk.loc[m, "y"].astype(int))
os.makedirs(a.out, exist_ok=True)
joblib.dump({"model": model, "features": feats}, os.path.join(a.out, "BTC_30m_model_058.joblib"))

# --- تصدير نسخة numpy (مستقلة عن إصدار sklearn) ---
trees = [t[0] for t in model._predictors]; nodes = [t.nodes for t in trees]; cat = np.concatenate(nodes)
off = np.cumsum([0] + [len(n) for n in nodes])
np.savez_compressed(os.path.join(a.out, "BTC_30m_model_058.npz"), features=np.array(feats), offsets=off.astype(np.int64),
    baseline=np.float64(np.ravel(model._baseline_prediction)[0]), feature_idx=cat["feature_idx"].astype(np.int64), thr=cat["num_threshold"].astype(np.float64),
    left=cat["left"].astype(np.int64), right=cat["right"].astype(np.int64), value=cat["value"].astype(np.float64),
    miss_left=cat["missing_go_to_left"].astype(np.int8), is_leaf=cat["is_leaf"].astype(np.int8))
print("trained rows:", int(m.sum()), "| last training candle:", pk.loc[m, "ts"].max(), "| trees:", len(trees), "| saved to", os.path.abspath(a.out))
