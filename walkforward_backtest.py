# إعادة الاختبار الشهري (walk-forward): كل شهر يُختبر بموديل تدرّب فقط على ما قبله.
#   python walkforward_backtest.py --start 2024-07 --end 2026-08      (حوالي 10 دقائق)
#   python walkforward_backtest.py --start 2026-07 --end 2026-08      (اختبار سريع)
import os, sys, argparse, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from common import build_features_from_csv
from sklearn.ensemble import HistGradientBoostingClassifier as H
ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=os.path.join(here, "..", "data", "aggtrades_1m_long.csv"))
ap.add_argument("--start", default="2024-07"); ap.add_argument("--end", default="2026-08"); ap.add_argument("--thresholds", default="0.58,0.60")
ap.add_argument("--out", default=os.path.join(here, "..", "results")); a = ap.parse_args()
PAY = 0.83; BE = 1 / (1 + PAY)
PARAMS = dict(max_iter=100, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=2.0, min_samples_leaf=300,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=42)
df, F = build_features_from_csv(a.csv); close = df["close"].astype(float)
ok = (df["ts"].shift(-30) - df["ts"]) == pd.Timedelta(minutes=30); y = (close.shift(-30) > close).where(ok)
pk = F.copy(); pk["ts"] = df["ts"]; pk["y"] = y; pk = pk.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
feats = list(F.columns); ts = pk["ts"]; res = []
for m in pd.period_range(a.start, a.end, freq="M"):
    ms = pd.Timestamp(str(m.start_time), tz="UTC"); me = pd.Timestamp(str(m.end_time.floor("min")), tz="UTC")
    tr = ts < ms - pd.Timedelta(minutes=31); te = (ts >= ms) & (ts <= me)
    if te.sum() == 0: continue
    mdl = H(**PARAMS).fit(pk.loc[tr, feats], pk.loc[tr, "y"].astype(int))
    d = pk.loc[te, ["ts", "y"]].copy(); d["p"] = mdl.predict_proba(pk.loc[te, feats])[:, 1]; res.append(d); print("done", m, flush=True)
P = pd.concat(res).reset_index(drop=True); P["conf"] = np.maximum(P.p, 1 - P.p); P["ok"] = ((P.p >= 0.5) == P.y.astype(bool)).astype(int)
P["month"] = P.ts.dt.strftime("%Y-%m")
def pick(d, th):                                  # صفقة واحدة كل 30 دقيقة
    idx = np.where(d.conf.values >= th)[0]; keep = []; last = -10**9
    for i in idx:
        if i < last + 30: continue
        keep.append(i); last = i
    return d.iloc[keep]
os.makedirs(a.out, exist_ok=True)
for th in [float(x) for x in a.thresholds.split(",")]:
    x = pick(P, th); g = x.groupby("month").ok.agg(trades="count", wins="sum"); g["win_rate_%"] = (g.wins / g.trades * 100).round(1)
    g["net_units"] = (g.wins * PAY - (g.trades - g.wins)).round(1); g["cumulative"] = g.net_units.cumsum().round(1)
    g.to_csv(os.path.join(a.out, f"walkforward_monthly_th{th:.2f}.csv")); print(f"\n=== threshold {th:.2f} (payout {PAY:.0%}, break-even {BE:.2%}) ==="); print(g.to_string())
    w = int(x.ok.sum()); n = len(x); print(f"TOTAL trades={n} win_rate={w/n*100:.1f}% net={w*PAY-(n-w):+.1f} units")
