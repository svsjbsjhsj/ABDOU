import os, json, joblib, requests, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
PAYOUT=0.83
def build_features(df):
    """Build exactly the same feature columns for historical CSV and live candles."""
    df=df.copy()
    df["ts"]=pd.to_datetime(df["ts"],utc=True,errors="coerce")
    df=df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for c in ["open","high","low","close","n_trades","buy_vol","sell_vol","quote_vol","vwap"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    close=df["close"].astype(float); f=pd.DataFrame(index=df.index)
    for n in [1,2,3,5,10,15,30,60,120,240]: f[f"ret_{n}"]=np.log(close/close.shift(n))
    r=np.log(close/close.shift(1))
    for n in [5,15,30,60,120,240]: f[f"vol_{n}"]=r.rolling(n).std()
    buy=df["buy_vol"].astype(float); sell=df["sell_vol"].astype(float)
    ofi=(buy-sell)/(buy+sell+1e-12)
    for n in [1,3,5,15,30,60,240]: f[f"ofi_{n}"]=ofi.rolling(n).mean()
    f["log_n_trades"]=np.log1p(df["n_trades"].clip(lower=0)); f["log_quote_vol"]=np.log1p(df["quote_vol"].clip(lower=0))
    f["vwap_dev"]=(close-df["vwap"].astype(float))/(df["vwap"].astype(float)+1e-12)
    f["range_pct"]=(df["high"]-df["low"])/(close+1e-12); f["body_pct"]=(df["close"]-df["open"])/(close+1e-12)
    f["close_position"]=(df["close"]-df["low"])/((df["high"]-df["low"])+1e-12)
    m=df["ts"].dt.hour*60+df["ts"].dt.minute
    f["hour_sin"]=np.sin(2*np.pi*m/1440); f["hour_cos"]=np.cos(2*np.pi*m/1440); f["dow"]=df["ts"].dt.dayofweek
    return df, f.replace([np.inf,-np.inf],np.nan)

def build_features_from_csv(path):
    return build_features(pd.read_csv(path))

def report(data, probs, threshold, outpath):
    conf=np.maximum(probs,1-probs); pred=probs>=0.5; eligible=np.where(conf>=threshold)[0]
    wins=[]; inds=[]; last=-10**9
    for i in eligible:
        if i < last+30: continue
        inds.append(i); wins.append(int(pred[i] == bool(data["y"].iloc[i]))); last=i
    n=len(wins); wr=float(np.mean(wins)) if n else float("nan")
    net=float(np.sum(np.array(wins)*PAYOUT-(1-np.array(wins)))) if n else 0.0
    days=max((data["ts"].iloc[-1]-data["ts"].iloc[0]).total_seconds()/86400,1e-9)
    print(f"threshold={threshold:.2f}, payout=83%, horizon=30m")
    print(f"trades={n}, win_rate={wr:.4%}, trades/day={n/days:.3f}, net=${net:.2f} at $1 stake")
    return {"trades":n,"win_rate":wr,"trades_per_day":n/days,"net_units":net,"threshold":threshold,"payout":PAYOUT}
