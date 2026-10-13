# ===== BTC 30m | لوحة التوقع الملونة (Colab) - بدون sklearn =====
import os, time
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
from zoneinfo import ZoneInfo

MODEL_FILE = "BTC_30m_model_058.npz"
THRESHOLD  = 0.58
WATCH      = False                  # True = يبقى يراقب كل دقيقة (وقّفه بزر Stop)
STAKE      = 100                    # مبلغ الصفقة (للحساب فقط)
PAYOUT     = 0.83                   # ربح المنصة
USE_DRIVE  = False                  # True = يحفظ السجل في Google Drive (لا يضيع إذا انقطع Colab)
MAX_LATE   = 120                    # لا يسجل الإشارة إذا تأخرت أكثر من هذا (ثواني)
LOG_PATH   = "btc_signals_log.csv"
TG_TOKEN   = ""                     # توكن البوت من @BotFather (اتركه فارغاً = بدون إشعارات)
TG_CHAT_ID = ""                     # رقم محادثتك مع البوت
if USE_DRIVE:
    from google.colab import drive; drive.mount("/content/drive"); LOG_PATH = "/content/drive/MyDrive/btc_signals_log.csv"
MA = ZoneInfo("Africa/Casablanca")

# --- رفع ملف الموديل إذا لم يكن موجوداً ---
if not os.path.exists(MODEL_FILE):
    try:
        from google.colab import files
        up = files.upload(); MODEL_FILE = list(up)[0]
    except ImportError:
        raise SystemExit("ضع ملف الموديل بجانب هذا الكود")

class NpModel:                       # يقرأ الموديل بدون أي مكتبة ML
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.features = [str(x) for x in z["features"]]; self.off = z["offsets"]; self.base = float(z["baseline"])
        self.fi, self.thr, self.l, self.r, self.val = z["feature_idx"], z["thr"], z["left"], z["right"], z["value"]
        self.ml, self.leaf = z["miss_left"], z["is_leaf"]
    def p_up(self, x):
        raw = self.base
        for t in range(len(self.off) - 1):
            o = self.off[t]; n = 0
            while not self.leaf[o + n]:
                k = o + n; v = x[self.fi[k]]
                go_left = (self.ml[k] == 1) if np.isnan(v) else (v <= self.thr[k])
                n = self.l[k] if go_left else self.r[k]
            raw += self.val[o + n]
        return 1.0 / (1.0 + np.exp(-raw))
model = NpModel(MODEL_FILE); FEATS = model.features

# --- حساب المؤشرات (نفس ما تدرّب عليه الموديل) ---
def build_features(df):
    df = df.copy(); df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    close = df["close"].astype(float); f = pd.DataFrame(index=df.index)
    for n in [1,2,3,5,10,15,30,60,120,240]: f[f"ret_{n}"] = np.log(close/close.shift(n))
    r = np.log(close/close.shift(1))
    for n in [5,15,30,60,120,240]: f[f"vol_{n}"] = r.rolling(n).std()
    buy, sell = df["buy_vol"].astype(float), df["sell_vol"].astype(float)
    ofi = (buy-sell)/(buy+sell+1e-12)
    for n in [1,3,5,15,30,60,240]: f[f"ofi_{n}"] = ofi.rolling(n).mean()
    f["log_n_trades"] = np.log1p(df["n_trades"].clip(lower=0)); f["log_quote_vol"] = np.log1p(df["quote_vol"].clip(lower=0))
    f["vwap_dev"] = (close-df["vwap"])/(df["vwap"]+1e-12)
    f["range_pct"] = (df["high"]-df["low"])/(close+1e-12); f["body_pct"] = (df["close"]-df["open"])/(close+1e-12)
    f["close_position"] = (df["close"]-df["low"])/((df["high"]-df["low"])+1e-12)
    m = df["ts"].dt.hour*60 + df["ts"].dt.minute
    f["hour_sin"] = np.sin(2*np.pi*m/1440); f["hour_cos"] = np.cos(2*np.pi*m/1440); f["dow"] = df["ts"].dt.dayofweek
    return df, f.replace([np.inf,-np.inf], np.nan)

# --- جلب آخر 300 شمعة دقيقة من Binance (data-api أولاً لأن Colab في أمريكا) ---
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"]
def fetch_candles():
    rows, err = None, None
    for h in HOSTS:
        try:
            r = requests.get(h+"/api/v3/klines", params={"symbol":"BTCUSDT","interval":"1m","limit":300}, timeout=20)
            r.raise_for_status(); rows = r.json(); break
        except Exception as e: err = e
    if rows is None: raise RuntimeError(f"تعذر جلب البيانات من Binance: {err!r}")
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","n_trades","taker_buy_base","taker_buy_quote","ignore"]
    raw = pd.DataFrame(rows, columns=cols)
    raw = raw[raw["close_time"].astype("int64") < int(time.time()*1000)].copy()      # الشموع المغلقة فقط
    for c in ["open","high","low","close","volume","quote_volume","n_trades","taker_buy_base"]: raw[c] = pd.to_numeric(raw[c])
    d = pd.DataFrame({"ts": pd.to_datetime(raw["open_time"], unit="ms", utc=True), "open": raw["open"], "high": raw["high"], "low": raw["low"],
                      "close": raw["close"], "n_trades": raw["n_trades"], "buy_vol": raw["taker_buy_base"],
                      "sell_vol": (raw["volume"]-raw["taker_buy_base"]).clip(lower=0), "quote_vol": raw["quote_volume"]})
    d["vwap"] = d["quote_vol"]/(d["buy_vol"]+d["sell_vol"]+1e-12)
    return d

def predict(candles):
    d, F = build_features(candles)
    F = F[FEATS].replace([np.inf,-np.inf], np.nan).dropna()
    if F.empty or F.index[-1] != d.index[-1]: return None
    i = F.index[-1]; p = float(model.p_up(F.loc[i, FEATS].values.astype(float)))
    return dict(open=d.loc[i,"ts"], p=p, conf=max(p,1-p), price=float(d.loc[i,"close"]), hist=d.tail(120).reset_index(drop=True))

GREEN, RED, GRAY, ORANGE = "#16a34a", "#dc2626", "#4b5563", "#d97706"
BE = 1/(1+PAYOUT)
COLS = ["signal_open_utc","direction","p_up","confidence","entry_price","result","exit_price"]

# ---------- إشعارات تيليجرام ----------
TG_STATE = "off" if not (TG_TOKEN and TG_CHAT_ID) else "ok"
def notify(text):
    global TG_STATE
    if not (TG_TOKEN and TG_CHAT_ID): return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id": TG_CHAT_ID, "text": text}, timeout=10)
        TG_STATE = "ok" if r.status_code == 200 else "fail"
    except Exception: TG_STATE = "fail"

# ---------- سجل الإشارات ----------
def read_log():
    if os.path.exists(LOG_PATH): return pd.read_csv(LOG_PATH, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLS)

def get_close_at(open_ts):           # سعر إغلاق الشمعة التي تبدأ عند open_ts
    ms = int(open_ts.timestamp()*1000)
    for h in HOSTS:
        try:
            r = requests.get(h+"/api/v3/klines", params={"symbol":"BTCUSDT","interval":"1m","startTime":ms,"limit":1}, timeout=20)
            r.raise_for_status(); k = r.json()
            if k and int(k[0][0]) == ms and int(k[0][6]) < int(time.time()*1000): return float(k[0][4])
            return None
        except Exception: continue
    return None

def update_results():
    df = read_log(); changed = False
    for i, row in df.iterrows():
        if row["result"]: continue
        exit_open = pd.Timestamp(row["signal_open_utc"]) + pd.Timedelta(minutes=30)
        if pd.Timestamp.now(tz="UTC") < exit_open + pd.Timedelta(minutes=1): continue
        ex = get_close_at(exit_open)
        if ex is None: continue
        en = float(row["entry_price"])
        res = "DRAW" if ex == en else ("WIN" if ((ex > en) == (row["direction"] == "BUY")) else "LOSS")
        df.loc[i, "exit_price"] = str(ex); df.loc[i, "result"] = res; changed = True
        notify({"WIN":"✅ ربح","LOSS":"❌ خسارة","DRAW":"➖ تعادل"}[res] + f" — {row['direction']} {pd.Timestamp(row['signal_open_utc']).tz_convert(MA):%H:%M}\n{en:,.2f} → {ex:,.2f}")
    if changed: df.to_csv(LOG_PATH, index=False)
    return df

def log_signal(info, direction):
    row = pd.DataFrame([[info["open"].isoformat(), direction, f"{info['p']:.4f}", f"{info['conf']:.4f}", str(info["price"]), "", ""]], columns=COLS)
    row.to_csv(LOG_PATH, mode="a", header=not os.path.exists(LOG_PATH), index=False)

def history_html(df):
    t = lambda z: z.tz_convert(MA).strftime("%H:%M")
    done = df[df["result"].isin(["WIN","LOSS"])]; n = len(done); w = int((done["result"]=="WIN").sum()); l = n-w
    pend = int((df["result"]=="").sum())
    if n:
        wr = w/n; units = w*PAYOUT-l; wc = "#4ade80" if wr >= BE else "#f87171"; uc = "#4ade80" if units >= 0 else "#f87171"
        stats = [("الصفقات المنتهية", f"{n} &nbsp;(✅{w} / ❌{l})", "#e5e7eb"), (f"نسبة الفوز (التعادل {BE*100:.1f}%)", f"{wr*100:.1f}%", wc),
                 ("الصافي بالوحدات", f"{units:+.2f}", uc), (f"الصافي بالدولار (مبلغ {STAKE})", f"{units*STAKE:+,.0f}$", uc)]
    else:
        stats = [("الصفقات المنتهية", "0", "#9ca3af"), ("نسبة الفوز", "—", "#9ca3af"), ("الصافي", "—", "#9ca3af")]
    if pend: stats.insert(1, ("جارية", f"⏳ {pend}", "#fbbf24"))
    st = "".join(f'<tr><td style="padding:6px 10px;color:#cbd5e1">{a}</td><td style="padding:6px 10px;text-align:left;font-weight:700;color:{c};direction:ltr">{b}</td></tr>' for a,b,c in stats)
    rows = ""
    for _, r in df.tail(8).iloc[::-1].iterrows():
        o = pd.Timestamp(r["signal_open_utc"]); en = float(r["entry_price"]); res = r["result"]
        d = "🟢 BUY" if r["direction"] == "BUY" else "🔴 SELL"
        if res == "WIN": rc, bg = "✅ ربح", "#12301f"
        elif res == "LOSS": rc, bg = "❌ خسارة", "#35171a"
        elif res == "DRAW": rc, bg = "➖ تعادل", "#1b2030"
        else: rc, bg = f"⏳ حتى {t(o + pd.Timedelta(minutes=31))}", "#1b2030"
        ex = f" → {float(r['exit_price']):,.0f}" if r["exit_price"] else ""
        rows += (f'<tr style="background:{bg}"><td style="padding:7px 8px;color:#cbd5e1;direction:ltr">{t(o + pd.Timedelta(minutes=1))}</td>'
                 f'<td style="padding:7px 8px;color:white;font-weight:700">{d}</td><td style="padding:7px 8px;color:#9ca3af;direction:ltr">{float(r["confidence"])*100:.1f}%</td>'
                 f'<td style="padding:7px 8px;color:#9ca3af;direction:ltr;font-size:12px">{en:,.0f}{ex}</td><td style="padding:7px 8px;color:white;font-weight:700">{rc}</td></tr>')
    if not rows: rows = '<tr><td colspan="5" style="padding:12px;text-align:center;color:#6b7280">لا توجد إشارات مسجلة بعد</td></tr>'
    warn = f'<div style="text-align:center;color:#fbbf24;font-size:12px;padding:6px">عدد الصفقات قليل للحكم — انتظر 300 صفقة على الأقل</div>' if n < 300 else ""
    return f"""<div dir="rtl" style="background:#14171f;border-radius:16px;padding:12px;max-width:560px;margin:10px auto;font-family:system-ui,Arial,sans-serif;">
      <div style="color:white;font-weight:800;font-size:17px;text-align:center;padding-bottom:6px">📒 سجل الإشارات</div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;background:#1b2030;border-radius:10px;overflow:hidden">{st}</table>{warn}
      <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;border-radius:10px;overflow:hidden">
      <tr style="color:#6b7280;font-size:11px"><td style="padding:4px 8px">الوقت</td><td>الإشارة</td><td>الثقة</td><td>الدخول</td><td>النتيجة</td></tr>{rows}</table></div>"""

def draw_chart(info, sig):
    h = info["hist"]; n = len(h); x = np.arange(n)
    up = h["close"] >= h["open"]; col = np.where(up, "#22c55e", "#ef4444")
    plt.close("all")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={"height_ratios": [4, 0.9], "hspace": 0.35}, facecolor="#14171f")
    for a_ in (ax, ax2): a_.set_facecolor("#14171f"); [sp.set_color("#374151") for sp in a_.spines.values()]; a_.tick_params(colors="#9ca3af", labelsize=8)
    ax.vlines(x, h["low"], h["high"], colors=col, linewidth=0.9)
    body_h = (h["close"]-h["open"]).abs().clip(lower=h["close"].iloc[-1]*1e-6)
    ax.bar(x, body_h, bottom=np.minimum(h["open"], h["close"]), width=0.65, color=col)
    last = h["close"].iloc[-1]; ax.axhline(last, color="#9ca3af", ls=":", lw=0.8)
    ax.text(n-0.5, last, f" {last:.2f}", color="#e5e7eb", fontsize=8, va="center")
    if sig == "BUY":  ax.scatter([n-1], [h["low"].iloc[-1]*0.9998], marker="^", s=170, color="#22c55e", zorder=5); ax.set_title("BUY signal", color="#22c55e", fontsize=11)
    elif sig == "SELL": ax.scatter([n-1], [h["high"].iloc[-1]*1.0002], marker="v", s=170, color="#ef4444", zorder=5); ax.set_title("SELL signal", color="#ef4444", fontsize=11)
    else: ax.set_title("BTCUSDT 1m - last 2 hours (no signal)", color="#9ca3af", fontsize=10)
    ticks = list(range(0, n, 20)); ax.set_xticks(ticks)
    ax.set_xticklabels([h["ts"].iloc[t].tz_convert(MA).strftime("%H:%M") for t in ticks]); ax.set_xlim(-1, n+6); ax.grid(color="#1f2937", lw=0.6)
    pu = info["p"]; ax2.barh([0], [pu], color="#22c55e", height=0.6); ax2.barh([0], [1-pu], left=[pu], color="#ef4444", height=0.6)
    for v, lab in ((1-THRESHOLD, f"{(1-THRESHOLD)*100:.0f}%"), (0.5, "50%"), (THRESHOLD, f"{THRESHOLD*100:.0f}%")):
        ax2.axvline(v, color="#fbbf24" if v != 0.5 else "white", ls="--", lw=1.2); ax2.text(v, 0.48, lab, color="#fbbf24" if v != 0.5 else "white", fontsize=8, ha="center")
    ax2.text(min(max(pu/2, 0.04), 0.96), 0, f"UP {pu*100:.1f}%", color="white", fontsize=9, ha="center", va="center", weight="bold") if pu > 0.18 else None
    ax2.text(min(max(pu+(1-pu)/2, 0.04), 0.96), 0, f"DOWN {(1-pu)*100:.1f}%", color="white", fontsize=9, ha="center", va="center", weight="bold") if pu < 0.82 else None
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.5, 0.75); ax2.set_yticks([]); ax2.set_xticks([0, .25, .5, .75, 1]); ax2.set_xticklabels(["0%","25%","50%","75%","100%"])
    plt.show()

def show(info):
    t = lambda z: z.tz_convert(MA).strftime("%H:%M:%S")
    df = update_results()
    close_t = info["open"] + pd.Timedelta(minutes=1); exp = info["open"] + pd.Timedelta(minutes=31)
    late = (pd.Timestamp.now(tz="UTC") - close_t).total_seconds()
    already = bool(len(df)) and (df["signal_open_utc"] == info["open"].isoformat()).any()
    last_sig = pd.Timestamp(df["signal_open_utc"].iloc[-1]) if len(df) else None
    locked = (last_sig is not None) and (info["open"] < last_sig + pd.Timedelta(minutes=30)) and not already
    strong = info["conf"] >= THRESHOLD
    if not strong: sig, title, color, note = None, "⚪ لا توجد إشارة", GRAY, f"الثقة أقل من {THRESHOLD*100:.0f}%"
    elif locked:   sig, title, color, note = None, "🟠 إشارة قوية — صفقة سابقة جارية", ORANGE, "تجاهلها حتى تنتهي الصفقة السابقة"
    elif info["p"] >= 0.5: sig, title, color, note = "BUY", "🟢 BUY — شراء (صعود)", GREEN, "ادخل الآن، الاستحقاق 30 دقيقة"
    else:          sig, title, color, note = "SELL", "🔴 SELL — بيع (نزول)", RED, "ادخل الآن، الاستحقاق 30 دقيقة"
    if sig and not already and late <= MAX_LATE:
        log_signal(info, sig); df = read_log()
        notify(("🟢 BUY شراء" if sig == "BUY" else "🔴 SELL بيع") + f" — BTC 30 دقيقة\nالثقة {info['conf']*100:.1f}%\nادخل الآن • ينتهي {exp.tz_convert(MA):%H:%M} (المغرب)\nالسعر المرجعي {info['price']:,.2f}")
    elif sig and not already: note = "⚠️ تأخرت على هذه الإشارة — لن تُسجّل في السجل"
    late_c = "#f87171" if late > MAX_LATE else "#e5e7eb"
    rows = [("🕐 آخر شمعة أغلقت", f"{t(close_t)} <small>(المغرب)</small>", "#e5e7eb"),
            ("⏱ مرّ عليها", f"{late:.0f} ثانية" + (" ⚠️ متأخر" if late > MAX_LATE and sig else ""), late_c),
            ("💰 السعر", f"{info['price']:,.2f}", "#e5e7eb"),
            ("📈 احتمال الصعود", f"{info['p']*100:.2f}%", "#4ade80"),
            ("📉 احتمال النزول", f"{(1-info['p'])*100:.2f}%", "#f87171"),
            ("🎯 الثقة", f"{info['conf']*100:.2f}% <small dir='rtl' style='unicode-bidi:isolate'>(العتبة {THRESHOLD*100:.0f}%)</small>", "#4ade80" if strong else "#9ca3af")]
    if sig: rows.append(("⏳ ينتهي الاستحقاق", f"{t(exp)} <small>(المغرب)</small>", "#fbbf24"))
    tg_line = {"off": "🔕 إشعارات تيليجرام: غير مفعّلة", "ok": "🔔 إشعارات تيليجرام: مفعّلة", "fail": "⚠️ فشل إرسال إشعار تيليجرام (راجع التوكن والرقم)"}[TG_STATE]
    tr = "".join(f'<tr style="background:{"#1b2030" if i%2==0 else "#171b27"}"><td style="padding:10px 12px;color:#cbd5e1">{a}</td><td style="padding:10px 12px;text-align:left;font-weight:700;color:{c};direction:ltr">{b}</td></tr>' for i,(a,b,c) in enumerate(rows))
    html = f"""<div dir="rtl" style="background:#14171f;border-radius:16px;padding:12px;max-width:560px;margin:auto;font-family:system-ui,Arial,sans-serif;">
      <div style="background:{color};border-radius:12px;padding:14px;text-align:center;color:white;font-size:24px;font-weight:800;">{title}</div>
      <div style="text-align:center;color:#e5e7eb;padding:8px 0;font-size:15px;">{note}</div>
      <table style="width:100%;border-collapse:collapse;font-size:15px;border-radius:10px;overflow:hidden">{tr}</table>
      <div style="text-align:center;color:#6b7280;font-size:12px;padding-top:8px">توقع تجريبي، وليس ضماناً للنتيجة.<br>{tg_line}</div></div>"""
    display(HTML(html)); draw_chart(info, sig); display(HTML(history_html(df)))

if not WATCH:
    show(predict(fetch_candles()))
else:
    notify("✅ بدأت المراقبة. سأرسل لك أي إشارة مباشرة.")
    last_seen = None
    while True:
        try:
            info = predict(fetch_candles())
            if info and info["open"] != last_seen:
                last_seen = info["open"]; clear_output(wait=True); show(info)
        except Exception as e: print("خطأ مؤقت:", repr(e))
        time.sleep(10)
