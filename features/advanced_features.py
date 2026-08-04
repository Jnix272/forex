"""
features/advanced_features.py
All advanced feature groups: L2 OB, tick imbalance, session clocks,
correlation regime, Hurst, options proxies, COT.
"""
import numpy as np
import pandas as pd
import polars as pl

from config.settings import PATHS


def synthetic_orderbook(bars,n_levels=10):
    n=len(bars); rng=np.random.default_rng(42)
    mid=bars["close"].values.astype(float)
    spd=bars["spread_avg"].values if "spread_avg" in bars.columns else np.full(n,5e-5)
    sp=spd.reshape(-1,1)*np.arange(1,n_levels+1)/2
    bv=bars["volume"].values.reshape(-1,1)/n_levels
    dec=1.0/(np.arange(1,n_levels+1)**0.7)
    return {"bid_levels":mid.reshape(-1,1)-sp,"ask_levels":mid.reshape(-1,1)+sp,
            "bid_vols":bv*dec*rng.uniform(0.8,1.2,(n,n_levels)),
            "ask_vols":bv*dec*rng.uniform(0.8,1.2,(n,n_levels))}

def order_book_features(ob,index,k=5):
    F=pd.DataFrame(index=index)
    bv=ob["bid_vols"][:,:k]; av=ob["ask_vols"][:,:k]
    bl=ob["bid_levels"]; al=ob["ask_levels"]
    F["ob_bid_depth"]=bv.sum(1); F["ob_ask_depth"]=av.sum(1)
    tot=F["ob_bid_depth"]+F["ob_ask_depth"]+1e-9
    F["ob_imbalance"]=(F["ob_bid_depth"]-F["ob_ask_depth"])/tot
    F["ob_spread_pip"]=(al[:,0]-bl[:,0])/0.0001
    F["ob_bid_wall"]=(bv>3*bv.mean(1,keepdims=True)).any(1).astype(float)
    F["ob_ask_wall"]=(av>3*av.mean(1,keepdims=True)).any(1).astype(float)
    bv0=ob["bid_vols"][:,0]; av0=ob["ask_vols"][:,0]
    F["microprice"]=(bl[:,0]*av0+al[:,0]*bv0)/(bv0+av0+1e-9)
    return F

def tick_volume_imbalance(bars,window=20):
    F=pd.DataFrame(index=bars.index)
    tick=np.sign(bars["close"].diff()); vol=bars["volume"].astype(float)
    buy=vol*(tick>0).astype(float)+vol*0.5*(tick==0).astype(float)
    sell=vol*(tick<0).astype(float)+vol*0.5*(tick==0).astype(float)
    F["tvi_buy"]=buy.rolling(window).sum(); F["tvi_sell"]=sell.rolling(window).sum()
    tot=F["tvi_buy"]+F["tvi_sell"]+1e-9
    F["tvi_imbalance"]=(F["tvi_buy"]-F["tvi_sell"])/tot
    F["tvi_ratio"]=(F["tvi_buy"]/(F["tvi_sell"]+1e-9)).clip(0,5)
    F["tvi_impact"]=(tick*vol/(vol.rolling(window).mean()+1e-9)).rolling(window).mean()
    return F

def session_features(index):
    F=pd.DataFrame(index=index); h=index.hour; m=index.minute; dec=h+m/60
    F["sess_tokyo"]=((dec>=0)&(dec<9)).astype(float)
    F["sess_london"]=((dec>=7)&(dec<16)).astype(float)
    F["sess_ny"]=((dec>=12)&(dec<21)).astype(float)
    F["sess_sydney"]=((dec>=21)|(dec<6)).astype(float)
    F["sess_ln_ny"]=((dec>=13)&(dec<16)).astype(float)
    F["sess_open"]=(((dec>=7)&(dec<7.25))|((dec>=12)&(dec<12.25))).astype(float)
    tm=h*60+m
    F["tod_sin"]=np.sin(2*np.pi*tm/1440); F["tod_cos"]=np.cos(2*np.pi*tm/1440)
    F["dow_sin"]=np.sin(2*np.pi*index.dayofweek/5); F["dow_cos"]=np.cos(2*np.pi*index.dayofweek/5)
    return F

def correlation_regime_features(returns_df,window=60):
    F=pd.DataFrame(index=returns_df.index)
    cols=list(returns_df.columns)
    nc=len(cols)
    if nc < 2 or len(returns_df) == 0:
        F["corr_avg"]=0.0
        F["corr_dispersion"]=0.0
        F["corr_eigenratio"]=1.0
        F["corr_zscore"]=0.0
        F["corr_break"]=0.0
        return F

    pair_corrs=[]
    min_periods=max(3, window//2)
    for i in range(nc):
        for j in range(i+1,nc):
            pair_corrs.append(
                returns_df[cols[i]].rolling(window, min_periods=min_periods).corr(returns_df[cols[j]])
            )

    C=pd.concat(pair_corrs,axis=1).replace([np.inf,-np.inf],np.nan)
    F["corr_avg"]=C.mean(axis=1)
    F["corr_dispersion"]=C.std(axis=1).fillna(0.0)

    # Equicorrelation approximation of the rolling correlation matrix spectrum.
    # This keeps the regime feature vectorized while preserving the intended
    # "dominant common factor vs residual factors" signal.
    avg=F["corr_avg"].clip(-0.99,0.99)
    lambda1=1.0+(nc-1)*avg.abs()
    lambda2=(1.0-avg.abs()).clip(lower=1e-6)
    F["corr_eigenratio"]=(lambda1/lambda2).clip(1.0,1e6)
    rm=F["corr_avg"].rolling(window*3).mean(); rs=F["corr_avg"].rolling(window*3).std()+1e-9
    F["corr_zscore"]=(F["corr_avg"]-rm)/rs; F["corr_break"]=(F["corr_zscore"].abs()>2.0).astype(float)
    return F.ffill().fillna(0)

def hurst_exponent(arr):
    n=len(arr)
    if n<20: return 0.5
    lags=range(4,min(50,n//2)); tau=[]
    for lag in lags:
        sub=arr[-lag*2:]; chunks=[sub[i:i+lag] for i in range(0,len(sub)-lag+1,lag)]; rs_vals=[]
        for c in chunks:
            if len(c)<2: continue
            c=np.array(c,dtype=float); std=c.std()
            if std<1e-12: continue
            dev=np.cumsum(c-c.mean()); rs=(dev.max()-dev.min())/std
            if rs>0: rs_vals.append(rs)
        if rs_vals: tau.append(np.mean(rs_vals))
    tau = [t for t in tau if t > 0 and np.isfinite(t)]
    if len(tau)<4: return 0.5
    try:
        H,_=np.polyfit(np.log(list(lags)[:len(tau)]),np.log(tau),1)
        return float(np.clip(H,0.1,0.9))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError): return 0.5

def rolling_hurst(series,window=120,step=20):
    n=len(series); val=np.full(n,np.nan)
    for i in range(window,n,step):
        h=hurst_exponent(series.values[i-window:i]); val[i-step:i]=h
    return pd.Series(val,index=series.index).ffill().fillna(0.5).rename("hurst")

def fast_trend_score(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Fast range-vs-close volatility ratio (Parkinson-style vs close-to-close vol).
    Ratio > ~1.2 suggests expanding ranges vs Gaussian returns (trending proxy).
    """
    low_s = low.astype(float).clip(lower=1e-12)
    high_s = high.astype(float).clip(lower=1e-12)
    close_s = close.astype(float).clip(lower=1e-12)
    log_hl = np.log(high_s / low_s)
    parkinson = np.sqrt(log_hl.pow(2).rolling(window, min_periods=2).mean() / (4 * np.log(2)))
    prev = close_s.shift(1).clip(lower=1e-12)
    cc = np.log(close_s / prev)
    cc_vol = cc.rolling(window, min_periods=2).std()
    ratio = parkinson / (cc_vol + 1e-8)
    return ratio.clip(lower=0.01).fillna(1.0).rename("parkinson_trend_ratio")

def fractal_dimension(series,window=30):
    arr=series.values.astype(float); n=len(arr); fd=np.full(n,np.nan)
    for i in range(window,n):
        sub=arr[i-window:i]; Lm=[]
        for k in range(1,window//2):
            segs=[abs(sub[j+k]-sub[j]) for j in range(0,window-k,k) if j+k<window]
            if segs: Lm.append(np.mean(segs)*(window-1)/(len(segs)*k*k))
        if len(Lm)>2:
            try:
                s,_=np.polyfit(np.log(np.arange(1,len(Lm)+1)),np.log(np.maximum(Lm,1e-12)),1)
                fd[i]=float(np.clip(-s,1.0,2.0))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError): fd[i]=1.5
    return pd.Series(fd,index=series.index).ffill().fillna(1.5).rename("fractal_dim")

def regime_label(h):
    return pd.Series(np.where(h>0.6,1.0,np.where(h<0.4,-1.0,0.0)),index=h.index,name="regime_label")

def options_proxy_features(bars,window=20):
    F=pd.DataFrame(index=bars.index)
    r=np.log(bars["close"]/bars["close"].shift(1))
    c=np.log(bars["close"]); h=np.log(bars["high"]); l=np.log(bars["low"])
    F["iv_proxy"]=np.sqrt((0.5*(h-l)**2-(2*np.log(2)-1)*(c-c.shift(1))**2).rolling(window).mean())*np.sqrt(252)
    F["skew_proxy"]=r.rolling(window).skew()
    F["term_proxy"]=(r.rolling(5).std()*np.sqrt(252)/(r.rolling(60).std()*np.sqrt(252)+1e-9)).clip(0.3,3.0)
    up=r[r>0].rolling(window,min_periods=5).std()*np.sqrt(252)
    dn=(-r[r<0]).rolling(window,min_periods=5).std()*np.sqrt(252)
    rr=(up-dn.reindex(bars.index).ffill()).fillna(0)
    F["risk_reversal"]=rr.clip(-0.05,0.05)/0.05
    return F.ffill().fillna(0)

def cot_features(index,cot_data=None):
    F=pd.DataFrame(index=index)
    if cot_data is None:
        # No COT data available: emit neutral zeros rather than synthetic random
        # noise so this never leaks fake signal into training/backtests.
        F["cot_net"]=0.0; F["cot_noncom_net"]=0.0
        F["cot_extreme"]=0.0; F["cot_change"]=0.0; return F
    cot=cot_data.copy(); cot.index=pd.to_datetime(cot.index,utc=True)
    cot=cot.reindex(index,method="ffill").ffill().bfill()
    ln=cot.get("long_noncom",pd.Series(0,index=index)); sn=cot.get("short_noncom",pd.Series(0,index=index))
    F["cot_net"]=(ln-sn)/(ln+sn+1e-9); F["cot_noncom_net"]=F["cot_net"]
    F["cot_extreme"]=(F["cot_net"].abs()>0.7).astype(float); F["cot_change"]=F["cot_net"].diff(7).fillna(0)
    return F

class AdvancedFeatureEngineer:
    def __init__(self,hurst_window=120,hurst_step=20,corr_window=60,tvi_window=20,options_window=20):
        self.hw=hurst_window;self.hs=hurst_step;self.cw=corr_window;self.tw=tvi_window;self.ow=options_window

    def build(self, bars: pl.DataFrame, base_features: pl.DataFrame, cot_data=None) -> pl.DataFrame:
        bars_pd = bars.to_pandas()
        if "timestamp_utc" in bars_pd.columns:
            bars_pd.set_index("timestamp_utc", inplace=True)

        base_pd = base_features.to_pandas()
        if "timestamp_utc" in base_pd.columns:
            base_pd.set_index("timestamp_utc", inplace=True)

        idx=base_pd.index; bars_a=bars_pd.reindex(idx).ffill()
        F=pd.DataFrame(index=idx)
        ob=synthetic_orderbook(bars_a); F=pd.concat([F,order_book_features(ob,idx)],axis=1)
        F=pd.concat([F,tick_volume_imbalance(bars_a,self.tw)],axis=1)
        F=pd.concat([F,session_features(idx)],axis=1)
        ret_cols=[c for c in base_pd.columns if c.endswith("_ret")]
        if ret_cols: F=pd.concat([F,correlation_regime_features(base_pd[ret_cols],self.cw)],axis=1)
        close=bars_a["close"]
        tr = fast_trend_score(bars_a["high"], bars_a["low"], close,
                              window=max(10, min(self.hw, 50)))
        log_ret = np.log(close.astype(float).clip(lower=1e-12) / close.astype(float).shift(1).clip(lower=1e-12)).fillna(0.0)
        H = rolling_hurst(log_ret, window=self.hw, step=self.hs)
        fd = fractal_dimension(close, 30)
        rl = regime_label(H)
        F = pd.concat([F, tr.to_frame(), H.to_frame(), fd.to_frame(), rl.to_frame()], axis=1)
        F=pd.concat([F,options_proxy_features(bars_a,self.ow)],axis=1)
        F=pd.concat([F,cot_features(idx,cot_data)],axis=1)
        F=F.replace([np.inf,-np.inf],np.nan).ffill().fillna(0)
        print(f"[AdvFeatures] +{F.shape[1]} cols")
        return pl.from_pandas(F.reset_index(drop=True))

# ── Compatibility wrappers for main.py ──────────────────────────────────

class L2OrderBookFeatures:
    def __init__(self, n_levels=10, lookback=20):
        self.n = n_levels
    def from_bars(self, bars):
        ob = synthetic_orderbook(bars, self.n)
        return order_book_features(ob, bars.index, k=min(5, self.n))

def session_clock_features(index):
    return session_features(index)

class CorrelationRegimeDetector:
    def __init__(self, window=60, break_thresh=0.3):
        self.w = window; self.t = break_thresh
    def build(self, returns_df):
        return correlation_regime_features(returns_df, self.w)

def rolling_hurst_fractal(bars, windows=[30, 60, 120]):
    import pandas as pd
    log_ret = pd.Series(np.log(bars['close'].values / bars['close'].shift(1).bfill().values), index=bars.index).fillna(0)
    df = pd.DataFrame(index=bars.index)
    for w in windows:
        h = rolling_hurst(log_ret, window=w, step=1)
        fd = fractal_dimension(log_ret, window=w)
        df[f'hurst_{w}']   = h.reindex(bars.index).ffill().fillna(0.5)
        df[f'fractal_{w}'] = fd.reindex(bars.index).ffill().fillna(1.5)
    mid_w = windows[min(1, len(windows)-1)]
    df['trending']       = (df[f'hurst_{mid_w}'] > 0.55).astype(float)
    df['mean_reverting'] = (df[f'hurst_{mid_w}'] < 0.45).astype(float)
    return df

class OptionsSkewFeatures:
    def __init__(self, windows=[5, 20, 60]):
        self.w = windows[-1] if windows else 20
    def build_synthetic(self, bars):
        return options_proxy_features(bars, window=self.w)

class COTFeatures:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or PATHS["data_raw_cot"]
    def build_synthetic(self, index):
        return cot_features(index)

class AdvancedFeatureBuilder:
    """Flexible wrapper — can be called as afb.build(bars) or afb.build(bars, feats)."""
    def __init__(self, hurst_windows=[30,60,120], **kw):
        self._inner = AdvancedFeatureEngineer(hurst_window=hurst_windows[-1] if hurst_windows else 60)
    def build(self, bars, base_features=None, **kw):
        if base_features is None:
            # Build a minimal base_features from bars
            from features.feature_engineering import FeatureEngineer
            fe = FeatureEngineer(atr_window=6, lag_windows=[5,20,60])
            base_features = fe.build(bars)
        return self._inner.build(bars, base_features)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-PAIR INTERACTION FEATURES  (B7–B11)
# Call compute_multipair_features(pair_bars_dict, atr_col="atr_6") to get a
# DataFrame aligned to the primary pair's index.
# ─────────────────────────────────────────────────────────────────────────────

def compute_multipair_features(
    pair_bars: "dict[str, pl.DataFrame | pd.DataFrame]",
    momentum_window: int = 20,
    atr_col: str = "atr_6",
    dispersion_window: int = 5,
    atr_window: int = 6,
    _ofi_z_fast: int = 20,
    _ofi_z_slow: int = 120,
    _tbm_default_horizon: int = 10,
) -> pl.DataFrame:
    """
    Compute cross-pair features B7–B11 from a dict of {pair: OHLCV DataFrame}.

    Returns a DataFrame indexed to the primary pair with columns:
      B7  rel_mom_{i}_{j}     : r_i(20) - r_j(20) for economically linked pairs
      B8  vol_share_{pair}    : ATR_i / sum(ATR basket)
      B9  cross_dispersion    : StdDev of 5-bar returns across all pairs
      B10 time_to_barrier_est : ATR_20 / |ΔP_5| proxy for momentum vs noise
      B11 no_trade_score      : 1 if low-vol + neutral OFI + choppy trend

    All features respect the temporal causality: only past data is used.
    """
    if not pair_bars:
        return pd.DataFrame()

    pairs = list(pair_bars.keys())
    primary = pairs[0]

    # Convert primary pair first so we can safely use .index
    primary_bars = pair_bars[primary]
    if isinstance(primary_bars, pl.DataFrame):
        b_pd = primary_bars.to_pandas()
        if "timestamp_utc" in b_pd.columns:
            b_pd.set_index("timestamp_utc", inplace=True)
        pair_bars[primary] = b_pd
        primary_bars = b_pd
    idx = primary_bars.index

    # Compute log returns and ATR for each pair, aligned to primary index
    returns = {}
    atrs    = {}
    for pair, bars in pair_bars.items():
        if isinstance(bars, pl.DataFrame):
            b_pd = bars.to_pandas()
            if "timestamp_utc" in b_pd.columns:
                b_pd.set_index("timestamp_utc", inplace=True)
            bars = b_pd

        bars_aligned = bars.reindex(idx, method="ffill")
        ret = np.log(bars_aligned["close"] / bars_aligned["close"].shift(1))
        returns[pair] = ret.fillna(0.0)
        # ATR proxy: rolling true-range mean
        prev_c = bars_aligned["close"].shift(1)
        tr = pd.concat([
            bars_aligned["high"] - bars_aligned["low"],
            (bars_aligned["high"] - prev_c).abs(),
            (bars_aligned["low"] - prev_c).abs(),
        ], axis=1).max(axis=1)
        atrs[pair] = tr.rolling(atr_window, min_periods=2).mean().fillna(1e-6)

    F = pd.DataFrame(index=idx)

    # B7. Relative momentum: r_i(window) - r_j(window) for all i<j pairs
    cum_rets = {p: returns[p].rolling(momentum_window, min_periods=2).sum() for p in pairs}
    for i, pi in enumerate(pairs):
        for pj in pairs[i + 1:]:
            col = f"rel_mom_{pi}_{pj}"
            F[col] = (cum_rets[pi] - cum_rets[pj]).fillna(0.0)

    # B8. Volatility dominance: ATR_i / basket_ATR
    atr_basket = sum(atrs[p] for p in pairs) + 1e-9
    for pair in pairs:
        F[f"vol_share_{pair}"] = (atrs[pair] / atr_basket).fillna(0.0)

    # B9. Cross-pair dispersion: StdDev of short-window returns across pairs
    ret_matrix = pd.DataFrame({p: returns[p].rolling(dispersion_window, min_periods=2).sum()
                                for p in pairs})
    F["cross_dispersion"] = ret_matrix.std(axis=1).fillna(0.0)

    # B10. Time-to-barrier estimate: ATR_20 / |ΔP_5| — short = momentum, long = drift
    primary_ret5   = returns[primary].rolling(dispersion_window, min_periods=2).sum().abs() + 1e-8
    primary_atr20  = atrs[primary].rolling(20, min_periods=5).mean().fillna(1e-6)
    F["time_to_barrier_est"] = (primary_atr20 / primary_ret5).clip(0.1, 20.0).fillna(5.0)

    # B11. No-trade zone score
    # Conditions: low vol + neutral OFI-Z + choppy trend
    # Vol condition: rolling ATR below 25th percentile
    atr_pct25 = atrs[primary].rolling(200, min_periods=50).quantile(0.25)
    low_vol   = (atrs[primary] < atr_pct25).astype(float)

    # OFI-Z neutral band: approximate with cross-pair dispersion being very low
    neutral_ofi = (F["cross_dispersion"] < F["cross_dispersion"].rolling(200, min_periods=50).quantile(0.3)).astype(float)

    # Trend stability: low dispersion of returns across window -> choppy
    trend_unstable = (F["cross_dispersion"] < 1e-5).astype(float)

    F["no_trade_score"] = ((low_vol + neutral_ofi + trend_unstable) / 3.0).clip(0.0, 1.0)

    F = F.ffill().bfill().fillna(0.0)
    return pl.from_pandas(F.reset_index(drop=True))

def compute_vpin(bars: pl.DataFrame, bucket_size: int = 50, n_buckets: int = 50) -> pl.Series:
    buy_vol = pl.when(pl.col("close") > pl.col("open")).then(pl.col("volume")).otherwise(0.0)
    sell_vol = pl.when(pl.col("close") <= pl.col("open")).then(pl.col("volume")).otherwise(0.0)

    df = bars.with_columns([
        buy_vol.alias("buy_vol"),
        sell_vol.alias("sell_vol")
    ])

    df = df.with_columns([
        pl.col("buy_vol").rolling_sum(window_size=bucket_size).alias("buy_bucket"),
        pl.col("sell_vol").rolling_sum(window_size=bucket_size).alias("sell_bucket"),
        pl.col("volume").rolling_sum(window_size=bucket_size).alias("total_bucket")
    ])

    vpin = (
        (df["buy_bucket"] - df["sell_bucket"]).abs() / (df["total_bucket"] + 1e-9)
    ).rolling_mean(window_size=n_buckets)

    return vpin.fill_null(0.0).fill_nan(0.0).alias("vpin")

def compute_realized_moments(close: pl.Series, window: int = 20) -> pl.DataFrame:
    ret = (close / close.shift(1)).log().alias("ret")

    up_ret = pl.when(ret > 0).then(ret).otherwise(0.0)
    dn_ret = pl.when(ret < 0).then(ret).otherwise(0.0)

    df = pl.DataFrame({"ret": ret, "up_ret": up_ret, "dn_ret": dn_ret})

    df = df.with_columns([
        pl.col("ret").rolling_skew(window_size=window).alias(f"rolling_skew_{window}"),
        pl.col("ret").rolling_mean(window_size=window).alias("mu"),
        (pl.col("ret")**2).rolling_mean(window_size=window).alias("mu2"),
        (pl.col("ret")**3).rolling_mean(window_size=window).alias("mu3"),
        (pl.col("ret")**4).rolling_mean(window_size=window).alias("mu4"),
        pl.col("up_ret").rolling_std(window_size=window).alias("up_std"),
        pl.col("dn_ret").rolling_std(window_size=window).alias("dn_std"),
    ])

    df = df.with_columns([
        (pl.col("mu2") - pl.col("mu")**2).alias("var")
    ])

    df = df.with_columns([
        (
            (pl.col("mu4") - 4*pl.col("mu3")*pl.col("mu") + 6*pl.col("mu2")*(pl.col("mu")**2) - 3*(pl.col("mu")**4))
            / (pl.col("var")**2 + 1e-9)
        ).alias(f"rolling_kurt_{window}"),
        (pl.col("up_std") / (pl.col("dn_std") + 1e-9)).alias("rvol_ratio")
    ])

    res = df.select([
        pl.col(f"rolling_skew_{window}").fill_nan(0.0).fill_null(0.0),
        pl.col(f"rolling_kurt_{window}").fill_nan(0.0).fill_null(0.0),
        pl.col("rvol_ratio").fill_nan(0.0).fill_null(0.0)
    ])
    return res

def compute_asia_london_gap(bars: pl.DataFrame, atr: pl.Series = None) -> pl.Series:
    time_col = 'timestamp_utc' if 'timestamp_utc' in bars.columns else 'timestamp' if 'timestamp' in bars.columns else 'datetime'

    df = bars.select([pl.col(time_col), pl.col('close')])
    df = df.with_columns([
        pl.col(time_col).dt.date().alias('date'),
        pl.col(time_col).dt.time().alias('time')
    ])

    df = df.with_columns([
        (pl.col('time') < pl.time(7, 0)).alias('is_asia'),
        (pl.col('time') >= pl.time(7, 0)).alias('is_london')
    ])

    london_open_times = (
        df.filter(pl.col('is_london'))
          .group_by('date')
          .agg(pl.col(time_col).first().alias('london_open_time'))
    )

    asia_close_vals = (
        df.filter(pl.col('is_asia'))
          .group_by('date')
          .agg(pl.col('close').last().alias('asia_close'))
    )

    daily_gaps = london_open_times.join(asia_close_vals, on='date', how='inner')

    df = df.join(daily_gaps, left_on=time_col, right_on='london_open_time', how='left')

    df = df.with_columns([
        (pl.col('close') - pl.col('asia_close')).alias('gap')
    ])

    gap_series = df['gap'].forward_fill()

    if atr is not None:
        gap_series = gap_series / (atr + 1e-9)

    return gap_series.fill_null(0.0).fill_nan(0.0).alias('asia_london_gap')
