"""Walk-forward diagnostic: is there ANY gross (pre-cost) signal in the 4H features?
Reports per-pair out-of-sample Spearman IC, hit rate, and gross vs net pips/trade."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_walkforward_4h import ALL_PAIRS, PAIRS, H, EMBARGO, N_FOLDS, TEST_FRAC, SLIP, make_model

d = np.load(ROOT / ".tmp" / "all_584_audit_data.npz")
meta = json.load(open(ROOT / "checkpoints/stationary_ensemble/metadata.json"))
X = d["X"][:, meta["curated_feature_indices"]]; idx = d["idx"]; close = d["close"]; spread = d["spread"]
m = len(X); t0 = int(m * (1 - TEST_FRAC)); bounds = np.linspace(t0, m, N_FOLDS + 1).astype(int)
res = {}
for pr in PAIRS:
    pi = ALL_PAIRS.index(pr); pip = 0.01 if "JPY" in pr else 0.0001
    cp = close[:, pi]; ok = idx + H < len(cp)
    fwd = np.full(m, np.nan); fwd[ok] = (cp[idx[ok] + H] - cp[idx[ok]]) / pip
    fin = np.isfinite(fwd)
    cost = (spread[idx, pi] + SLIP * pip) / pip
    pred = np.full(m, np.nan); th = np.full(m, np.nan)
    for f in range(N_FOLDS):
        a, b = bounds[f], bounds[f + 1]
        tr = fin & (idx < idx[a] - EMBARGO)
        mdl = make_model().fit(X[tr], np.clip(fwd[tr] / pip * 0 + (cp[idx[tr] + H] / cp[idx[tr]] - 1) * 1e4, -150, 150))
        th[a:b] = np.percentile(np.abs(mdl.predict(X[tr])), 70); pred[a:b] = mdl.predict(X[a:b])
    s = slice(t0, m); v = fin[s]
    ic = spearmanr(pred[s][v], fwd[s][v])[0]
    sel = v & (np.abs(pred[s]) > th[s])
    gross = np.sign(pred[s][sel]) * fwd[s][sel]
    res[pr] = {"IC": float(ic), "n_signals": int(sel.sum()), "hit_rate": float((gross > 0).mean()),
               "gross_pips_per_trade": float(gross.mean()), "avg_cost_pips": float(cost[s][sel].mean()),
               "net_pips_per_trade": float((gross - cost[s][sel]).mean())}
    print(pr, res[pr], flush=True)
(ROOT / "docs" / "walkforward_4h_diagnostic.json").write_text(json.dumps(res, indent=2))
