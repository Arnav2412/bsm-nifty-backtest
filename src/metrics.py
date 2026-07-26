# error measures for comparing model prices to real prices, plus a couple of
# helpers for slicing the data by moneyness / expiry.

import numpy as np
import pandas as pd


def error_metrics(actual, model):
    # rmse, mae, mape, average error and R2 between two sets of prices
    actual = np.asarray(actual, float); model = np.asarray(model, float)
    keep = np.isfinite(actual) & np.isfinite(model)
    a, m = actual[keep], model[keep]
    if len(a) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "MAPE_%": np.nan,
                "MeanSignedErr": np.nan, "R2": np.nan}
    e = m - a
    denom = np.where(np.abs(a) < 1e-8, np.nan, a)
    ss_res = np.nansum(e**2)
    ss_tot = np.nansum((a - a.mean())**2)
    return {
        "n": int(keep.sum()),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAE": float(np.mean(np.abs(e))),
        "MAPE_%": float(np.nanmean(np.abs(e / denom)) * 100),
        "MeanSignedErr": float(np.mean(e)),   # negative = model too cheap
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def moneyness_bucket(row, band=0.01):
    # label each option ITM / ATM / OTM based on how far the strike is from the
    # forward. x = log(K/F): ~0 is at-the-money.
    x = row["log_moneyness"]
    if abs(x) <= band:
        return "ATM"
    if row["option_type"] == "call":
        return "ITM" if x < 0 else "OTM"
    else:
        return "ITM" if x > 0 else "OTM"


def expiry_bucket(dte):
    if dte <= 7:
        return "<=7d"
    if dte <= 30:
        return "8-30d"
    if dte <= 60:
        return "31-60d"
    return ">60d"


def grouped_metrics(df, by, price_col="model_price"):
    # run error_metrics separately for each group (e.g. per moneyness bucket)
    out = []
    for key, g in df.groupby(by):
        row = error_metrics(g["market_price"], g[price_col])
        row[by if isinstance(by, str) else "group"] = key
        out.append(row)
    keycol = by if isinstance(by, str) else "group"
    cols = [keycol, "n", "RMSE", "MAE", "MAPE_%", "MeanSignedErr", "R2"]
    return pd.DataFrame(out)[cols].sort_values(keycol).reset_index(drop=True)


def bootstrap_ci(actual, model, stat="RMSE", n_boot=2000, seed=0):
    # 95% confidence interval for an error stat, by resampling the errors.
    # gives a sense of whether differences between models are real or noise.
    actual = np.asarray(actual, float); model = np.asarray(model, float)
    keep = np.isfinite(actual) & np.isfinite(model)
    e = (model - actual)[keep]
    rng = np.random.default_rng(seed)
    vals = []
    n = len(e)
    for _ in range(n_boot):
        s = e[rng.integers(0, n, n)]
        if stat == "RMSE":
            vals.append(np.sqrt(np.mean(s**2)))
        elif stat == "MAE":
            vals.append(np.mean(np.abs(s)))
        else:
            vals.append(np.mean(s))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
