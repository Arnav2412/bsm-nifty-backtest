# takes the raw NSE F&O bhavcopy files (the real daily settlement files) and
# turns them into one clean csv i can actually backtest on.
#
# the raw files have every F&O instrument in them; i only want NIFTY index
# options that actually traded (CONTRACTS > 0), so the prices are real trades
# and not stale settlement marks.
#
# the tricky bit: nifty has weekly options but futures are only monthly, so i
# can't always find a matching future to use as the forward. instead i pull the
# forward straight out of the option prices using put-call parity:
#     C(K) - P(K) = D * (F - K)      where D = exp(-rT)
# so F = K + (C - P)/D. this is basically what CBOE does for VIX. no dividend
# assumption needed and it's consistent with the very prices i'm testing.

import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
RAW_DIR = os.path.join(HERE, "..", "data", "raw_bhavcopy")
OUT_OPT = os.path.join(HERE, "..", "data", "nifty_options.csv")
OUT_UND = os.path.join(HERE, "..", "data", "nifty_underlying.csv")

MAX_DTE = 120     # ignore the really long-dated illiquid stuff
MIN_PAIRS = 2     # need at least this many strikes to pin down the forward
RISK_FREE = 0.069 # 91-day t-bill yield around Q1 2024, ~6.9%


def _load_raw():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no bhavcopy files found in {RAW_DIR}")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["SYMBOL"] == "NIFTY"].copy()
    raw["date"] = pd.to_datetime(raw["TIMESTAMP"], format="%d-%b-%Y")
    raw["expiry"] = pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y")
    for c in ("STRIKE_PR", "CLOSE", "SETTLE_PR", "CONTRACTS", "OPEN_INT"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def _forward_from_parity(opts):
    # given all the options for one (date, expiry), back out the forward.
    # i fix the rate at the t-bill level and only solve for F, using strikes
    # near the money where both the call and put are liquid (far strikes have
    # stale prices that mess up the estimate).
    T = float(opts["T"].iloc[0])
    D = np.exp(-RISK_FREE * T)
    ce = opts[opts["OPTION_TYP"] == "CE"].set_index("STRIKE_PR")
    pe = opts[opts["OPTION_TYP"] == "PE"].set_index("STRIKE_PR")
    both = ce.index.intersection(pe.index)
    if len(both) < MIN_PAIRS:
        return np.nan, np.nan, np.nan
    K = both.values.astype(float)
    c_minus_p = (ce.loc[both, "CLOSE"] - pe.loc[both, "CLOSE"]).values
    # rough ATM = the strike where call and put prices are closest together
    atm = K[np.argmin(np.abs(c_minus_p))]
    near = np.abs(K - atm) <= 0.05 * atm
    if near.sum() < MIN_PAIRS:
        near = np.abs(K - atm) <= 0.08 * atm
    Kn, cpn = K[near], c_minus_p[near]
    w = np.minimum(ce.loc[both, "CONTRACTS"].values[near],
                   pe.loc[both, "CONTRACTS"].values[near]).astype(float)
    w = np.where(w <= 0, 1.0, w)              # weight by liquidity
    F_each = Kn + cpn / D                       # one forward guess per strike
    F = float(np.average(F_each, weights=w))    # liquidity-weighted average
    return F, float(D), RISK_FREE


def build():
    raw = _load_raw()

    # underlying series = the near-month future close each day (for vol calc)
    fut = raw[raw["INSTRUMENT"] == "FUTIDX"].copy()
    fut["dte"] = (fut["expiry"] - fut["date"]).dt.days
    fut = fut[fut["dte"] > 0]
    near = (fut.sort_values("dte").groupby("date").first().reset_index()
            [["date", "CLOSE"]].rename(columns={"CLOSE": "underlying"}))
    near.sort_values("date").to_csv(OUT_UND, index=False)

    # now the options
    opt = raw[raw["INSTRUMENT"] == "OPTIDX"].copy()
    opt = opt[(opt["CONTRACTS"] > 0) & (opt["CLOSE"] > 0)]   # only real trades
    opt["days_to_expiry"] = (opt["expiry"] - opt["date"]).dt.days
    opt = opt[(opt["days_to_expiry"] >= 1) & (opt["days_to_expiry"] <= MAX_DTE)]
    opt["T"] = opt["days_to_expiry"] / 365.0

    rows = []
    for (date, expiry), grp in opt.groupby(["date", "expiry"]):
        F, D, r = _forward_from_parity(grp)
        if not np.isfinite(F) or F <= 0:
            continue
        for _, o in grp.iterrows():
            kind = "call" if o["OPTION_TYP"] == "CE" else "put"
            rows.append({
                "date": date.date(),
                "expiry": expiry.date(),
                "days_to_expiry": int(o["days_to_expiry"]),
                "T": round(o["T"], 6),
                "forward": round(F, 2),
                "discount": round(D, 6),
                "risk_free_rate": round(r, 5),
                "strike": float(o["STRIKE_PR"]),
                "option_type": kind,
                "market_price": float(o["CLOSE"]),
                "contracts": int(o["CONTRACTS"]),
                "open_int": int(o["OPEN_INT"]) if np.isfinite(o["OPEN_INT"]) else 0,
                "log_moneyness": round(float(np.log(o["STRIKE_PR"] / F)), 6),
            })
    df = pd.DataFrame(rows).sort_values(["date", "expiry", "strike", "option_type"])
    df.to_csv(OUT_OPT, index=False)
    print(f"got {len(df):,} real traded NIFTY option quotes")
    print(f"{df['date'].nunique()} trading days, {df['expiry'].nunique()} expiries")
    print(f"implied risk-free rate: {df['risk_free_rate'].median():.2%} (median)")
    print(f"saved -> {os.path.relpath(OUT_OPT)}")
    return df


if __name__ == "__main__":
    build()
