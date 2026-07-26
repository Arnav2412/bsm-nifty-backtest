# the main backtest. loads the real nifty option data, prices everything with
# black-scholes (black-76 forward version), and checks how close it lands to
# the actual traded prices.
#
# what i look at:
#   - accuracy under 3 different vol inputs (realized / ewma / garch)
#   - error broken down by moneyness and by days-to-expiry
#   - the implied vol of every option -> the volatility smile
#   - a "cheat" benchmark where i fit the smile and re-price, to prove the
#     smile is what's actually hurting black-scholes
#
# run:  python src/backtest.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # so it saves pngs without needing a screen
import matplotlib.pyplot as plt

from black_scholes import black76_price, black76_iv_vec
from volatility import all_models
from metrics import (error_metrics, moneyness_bucket, expiry_bucket,
                     grouped_metrics, bootstrap_ci)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
MODELS = ["realized", "ewma", "garch"]


def load():
    opt_csv = os.path.join(DATA, "nifty_options.csv")
    if not os.path.exists(opt_csv):
        print("no dataset yet, building it from the raw bhavcopy first...")
        from parse_bhavcopy import build
        build()
    df = pd.read_csv(opt_csv, parse_dates=["date", "expiry"])
    und = (pd.read_csv(os.path.join(DATA, "nifty_underlying.csv"), parse_dates=["date"])
           .set_index("date")["underlying"].sort_index())
    return df, und


def price_everything(df, vols):
    # add a black-scholes price column for each of the three vol models
    F = df["forward"].values; K = df["strike"].values
    T = df["T"].values; r = df["risk_free_rate"].values
    is_call = df["option_type"].eq("call").values
    for m in MODELS:
        sig = df["date"].map(vols[m]).values
        px = np.empty(len(df))
        px[is_call] = black76_price(F[is_call], K[is_call], T[is_call], r[is_call], sig[is_call], "call")
        px[~is_call] = black76_price(F[~is_call], K[~is_call], T[~is_call], r[~is_call], sig[~is_call], "put")
        df[f"px_{m}"] = px
        df[f"vol_{m}"] = sig
    return df


def add_implied_vol(df):
    df["implied_vol"] = black76_iv_vec(
        df["market_price"].values, df["forward"].values, df["strike"].values,
        df["T"].values, df["risk_free_rate"].values, df["option_type"].eq("call").values)
    return df


def smile_benchmark(df):
    # for each expiry, fit implied vol as a quadratic in log-moneyness
    #   iv(x) = a + b*x + c*x^2
    # then re-price with that fitted vol. if the error collapses, it means BS
    # was only wrong because it used one flat vol. b = skew, c = curvature.
    df["px_smile"] = np.nan
    skews, curves = [], []
    for (d, e), g in df.groupby(["date", "expiry"]):
        v = g.dropna(subset=["implied_vol"])
        v = v[(v["implied_vol"] > 0.02) & (v["implied_vol"] < 1.5)]
        if len(v) < 5:
            continue
        x = v["log_moneyness"].values; y = v["implied_vol"].values
        try:
            c2, c1, c0 = np.polyfit(x, y, 2)
        except Exception:
            continue
        skews.append(c1); curves.append(c2)
        fit = np.clip(c0 + c1 * g["log_moneyness"].values + c2 * g["log_moneyness"].values**2, 0.02, 1.5)
        ic = g["option_type"].eq("call").values
        px = np.empty(len(g))
        px[ic] = black76_price(g["forward"].values[ic], g["strike"].values[ic],
                               g["T"].values[ic], g["risk_free_rate"].values[ic], fit[ic], "call")
        px[~ic] = black76_price(g["forward"].values[~ic], g["strike"].values[~ic],
                                g["T"].values[~ic], g["risk_free_rate"].values[~ic], fit[~ic], "put")
        df.loc[g.index, "px_smile"] = px
    return df, np.array(skews), np.array(curves)


def run():
    os.makedirs(RESULTS, exist_ok=True)
    df, und = load()
    print(f"loaded {len(df):,} real option quotes over {df['date'].nunique()} days "
          f"(forward {df['forward'].min():.0f}-{df['forward'].max():.0f})")

    vols = all_models(und)
    print("vol inputs (median): " + ", ".join(f"{m}={vols[m].median():.1%}" for m in MODELS))

    df = price_everything(df, vols)
    print("solving for implied vol on every option...")
    df = add_implied_vol(df)
    df, skews, curves = smile_benchmark(df)

    df["moneyness"] = df.apply(moneyness_bucket, axis=1)
    df["expiry_b"] = df["days_to_expiry"].apply(expiry_bucket)
    df.to_csv(os.path.join(RESULTS, "backtest_detail.csv"), index=False)

    # build up the text summary
    out = []
    def w(s): out.append(s)
    w("=" * 66)
    w("BLACK-SCHOLES ACCURACY ON REAL NIFTY50 OPTIONS  (black-76 form)")
    w("=" * 66)
    w(f"data: {df['date'].nunique()} days ({df['date'].min().date()} to {df['date'].max().date()}), "
      f"{len(df):,} quotes, {df['expiry'].nunique()} expiries.")
    w(f"forward {df['forward'].min():.0f}-{df['forward'].max():.0f}; "
      f"strikes {df['strike'].min():.0f}-{df['strike'].max():.0f}; rate {df['risk_free_rate'].median():.2%}.")
    w("")
    w("(1) accuracy by volatility input  [errors in Rs.]")
    w("-" * 66)
    rows = []
    for m in MODELS:
        met = error_metrics(df["market_price"], df[f"px_{m}"])
        lo, hi = bootstrap_ci(df["market_price"], df[f"px_{m}"], "RMSE")
        met["model"] = m; met["RMSE_95CI"] = f"[{lo:.1f}, {hi:.1f}]"
        rows.append(met)
    tbl = pd.DataFrame(rows)[["model", "n", "RMSE", "RMSE_95CI", "MAE", "MAPE_%", "MeanSignedErr", "R2"]]
    w(tbl.to_string(index=False))
    w("")
    w("(2) smile-fitted benchmark vs plain flat-vol BS")
    w("-" * 66)
    sm = error_metrics(df["market_price"], df["px_smile"])
    base = error_metrics(df["market_price"], df["px_realized"])
    w(f"  flat-vol BS (realized): RMSE={base['RMSE']:.2f}  MAPE={base['MAPE_%']:.1f}%  R2={base['R2']:.3f}")
    w(f"  smile-fitted          : RMSE={sm['RMSE']:.2f}  MAPE={sm['MAPE_%']:.1f}%  R2={sm['R2']:.3f}")
    if base["RMSE"] > 0:
        w(f"  -> fitting the smile drops RMSE {100*(1-sm['RMSE']/base['RMSE']):.0f}%, so the flat-vol "
          f"assumption is the real problem, not the formula.")
    w("")
    w("(3) smile shape  (iv = a + b*x + c*x^2 per expiry)")
    w("-" * 66)
    w(f"  median skew  b = {np.median(skews):+.3f}  (negative = the usual equity skew)")
    w(f"  median curve c = {np.median(curves):+.3f}  (positive = smile BS can't match)")
    w("")
    w("(4) error by moneyness  (using realized vol)")
    w("-" * 66)
    w(grouped_metrics(df, "moneyness", "px_realized").to_string(index=False))
    w("")
    w("(5) error by time to expiry")
    w("-" * 66)
    w(grouped_metrics(df, "expiry_b", "px_realized").to_string(index=False))

    summary = "\n".join(out)
    with open(os.path.join(RESULTS, "summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary + "\n")

    make_charts(df, vols, skews, curves)
    print("done -> results/")
    return df


def make_charts(df, vols, skews, curves):
    # 1 - predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["market_price"], df["px_realized"], s=4, alpha=0.25)
    lim = [0, np.nanpercentile(df["market_price"], 99)]
    ax.plot(lim, lim, "r--", lw=1, label="perfect")
    ax.set(xlim=lim, ylim=lim, xlabel="market price (Rs.)", ylabel="black-scholes price (Rs.)",
           title="market vs black-scholes (realized vol)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "actual_vs_bsm.png"), dpi=120); plt.close(fig)

    # 2 - rmse per model
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vals = [error_metrics(df["market_price"], df[f"px_{m}"])["RMSE"] for m in MODELS]
    smile_rmse = error_metrics(df["market_price"], df["px_smile"])["RMSE"]
    bars = ax.bar(MODELS + ["smile-fit"], vals + [smile_rmse], color=["#4C72B0"]*3 + ["#55A868"])
    ax.set(ylabel="RMSE (Rs.)", title="pricing error by model")
    for b, v in zip(bars, vals + [smile_rmse]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "rmse_by_model.png"), dpi=120); plt.close(fig)

    # 3 - the smile, on a near-dated liquid expiry
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cand = df[df["days_to_expiry"].between(5, 35)]
    if len(cand):
        e = cand.groupby("expiry")["contracts"].sum().idxmax()
        s = df[(df["expiry"] == e) & df["implied_vol"].notna()]
        s = s[(s["implied_vol"] > 0.02) & (s["implied_vol"] < 0.8)]
        d0 = s["date"].min(); s = s[s["date"] == d0].sort_values("log_moneyness")
        for ot, sub in s.groupby("option_type"):
            ax.scatter(sub["log_moneyness"], sub["implied_vol"]*100, s=18, label=f"{ot} IV")
        flat = vols["realized"].get(d0, vols["realized"].median()) * 100
        ax.axhline(flat, color="k", ls="--", lw=1, label=f"BS flat vol ({flat:.0f}%)")
        ax.set_title(f"volatility smile: market IV vs one flat BS vol\n(expiry {pd.Timestamp(e).date()}, on {pd.Timestamp(d0).date()})")
    ax.set(xlabel="log(strike / forward)", ylabel="implied vol (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "volatility_smile.png"), dpi=120); plt.close(fig)

    # 4 - mape by moneyness
    fig, ax = plt.subplots(figsize=(7, 4.5))
    g = df.assign(ape=(df["px_realized"] - df["market_price"]).abs() / df["market_price"].abs() * 100)
    piv = g.groupby(["moneyness", "option_type"])["ape"].mean().unstack().reindex(["ITM", "ATM", "OTM"])
    piv.plot(kind="bar", ax=ax)
    ax.set(ylabel="MAPE (%)", xlabel="", title="error by moneyness")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "error_by_moneyness.png"), dpi=120); plt.close(fig)

    # 5 - the three vol estimates over time
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m in MODELS:
        ax.plot(vols.index, vols[m]*100, label=m, lw=1.4)
    ax.set(ylabel="annualized vol (%)", title="volatility estimates over Q1-2024"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "vol_models.png"), dpi=120); plt.close(fig)

    # 6 - error distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    err = (df["px_realized"] - df["market_price"]).dropna()
    err = err[err.between(err.quantile(0.01), err.quantile(0.99))]
    ax.hist(err, bins=60, color="#C44E52", alpha=0.8); ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="BS price - market price (Rs.)", ylabel="count",
           title=f"pricing error spread (mean {err.mean():+.1f})")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "error_distribution.png"), dpi=120); plt.close(fig)
    print("saved 6 charts")


if __name__ == "__main__":
    run()
