# different ways to estimate the volatility i feed into black-scholes.
# this matters a lot - BS is only as good as the vol you give it. all three
# of these only use data from BEFORE the option is priced, so the backtest
# stays honest (no peeking at the future).

import numpy as np
import pandas as pd

YEAR = 252   # trading days in a year, for annualising


def log_returns(px):
    return np.log(px / px.shift(1)).dropna()


def realized_vol(px, window=20):
    # plain historical vol: std of the last ~month of returns, annualised
    r = log_returns(px)
    v = r.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(YEAR)
    return v.reindex(px.index).bfill()


def ewma_vol(px, lam=0.94):
    # riskmetrics ewma - weights recent days more, so it reacts faster to shocks.
    # lam=0.94 is the standard riskmetrics number for daily data.
    r = log_returns(px)
    var = pd.Series(index=r.index, dtype=float)
    running = r.var()   # seed it with the plain variance
    for t, x in r.items():
        running = lam * running + (1 - lam) * x**2
        var[t] = running
    v = np.sqrt(var * YEAR)
    return v.reindex(px.index).bfill()


def garch_vol(px):
    # garch(1,1) - the proper way to model vol clustering (calm days follow calm
    # days, wild days follow wild days). uses the arch library; if it's not
    # installed i just fall back to ewma so the script still runs.
    r = log_returns(px) * 100.0   # arch likes returns in percent
    try:
        from arch import arch_model
        model = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        fit = model.fit(disp="off")
        v = (fit.conditional_volatility / 100.0) * np.sqrt(YEAR)
        v.index = r.index
        return v.reindex(px.index).bfill()
    except Exception as e:
        print(f"(garch didn't run - {type(e).__name__} - using ewma instead)")
        return ewma_vol(px)


def all_models(px, window=20):
    # bundle all three into one dataframe indexed by date
    return pd.DataFrame({
        "realized": realized_vol(px, window),
        "ewma": ewma_vol(px),
        "garch": garch_vol(px),
    })
