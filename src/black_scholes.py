# black-scholes stuff: pricing calls/puts, the greeks, and solving for
# implied vol. everything here works on single numbers or numpy arrays.
#
# quick notes on the inputs so i don't confuse myself later:
#   S     = spot price
#   K     = strike
#   T     = time left to expiry, in YEARS (so 15 days = 15/365)
#   r     = risk free rate (annual, decimal - 0.069 = 6.9%)
#   q     = dividend yield (annual). 0 unless i say otherwise
#   sigma = volatility (annual, decimal)

import numpy as np
from scipy.stats import norm

TINY = 1e-8   # to dodge divide-by-zero when T or sigma is ~0


def _d1_d2(S, K, T, r, sigma, q=0.0):
    # the two d terms that show up everywhere in black-scholes
    S = np.asarray(S, float); K = np.asarray(K, float)
    T = np.asarray(T, float); sigma = np.asarray(sigma, float)

    bottom = sigma * np.sqrt(np.maximum(T, TINY))
    bottom = np.where(bottom < TINY, TINY, bottom)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / bottom
    d2 = d1 - bottom
    return d1, d2


def bsm_price(S, K, T, r, sigma, option_type="call", q=0.0):
    # the actual black-scholes price of a european call or put
    S = np.asarray(S, float); K = np.asarray(K, float); T = np.asarray(T, float)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T)     # discount factor on the strike
    df_q = np.exp(-q * T)     # discount factor on the spot (dividends)

    kind = str(option_type).lower()
    if kind in ("call", "c"):
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
        price = np.where(T <= 0, np.maximum(S - K, 0.0), price)   # at expiry = payoff
    elif kind in ("put", "p"):
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
        price = np.where(T <= 0, np.maximum(K - S, 0.0), price)
    else:
        raise ValueError(f"option_type should be call or put, got {option_type!r}")
    return price


def greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    # delta / gamma / vega / theta / rho. handy for sanity checks and hedging.
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    df_r = np.exp(-r * T); df_q = np.exp(-q * T)
    n_d1 = norm.pdf(d1)
    is_call = str(option_type).lower() in ("call", "c")

    delta = df_q * (norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)
    gamma = df_q * n_d1 / (S * sigma * np.sqrt(np.maximum(T, TINY)))
    vega = S * df_q * n_d1 * np.sqrt(np.maximum(T, 0.0))

    if is_call:
        theta = (-S * df_q * n_d1 * sigma / (2 * np.sqrt(np.maximum(T, TINY)))
                 - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1))
        rho = K * T * df_r * norm.cdf(d2)
    else:
        theta = (-S * df_q * n_d1 * sigma / (2 * np.sqrt(np.maximum(T, TINY)))
                 + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1))
        rho = -K * T * df_r * norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,     # per 1% move in vol
        "theta": theta / 365.0,   # per day
        "rho": rho / 100.0,       # per 1% move in rate
    }


def implied_vol(price, S, K, T, r, option_type="call", q=0.0, tol=1e-6, max_iter=100):
    # work backwards from a market price to the vol that produced it.
    # newton's method first, and if that goes haywire fall back to bisection.
    price = float(price); S = float(S); K = float(K); T = float(T); r = float(r)

    df_r = np.exp(-r * T); df_q = np.exp(-q * T)
    if str(option_type).lower() in ("call", "c"):
        floor = max(S * df_q - K * df_r, 0.0)
    else:
        floor = max(K * df_r - S * df_q, 0.0)
    if price < floor - 1e-8 or T <= 0:
        return np.nan   # price is below intrinsic -> no real vol exists

    sigma = 0.20   # 20% is a reasonable starting point for index options
    for _ in range(max_iter):
        model = float(bsm_price(S, K, T, r, sigma, option_type, q))
        gap = model - price
        if abs(gap) < tol:
            return float(sigma)
        v = greeks(S, K, T, r, sigma, option_type, q)["vega"] * 100.0
        if v < 1e-8:
            break
        sigma -= gap / v
        if sigma <= 0 or sigma > 5:
            break   # newton diverged, let bisection take over

    lo, hi = 1e-4, 5.0
    f_lo = float(bsm_price(S, K, T, r, lo, option_type, q)) - price
    f_hi = float(bsm_price(S, K, T, r, hi, option_type, q)) - price
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = float(bsm_price(S, K, T, r, mid, option_type, q)) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def implied_vol_vec(price, S, K, T, r, is_call, q=0.0, n_iter=60, tol=1e-5):
    # same idea as implied_vol but done on whole arrays at once so it's fast.
    # i need this because looping the scalar version over 40k+ options is slow.
    price = np.asarray(price, float); S = np.asarray(S, float)
    K = np.asarray(K, float); T = np.asarray(T, float)
    r = np.asarray(r, float); is_call = np.asarray(is_call, bool)
    df_r = np.exp(-r * T); df_q = np.exp(-q * T)
    floor = np.where(is_call, np.maximum(S * df_q - K * df_r, 0.0),
                     np.maximum(K * df_r - S * df_q, 0.0))
    ok = (price >= floor - 1e-6) & (T > 0) & (price > 0)
    sigma = np.full(S.shape, 0.20)
    for _ in range(n_iter):
      with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(S, K, T, r, sigma, q)
        model = np.where(is_call,
                         S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2),
                         K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1))
        vega = S * df_q * norm.pdf(d1) * np.sqrt(np.maximum(T, TINY))
        step = np.where(vega > 1e-8, (model - price) / vega, 0.0)
        sigma = np.clip(sigma - step, 1e-4, 5.0)
        if ok.any() and np.nanmax(np.abs(step[ok])) < tol:
            break
    return np.where(ok, sigma, np.nan)


# --- black-76 versions ---
# index options are cleanest to price off the FORWARD instead of spot, which
# skips the whole dividend-yield headache. black-76 is just black-scholes with
# S = forward and q = r (so the drift term cancels out). same formula really.
def black76_price(F, K, T, r, sigma, option_type="call"):
    return bsm_price(F, K, T, r, sigma, option_type, q=r)


def black76_iv(price, F, K, T, r, option_type="call", **kw):
    return implied_vol(price, F, K, T, r, option_type, q=r, **kw)


def black76_iv_vec(price, F, K, T, r, is_call, **kw):
    return implied_vol_vec(price, F, K, T, r, is_call, q=r, **kw)


def black76_greeks(F, K, T, r, sigma, option_type="call"):
    return greeks(F, K, T, r, sigma, option_type, q=r)


if __name__ == "__main__":
    # sanity check against a textbook example (Hull): should get ~10.4506
    c = bsm_price(100, 100, 1.0, 0.05, 0.20, "call")
    p = bsm_price(100, 100, 1.0, 0.05, 0.20, "put")
    print(f"call = {c:.4f}  (expect ~10.4506)")
    print(f"put  = {p:.4f}  (expect ~5.5735)")
    print(f"implied vol back-out = {implied_vol(c, 100, 100, 1.0, 0.05, 'call'):.4f}  (expect ~0.20)")
