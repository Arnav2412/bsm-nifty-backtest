# quick tests for the pricing code. run with:  pytest -q
# these mostly check the formula against known textbook values and against
# things that must be true (put-call parity, iv round-tripping, etc).

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from black_scholes import (bsm_price, greeks, implied_vol, implied_vol_vec,
                           black76_price)


def test_matches_textbook():
    # Hull's classic example: S=K=100, T=1, r=5%, vol=20%
    assert bsm_price(100, 100, 1, 0.05, 0.20, "call") == pytest.approx(10.4506, abs=1e-3)
    assert bsm_price(100, 100, 1, 0.05, 0.20, "put") == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    # C - P should equal S*e^-qT - K*e^-rT, always
    S, K, T, r, q, v = 21900, 22000, 0.1, 0.069, 0.0, 0.13
    c = float(bsm_price(S, K, T, r, v, "call", q))
    p = float(bsm_price(S, K, T, r, v, "put", q))
    assert c - p == pytest.approx(S*np.exp(-q*T) - K*np.exp(-r*T), abs=1e-6)


def test_black76_atm_forward():
    # at F=K a call and put should be worth exactly the same
    c = float(black76_price(22000, 22000, 0.25, 0.069, 0.15, "call"))
    p = float(black76_price(22000, 22000, 0.25, 0.069, 0.15, "put"))
    assert c == pytest.approx(p, abs=1e-6)


def test_payoff_at_expiry():
    assert float(bsm_price(120, 100, 0, 0.05, 0.2, "call")) == pytest.approx(20.0)
    assert float(bsm_price(80, 100, 0, 0.05, 0.2, "put")) == pytest.approx(20.0)


def test_more_vol_costs_more():
    assert float(bsm_price(100, 100, 1, 0.05, 0.40, "call")) > float(bsm_price(100, 100, 1, 0.05, 0.10, "call"))


def test_delta_in_range():
    assert 0 < greeks(100, 100, 1, 0.05, 0.2, "call")["delta"] < 1
    assert -1 < greeks(100, 100, 1, 0.05, 0.2, "put")["delta"] < 0


def test_iv_roundtrips():
    price = float(bsm_price(100, 105, 0.5, 0.05, 0.27, "call"))
    assert implied_vol(price, 100, 105, 0.5, 0.05, "call") == pytest.approx(0.27, abs=1e-3)


def test_iv_vectorized_roundtrips():
    F = np.array([22000.0, 22000, 22000]); K = np.array([21500.0, 22000, 22500])
    T = np.array([0.1, 0.1, 0.1]); r = np.full(3, 0.069)
    true = np.array([0.16, 0.13, 0.15]); is_call = np.array([True, True, False])
    px = np.array([float(black76_price(F[i], K[i], T[i], r[i], true[i],
                  "call" if is_call[i] else "put")) for i in range(3)])
    assert np.allclose(implied_vol_vec(px, F, K, T, r, is_call, q=r), true, atol=1e-3)


def test_no_iv_below_intrinsic():
    assert np.isnan(implied_vol(0.01, 100, 50, 1, 0.05, "call"))
