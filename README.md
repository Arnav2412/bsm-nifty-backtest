# Assessing Black-Scholes Accuracy on Nifty50 Index Options
**FinSearch research project — Arnav Srivastava**

Does the Black-Scholes model actually price real options correctly? This project
tests it against **43,817 real traded Nifty50 option quotes** from official NSE
F&O bhavcopy data (Q1 2024) and shows exactly where and why it breaks down.

## Key finding
Black-Scholes explains the *level* of option prices almost perfectly (R² > 0.99)
but **systematically under-prices**, and the error blows up out-of-the-money. The
reason is the **volatility smile**: the market uses a different volatility for
every strike, while Black-Scholes assumes just one. Feeding it a GARCH volatility
forecast cuts pricing error by 22%, and fitting the smile cuts it by 41% —
showing the flaw is the flat-volatility *assumption*, not the formula itself.

> **Read `results/BSM_Nifty_Report.pdf` first** — the full write-up with charts.

## How to run
```bash
pip install -r requirements.txt
python src/parse_bhavcopy.py     # build the dataset from raw NSE bhavcopy
python src/backtest.py           # run the analysis, save charts + summary.txt
python src/generate_report.py    # build the PDF report
python -m pytest -q              # (optional) run the tests — 9 should pass
```
The results are already generated and committed, so you can also just open
`results/` directly without running anything.

## What to look at
| File | What it is |
|------|-----------|
| `results/BSM_Nifty_Report.pdf` | the research report (start here) |
| `results/summary.txt` | all the accuracy numbers |
| `results/volatility_smile.png` | the key chart — why BS is wrong |
| `results/rmse_by_model.png` | accuracy under realized / EWMA / GARCH vol |
| `src/` | the code |
| `data/raw_bhavcopy/` | the raw real NSE data it's built from |

## Method in one paragraph
Options are priced with Black-76 (the forward form of Black-Scholes). For every
expiry the forward and discount are recovered directly from put-call parity —
the CBOE/VIX approach — so no dividend assumption is needed and the forward is
consistent with the prices being tested. Volatility is estimated three ways
(historical, EWMA, GARCH), and a smile-fitted benchmark isolates how much of the
error is due to the constant-volatility assumption.

## Repo layout
```
├── README.md
├── requirements.txt
├── run.sh
├── data/
│   ├── raw_bhavcopy/         # real NSE F&O bhavcopy (60 trading days)
│   ├── nifty_options.csv     # clean dataset (built by parse_bhavcopy.py)
│   └── nifty_underlying.csv
├── src/
│   ├── black_scholes.py      # pricing, Greeks, implied-vol solvers
│   ├── volatility.py         # realized, EWMA, GARCH(1,1)
│   ├── parse_bhavcopy.py     # raw data -> clean dataset
│   ├── metrics.py            # error metrics + bucketing
│   ├── backtest.py           # the full back-test + charts
│   └── generate_report.py    # builds the PDF report
├── tests/                    # 9 unit tests
└── results/                  # charts, summary.txt, the PDF report
```

## Data source
Real NSE Futures & Options daily bhavcopy, filtered to Nifty index options that
actually traded (not stale settlement marks). 60 trading days, 20 expiries.

## License
MIT
