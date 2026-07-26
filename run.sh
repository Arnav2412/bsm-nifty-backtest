#!/usr/bin/env bash
# Full reproducible pipeline: real NSE data -> back-test -> charts -> report.
set -e
pip install -r requirements.txt
python src/parse_bhavcopy.py     # build clean dataset from data/raw_bhavcopy/
python src/backtest.py           # price, compare, metrics + charts -> results/
python src/generate_report.py    # results/BSM_Nifty_Report.pdf
pytest -q                        # validate the pricing library
echo "Done. See results/ (summary.txt, charts, BSM_Nifty_Report.pdf)"
