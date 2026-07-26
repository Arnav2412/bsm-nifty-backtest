import requests
import time
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2025, 12, 31)

BASE_FOLDER = Path("data/raw_options")
BASE_FOLDER.mkdir(parents=True, exist_ok=True)

# NSE discontinued the old F&O bhavcopy from 8 July 2024. 
UDIFF_START = datetime(2024, 7, 8)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
})


# ============================================================
# OLD NSE BHAVCOPY
#
# Used before 8 July 2024.
#
# Example:
# fo04JAN2021bhav.csv.zip
# ============================================================

def download_old_bhavcopy(date):

    year = date.strftime("%Y")
    month = date.strftime("%b").upper()

    filename = f"fo{date.strftime('%d%b%Y').upper()}bhav.csv.zip"

    url = (
        "https://nsearchives.nseindia.com/"
        f"content/historical/DERIVATIVES/"
        f"{year}/{month}/{filename}"
    )

    folder = BASE_FOLDER / year
    folder.mkdir(parents=True, exist_ok=True)

    output = folder / filename

    # Don't download again if already present
    if output.exists():
        print(f"[EXISTS]      {date.date()}")
        return True

    response = session.get(url, timeout=30)

    if response.status_code == 200:

        # Basic protection against saving an HTML error page
        if len(response.content) < 1000:
            return False

        output.write_bytes(response.content)

        print(
            f"[DOWNLOADED]  {date.date()} "
            f"OLD "
            f"({len(response.content) / 1024:.1f} KB)"
        )

        return True

    return False


# ============================================================
# NEW NSE UDIFF BHAVCOPY
#
# Used from 8 July 2024 onward.
#
# Example:
# BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip
# ============================================================

def download_udiff_bhavcopy(date):

    year = date.strftime("%Y")
    date_string = date.strftime("%Y%m%d")

    filename = (
        f"BhavCopy_NSE_FO_0_0_0_"
        f"{date_string}_F_0000.csv.zip"
    )

    url = (
        "https://nsearchives.nseindia.com/"
        "content/fo/"
        f"{filename}"
    )

    folder = BASE_FOLDER / year
    folder.mkdir(parents=True, exist_ok=True)

    output = folder / filename

    if output.exists():
        print(f"[EXISTS]      {date.date()}")
        return True

    response = session.get(url, timeout=30)

    if response.status_code == 200:

        if len(response.content) < 1000:
            return False

        output.write_bytes(response.content)

        print(
            f"[DOWNLOADED]  {date.date()} "
            f"UDIFF "
            f"({len(response.content) / 1024:.1f} KB)"
        )

        return True

    return False


# ============================================================
# CHOOSE CORRECT FORMAT
# ============================================================

def download_bhavcopy(date):

    if date < UDIFF_START:
        return download_old_bhavcopy(date)

    return download_udiff_bhavcopy(date)


# ============================================================
# MAIN LOOP
# ============================================================

current = START_DATE

downloaded = 0
already_exists = 0
failed = []

print("\n============================================")
print(" NSE F&O BHAVCOPY DOWNLOADER")
print("============================================")
print(f"Start : {START_DATE.date()}")
print(f"End   : {END_DATE.date()}")
print("============================================\n")


while current <= END_DATE:

    # --------------------------------------------------------
    # Skip weekends
    # --------------------------------------------------------

    if current.weekday() >= 5:
        current += timedelta(days=1
        )
        continue

    # --------------------------------------------------------
    # Check whether expected file already exists
    # --------------------------------------------------------

    year_folder = BASE_FOLDER / current.strftime("%Y")

    old_filename = (
        f"fo{current.strftime('%d%b%Y').upper()}bhav.csv.zip"
    )

    udiff_filename = (
        f"BhavCopy_NSE_FO_0_0_0_"
        f"{current.strftime('%Y%m%d')}_F_0000.csv.zip"
    )

    if current < UDIFF_START:
        expected_file = year_folder / old_filename
    else:
        expected_file = year_folder / udiff_filename

    if expected_file.exists():

        print(f"[EXISTS]      {current.date()}")

        already_exists += 1

        current += timedelta(days=1)
        continue

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    try:

        success = download_bhavcopy(current)

        if success:
            downloaded += 1

        else:
            print(
                f"[NOT FOUND]   {current.date()} "
                "(holiday or unavailable)"
            )

            failed.append(current.strftime("%Y-%m-%d"))

    except requests.exceptions.RequestException as e:

        print(
            f"[NETWORK ERR]  {current.date()} -> {e}"
        )

        failed.append(current.strftime("%Y-%m-%d"))

    except Exception as e:

        print(
            f"[ERROR]        {current.date()} -> {e}"
        )

        failed.append(current.strftime("%Y-%m-%d"))

    # --------------------------------------------------------
    # Don't hammer NSE
    # --------------------------------------------------------

    time.sleep(0.35)

    current += timedelta(days=1)


# ============================================================
# SAVE FAILED DATES
# ============================================================

failed_file = BASE_FOLDER / "failed_dates.txt"

with open(failed_file, "w") as f:

    for date in failed:
        f.write(date + "\n")


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print(" DOWNLOAD COMPLETE")
print("============================================")

print(f"New downloads   : {downloaded}")
print(f"Already existed : {already_exists}")
print(f"Not found       : {len(failed)}")

print("============================================")

print(
    f"\nFailed dates saved to:\n{failed_file}"
)