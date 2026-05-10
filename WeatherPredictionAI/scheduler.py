"""
scheduler.py — tygodniowy trening modeli.

Harmonogram:
    Niedziela 02:00 — trening PatchTST (patchtst_train.py)
    Niedziela 03:00 — trening ryb (fish_train.py)
    Niedziela 05:00 — trening TFT (tft_train.py)

Predykcje są teraz obsługiwane przez api_server.py (REST API na porcie 8000).

Użycie:
    python scheduler.py              # uruchom harmonogram
    python scheduler.py --now-train  # natychmiastowy trening wszystkiego

Cron (alternatywa):
    0 2 * * 0  cd /home/ec2-user/WeatherPredictionAI && python patchtst_train.py >> logs/patchtst_train.log 2>&1
    0 3 * * 0  cd /home/ec2-user/WeatherPredictionAI && python fish_train.py     >> logs/fish_train.log     2>&1
    0 5 * * 0  cd /home/ec2-user/WeatherPredictionAI && python tft_train.py      >> logs/tft_train.log      2>&1
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(LOG_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def run_script(script: str, extra_args=None) -> int:
    cmd      = [sys.executable, "-u", script] + (extra_args or [])
    name     = os.path.basename(script).replace(".py", "")
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    log(f"START: {' '.join(os.path.basename(c) for c in cmd)}")
    with open(log_file, "a") as lf:
        lf.write(f"\n{'='*60}\n[{datetime.now().isoformat()}]\n")
        proc = subprocess.run(cmd, stdout=lf, stderr=lf, cwd=BASE_DIR)
    if proc.returncode != 0:
        log(f"BLAD ({proc.returncode}): {script} — sprawdz {log_file}")
    else:
        log(f"OK: {script}")
    return proc.returncode


def train_all():
    log("=== TRENING WSZYSTKICH MODELI ===")
    run_script(os.path.join(BASE_DIR, "patchtst_train.py"))
    run_script(os.path.join(BASE_DIR, "fish_train.py"))
    run_script(os.path.join(BASE_DIR, "tft_train.py"))
    log("=== TRENING ZAKONCZONY ===")


def run_scheduler():
    log("Scheduler uruchomiony — trening co niedziele")
    log("Predykcje: api_server.py (REST API na porcie 8000)")

    while True:
        now = datetime.now()

        # Niedziela (weekday=6) o 02:00 — PatchTST
        if now.weekday() == 6 and now.hour == 2 and now.minute == 0:
            log("Niedziela 02:00 — trening PatchTST")
            run_script(os.path.join(BASE_DIR, "patchtst_train.py"))

        # Niedziela 03:00 — ryby
        if now.weekday() == 5 and now.hour == 23 and now.minute == 0:
            log("Sobota 23:00 — trening Fish")
            run_script(os.path.join(BASE_DIR, "fish_train.py"))

        # Niedziela 05:00 — TFT
        if now.weekday() == 5 and now.hour == 23 and now.minute == 30:
            log("Sobota 23:30 — trening TFT")
            run_script(os.path.join(BASE_DIR, "tft_train.py"))

        # Codziennie 01:00 — aktualizacja danych historycznych
        if now.hour == 1 and now.minute == 0:
            log("01:00 — aktualizacja InfluxDB")
            run_script(os.path.join(BASE_DIR, "openmeteo_to_influx.py"), ["--recent"])

        time.sleep(55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--now-train", action="store_true")
    args = parser.parse_args()

    if args.now_train:
        train_all()
    else:
        run_scheduler()
