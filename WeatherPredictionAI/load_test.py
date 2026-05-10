#!/usr/bin/env python3
"""
load_test.py — testy obciążeniowe API serwera WeatherPredictionAI.

Użycie:
    python load_test.py                        # domyślnie: 10 users, patchtst+fish
    python load_test.py --users 50             # 50 równoległych requestów
    python load_test.py --users 1,10,50,100,1000  # pełne testy skalowania
    python load_test.py --models patchtst,fish # które modele testować
    python load_test.py --report               # zapisz raport do JSON

Wymagania:
    pip install requests
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────

BASE_URL = "http://localhost:8000"

# Testowe zbiorniki z różnych lokalizacji Polski
TEST_RESERVOIRS = [
    {"id": 1,  "name": "Zalew Zegrzyński",     "lat": 52.4763, "lon": 21.0532},
    {"id": 2,  "name": "Jezioro Śniardwy",     "lat": 53.7500, "lon": 21.7200},
    {"id": 3,  "name": "Zalew Wiślany",        "lat": 54.3520, "lon": 19.5000},
    {"id": 4,  "name": "Jeziorko Czerniakowskie", "lat": 52.1800, "lon": 21.0800},
    {"id": 5,  "name": "Jezioro Wigry",        "lat": 54.0200, "lon": 23.0700},
    {"id": 6,  "name": "Zbiornik Solina",      "lat": 49.3700, "lon": 22.4500},
    {"id": 7,  "name": "Jezioro Gopło",        "lat": 52.5700, "lon": 18.2800},
    {"id": 8,  "name": "Zbiornik Czaniec",     "lat": 49.8900, "lon": 19.1400},
    {"id": 9,  "name": "Jezioro Mamry",        "lat": 54.1500, "lon": 21.7300},
    {"id": 10, "name": "Zbiornik Dobczyce",    "lat": 49.8700, "lon": 20.0800},
    {"id": 11, "name": "Jezioro Drawsko",      "lat": 53.6800, "lon": 15.8100},
    {"id": 12, "name": "Zbiornik Koronowski",  "lat": 53.3200, "lon": 17.9400},
    {"id": 13, "name": "Jezioro Niegocin",     "lat": 54.0300, "lon": 21.7900},
    {"id": 14, "name": "Zbiornik Nyski",       "lat": 50.4800, "lon": 17.3300},
    {"id": 15, "name": "Jezioro Charzykowskie","lat": 53.7900, "lon": 17.5000},
]


# ─────────────────────────────────────────────
# FUNKCJE POMOCNICZE
# ─────────────────────────────────────────────

def check_health() -> bool:
    """Sprawdź czy API jest dostępne."""
    try:
        import requests
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        print(f"  Status API: {data['status']}")
        for model, info in data.get('models', {}).items():
            status = "✓" if info.get('available') else "✗"
            print(f"  {status} {model}: {'dostępny' if info.get('available') else 'NIEDOSTĘPNY'}")
        return data['status'] == 'ok'
    except Exception as e:
        print(f"  BŁĄD połączenia: {e}")
        return False


def single_request(reservoir: dict, models: str) -> dict:
    """Wykonuje jeden request i zwraca metryki."""
    import requests

    t_start = time.perf_counter()
    result  = {
        "reservoir_id":   reservoir["id"],
        "reservoir_name": reservoir["name"],
        "lat":            reservoir["lat"],
        "lon":            reservoir["lon"],
        "models":         models,
        "success":        False,
        "elapsed_ms":     0,
        "models_run":     [],
        "errors":         [],
        "http_status":    None,
    }

    try:
        r = requests.get(
            f"{BASE_URL}/predict",
            params={
                "latitude":       reservoir["lat"],
                "longitude":      reservoir["lon"],
                "reservoir_id":   reservoir["id"],
                "reservoir_name": reservoir["name"],
                "models":         models,
            },
            timeout=120,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result["elapsed_ms"]  = round(elapsed_ms, 1)
        result["http_status"] = r.status_code

        if r.status_code == 200:
            data = r.json()
            result["models_run"] = data.get("models_run", [])
            result["errors"]     = data.get("errors", [])
            result["success"]    = len(data.get("models_run", [])) > 0
        else:
            result["errors"] = [{"error": f"HTTP {r.status_code}"}]

    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result["elapsed_ms"] = round(elapsed_ms, 1)
        result["errors"]     = [{"error": "TIMEOUT"}]
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result["elapsed_ms"] = round(elapsed_ms, 1)
        result["errors"]     = [{"error": str(e)}]

    return result


def run_load_test(n_users: int, models: str, reservoirs: list) -> dict:
    """Uruchamia test obciążeniowy z n_users równoległymi requestami."""

    # Wybierz zbiorniki (z powtórzeniami jeśli więcej użytkowników niż zbiorników)
    selected = []
    for i in range(n_users):
        selected.append(reservoirs[i % len(reservoirs)])

    # Losowa kolejność żeby symulować różnych użytkowników
    random.shuffle(selected)

    print(f"\n  Uruchamiam {n_users} równoległych requestów ({models})...", flush=True)
    t_wall_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=min(n_users, 50)) as executor:
        futures = [executor.submit(single_request, r, models) for r in selected]
        for future in as_completed(futures):
            results.append(future.result())

    t_wall = (time.perf_counter() - t_wall_start) * 1000

    # Analiza wyników
    successes    = [r for r in results if r["success"]]
    failures     = [r for r in results if not r["success"]]
    elapsed_list = [r["elapsed_ms"] for r in results]

    stats = {
        "n_users":       n_users,
        "models":        models,
        "total_ms":      round(t_wall, 0),
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate":  round(len(successes) / n_users * 100, 1),
        "throughput_rps": round(n_users / (t_wall / 1000), 2),
    }

    if elapsed_list:
        stats["latency"] = {
            "min_ms":    round(min(elapsed_list), 0),
            "max_ms":    round(max(elapsed_list), 0),
            "mean_ms":   round(statistics.mean(elapsed_list), 0),
            "median_ms": round(statistics.median(elapsed_list), 0),
            "p95_ms":    round(sorted(elapsed_list)[int(len(elapsed_list)*0.95)], 0),
            "p99_ms":    round(sorted(elapsed_list)[int(len(elapsed_list)*0.99)], 0),
        }

    if failures:
        error_types = {}
        for f in failures:
            for e in f["errors"]:
                key = e.get("error", "unknown")[:50]
                error_types[key] = error_types.get(key, 0) + 1
        stats["error_types"] = error_types

    return stats, results


def print_stats(stats: dict):
    """Wyświetla statystyki w czytelnym formacie."""
    ok  = "✓" if stats["success_rate"] >= 95 else ("⚠" if stats["success_rate"] >= 80 else "✗")
    lat = stats.get("latency", {})

    print(f"\n  {'─'*55}")
    print(f"  {ok} Użytkownicy:    {stats['n_users']}")
    print(f"  ✓ Sukces:          {stats['success_count']}/{stats['n_users']} ({stats['success_rate']}%)")
    if stats['failure_count'] > 0:
        print(f"  ✗ Błędy:          {stats['failure_count']}")
        for err, cnt in stats.get("error_types", {}).items():
            print(f"      {cnt}× {err}")
    print(f"  ⏱ Całkowity czas:  {stats['total_ms']:.0f}ms")
    print(f"  ⚡ Przepustowość:  {stats['throughput_rps']} req/s")
    if lat:
        print(f"  📊 Latencja:")
        print(f"      min:    {lat['min_ms']:.0f}ms")
        print(f"      mediana:{lat['median_ms']:.0f}ms")
        print(f"      średnia:{lat['mean_ms']:.0f}ms")
        print(f"      p95:    {lat['p95_ms']:.0f}ms")
        print(f"      p99:    {lat['p99_ms']:.0f}ms")
        print(f"      max:    {lat['max_ms']:.0f}ms")


def print_summary_table(all_stats: list):
    """Tabela porównawcza wszystkich testów."""
    print(f"\n{'═'*75}")
    print(f"  PODSUMOWANIE TESTÓW OBCIĄŻENIOWYCH")
    print(f"{'═'*75}")
    print(f"  {'Users':>6}  {'Success%':>9}  {'Median':>8}  {'p95':>8}  {'p99':>8}  {'RPS':>8}  {'Status':>8}")
    print(f"  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for s in all_stats:
        lat = s.get("latency", {})
        sr  = s["success_rate"]
        status = "✓ OK" if sr >= 95 else ("⚠ DEGRADED" if sr >= 80 else "✗ FAIL")
        print(
            f"  {s['n_users']:>6}  "
            f"  {sr:>7.1f}%  "
            f"  {lat.get('median_ms', 0):>6.0f}ms  "
            f"  {lat.get('p95_ms', 0):>6.0f}ms  "
            f"  {lat.get('p99_ms', 0):>6.0f}ms  "
            f"  {s['throughput_rps']:>6.2f}  "
            f"  {status}"
        )

    print(f"{'═'*75}")

    # Wnioski
    print(f"\n  WNIOSKI:")
    stable = [s for s in all_stats if s["success_rate"] >= 95]
    degraded = [s for s in all_stats if 80 <= s["success_rate"] < 95]
    failing  = [s for s in all_stats if s["success_rate"] < 80]

    if stable:
        max_stable = max(s["n_users"] for s in stable)
        print(f"  ✓ Serwer stabilny do {max_stable} równoległych użytkowników (≥95% sukces)")
    if degraded:
        print(f"  ⚠ Degradacja przy: {[s['n_users'] for s in degraded]} użytkownikach")
    if failing:
        print(f"  ✗ Przeciążenie przy: {[s['n_users'] for s in failing]} użytkownikach")

    # Zalecenia
    print(f"\n  ZALECENIA:")
    if stable:
        max_stable = max(s["n_users"] for s in stable)
        med = next(s["latency"]["median_ms"] for s in all_stats if s["n_users"] == max_stable)
        print(f"  - Maksymalne bezpieczne obciążenie: {max_stable} równoległych requestów")
        print(f"  - Mediana czasu odpowiedzi przy {max_stable} userach: {med:.0f}ms")
    if any(s["n_users"] >= 50 and s["success_rate"] < 95 for s in all_stats):
        print(f"  - Rozważ cache po stronie aplikacji (TTL=1h)")
        print(f"  - Rozważ kolejkowanie requestów (max 5 równolegle)")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Testy obciążeniowe WeatherPredictionAI API")
    parser.add_argument("--url",     default="http://localhost:8080",
                        help="Base URL serwera (domyślnie: http://localhost:8080)")
    parser.add_argument("--users",   default="1,10,50,100",
                        help="Liczba równoległych użytkowników (np. '1,10,50,100,1000')")
    parser.add_argument("--models",  default="patchtst,fish",
                        help="Modele do testowania (domyślnie: patchtst,fish)")
    parser.add_argument("--report",  action="store_true",
                        help="Zapisz pełny raport do JSON")
    parser.add_argument("--quick",   action="store_true",
                        help="Szybki test: tylko 1,10,50 userów")
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    print(f"\n{'═'*60}")
    print(f"  WeatherPredictionAI — Testy Obciążeniowe")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  URL: {BASE_URL}")
    print(f"  Modele: {args.models}")
    print(f"{'═'*60}")

    # Health check
    print("\n[1/3] Sprawdzam dostępność API...")
    if not check_health():
        print("\nERROR: API niedostępne. Sprawdź czy serwer jest uruchomiony.")
        sys.exit(1)

    # Parsuj poziomy użytkowników
    if args.quick:
        user_levels = [1, 10, 50]
    else:
        user_levels = []
        for u in args.users.split(","):
            u = u.strip()
            if u.isdigit():
                user_levels.append(int(u))

    user_levels = sorted(set(user_levels))
    print(f"\n[2/3] Poziomy testowe: {user_levels}")
    print(f"      Zbiorniki: {len(TEST_RESERVOIRS)} lokalizacji")

    # Warmup
    print("\n[3/3] Warmup (1 request)...", end=" ", flush=True)
    warmup = single_request(TEST_RESERVOIRS[0], args.models)
    if warmup["success"]:
        print(f"OK ({warmup['elapsed_ms']:.0f}ms)")
    else:
        print(f"BŁĄD: {warmup['errors']}")
        print("Kontynuuję mimo błędu warmup...")

    # Testy
    all_stats   = []
    all_results = []

    for n_users in user_levels:
        print(f"\n{'─'*60}")
        print(f"  TEST: {n_users} równoległych użytkowników")

        stats, results = run_load_test(n_users, args.models, TEST_RESERVOIRS)
        all_stats.append(stats)
        all_results.extend(results)

        print_stats(stats)

        # Przerwa między testami (pozwól serwerowi odpocząć)
        if n_users < user_levels[-1]:
            cooldown = min(n_users // 10 + 2, 15)
            print(f"\n  Cooldown {cooldown}s...", end=" ", flush=True)
            time.sleep(cooldown)
            print("OK")

    # Podsumowanie
    print_summary_table(all_stats)

    # Raport JSON
    if args.report:
        report = {
            "generated_at": datetime.now().isoformat(),
            "base_url":     BASE_URL,
            "models":       args.models,
            "user_levels":  user_levels,
            "summary":      all_stats,
            "raw_results":  all_results,
        }
        report_path = f"load_test_report_{datetime.now():%Y%m%d_%H%M}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Raport zapisany: {report_path}")


if __name__ == "__main__":
    main()