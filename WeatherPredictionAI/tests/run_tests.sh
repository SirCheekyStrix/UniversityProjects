#!/bin/bash
# run_tests.sh — uruchamia wszystkie testy z podsumowaniem
#
# Użycie:
#   ./run_tests.sh              # wszystkie testy bez API
#   ./run_tests.sh --api        # z testami API (wymaga działającego serwera)
#   ./run_tests.sh --slow       # z testami Open-Meteo (wolniejsze)
#   ./run_tests.sh --all        # wszystko

set -e

echo "========================================"
echo "  WeatherPredictionAI — Testy"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

cd "$(dirname "$0")"

# Instalacja zależności testowych
pip install pytest pytest-mock -q 2>/dev/null || true

RUN_API=0
RUN_SLOW=0

for arg in "$@"; do
    case $arg in
        --api)  RUN_API=1  ;;
        --slow) RUN_SLOW=1 ;;
        --all)  RUN_API=1; RUN_SLOW=1 ;;
    esac
done

# Buduj parametry pytest
PYTEST_ARGS="-v --tb=short"

if [ $RUN_API -eq 0 ] && [ $RUN_SLOW -eq 0 ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m 'not api and not slow'"
elif [ $RUN_API -eq 1 ] && [ $RUN_SLOW -eq 0 ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
elif [ $RUN_API -eq 0 ] && [ $RUN_SLOW -eq 1 ]; then
    PYTEST_ARGS="$PYTEST_ARGS -m 'not api'"
fi

echo ""
echo "Konfiguracja:"
echo "  Testy API:    $([ $RUN_API  -eq 1 ] && echo 'TAK' || echo 'NIE (--api aby włączyć)')"
echo "  Testy wolne:  $([ $RUN_SLOW -eq 1 ] && echo 'TAK' || echo 'NIE (--slow aby włączyć)')"
echo ""

# Uruchom testy jednostkowe
echo "─── Testy jednostkowe ───────────────────"
python3 -m pytest test_unit.py $PYTEST_ARGS 2>&1
UNIT_EXIT=$?

# Uruchom testy integracyjne
echo ""
echo "─── Testy integracyjne ──────────────────"
if [ $RUN_API -eq 1 ]; then
    python3 -m pytest test_integration.py $PYTEST_ARGS 2>&1
else
    python3 -m pytest test_integration.py $PYTEST_ARGS 2>&1
fi
INT_EXIT=$?

echo ""
echo "========================================"
echo "  WYNIKI:"
echo "  Jednostkowe:    $([ $UNIT_EXIT -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  Integracyjne:   $([ $INT_EXIT  -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "========================================"

# Zwróć 1 jeśli którykolwiek test się nie powiódł
[ $UNIT_EXIT -eq 0 ] && [ $INT_EXIT -eq 0 ] && exit 0 || exit 1
