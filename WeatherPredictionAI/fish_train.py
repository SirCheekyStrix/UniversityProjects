"""
fish_train.py — trenuje modele LightGBM dla wszystkich gatunków ryb.

Użycie:
    python fish_train.py
    python fish_train.py --years 5
    python fish_train.py --fitzroy /path/to/Fitzroy_Basin.csv
"""
import argparse
import os
from datetime import datetime

import os
from model_swap import TrainingContext
from fish_model import SPECIES, MODEL_DIR, train_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",   type=int, default=3)
    parser.add_argument("--fitzroy", type=str, default=None,
                        help="Sciezka do pliku CSV Fitzroy Basin (fine-tuning)")
    args = parser.parse_args()

    # Sprawdź domyślną ścieżkę Fitzroy jeśli nie podano
    if args.fitzroy is None:
        default_path = os.path.join(os.path.dirname(__file__), "Fitzroy_Basin.csv")
        if os.path.exists(default_path):
            args.fitzroy = default_path
            print(f"Znaleziono Fitzroy Basin: {default_path}")

    import fish_model as _fm
    with TrainingContext(MODEL_DIR) as ctx:
        _fm.MODEL_DIR = ctx.tmp_dir
        train_all(years_back=args.years, fitzroy_csv=args.fitzroy)
        _fm.MODEL_DIR = MODEL_DIR
    print("Modele ryb podmienione atomowo")