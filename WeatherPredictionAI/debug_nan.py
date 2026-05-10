"""
debug_nan.py — uruchom żeby znaleźć NaN/inf w danych przed treningiem.
    python debug_nan.py
"""
import sys
import numpy as np
from weather_model import fetch_training_data, add_features, FEATURES_IN, TARGETS
from sklearn.preprocessing import MinMaxScaler

print("Pobieram 1 rok danych (szybki test)...")
df = fetch_training_data(years_back=1)
df = add_features(df)

print(f"\nKolumny DataFrame: {list(df.columns)}")

issues = False
print("\n=== Sprawdzam FEATURES_IN ===")
for col in FEATURES_IN:
    if col not in df.columns:
        print(f"  BRAK KOLUMNY: {col}")
        issues = True
        continue
    nan_n = df[col].isna().sum()
    inf_n = np.isinf(df[col].values.astype(float)).sum()
    if nan_n > 0 or inf_n > 0:
        print(f"  PROBLEM {col}: NaN={nan_n} inf={inf_n} min={df[col].min():.2f} max={df[col].max():.2f}")
        issues = True

print("\n=== Sprawdzam TARGETS ===")
for col in TARGETS:
    if col not in df.columns:
        print(f"  BRAK KOLUMNY: {col}")
        issues = True
        continue
    nan_n = df[col].isna().sum()
    inf_n = np.isinf(df[col].values.astype(float)).sum()
    if nan_n > 0 or inf_n > 0:
        print(f"  PROBLEM {col}: NaN={nan_n} inf={inf_n} min={df[col].min():.2f} max={df[col].max():.2f}")
        issues = True

print("\n=== Test skalowania ===")
try:
    sc = MinMaxScaler(feature_range=(0.05, 0.95))
    X = sc.fit_transform(df[FEATURES_IN])
    nan_after = np.isnan(X).sum()
    inf_after = np.isinf(X).sum()
    print(f"  Skalowanie wejść: NaN={nan_after} inf={inf_after}")
    if nan_after > 0 or inf_after > 0:
        for i, col in enumerate(FEATURES_IN):
            if np.isnan(X[:, i]).any() or np.isinf(X[:, i]).any():
                print(f"    Problematyczna kolumna: {col}")
except Exception as e:
    print(f"  BŁĄD skalowania: {e}")

for col in TARGETS:
    try:
        sc = MinMaxScaler(feature_range=(0.05, 0.95))
        y = sc.fit_transform(df[[col]])
        nan_n = np.isnan(y).sum()
        inf_n = np.isinf(y).sum()
        if nan_n > 0 or inf_n > 0:
            print(f"  PROBLEM po skalowaniu {col}: NaN={nan_n} inf={inf_n}")
            issues = True
    except Exception as e:
        print(f"  BŁĄD skalowania {col}: {e}")
        issues = True

if not issues:
    print("\nWszystko OK — problem leży w architekturze modelu, nie w danych")
else:
    print("\nZnaleziono problemy — napraw powyższe błędy przed treningiem")