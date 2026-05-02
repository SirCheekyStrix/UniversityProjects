import pandas as pd

df = pd.read_parquet("data/station_timeseries.parquet")

print(df.head(20))
print(df.info())
print(df.columns.tolist())