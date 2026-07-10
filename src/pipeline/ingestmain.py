# # Data Ingestion
# 
# Load FD00X space-delimited CSV files as Polars, clean data, save as parquet files.

# %%
import pandas as pd
import polars as pl
from pathlib import Path
import os

# %%
# Paths like data/raw/... are relative to the project root, not this notebook folder
PROJECT_ROOT = next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not find project root (pyproject.toml)")

os.chdir(PROJECT_ROOT)
print(f"Working directory: {PROJECT_ROOT.resolve()}")


# %%
# Create Ingest Function
def ingest(file_path, dataset_id):
    
    #Define Column Names
    col_names = ['unit_number', 'cycle', 'op_set1', 'op_set2', 'op_set3', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14', 'sensor15', 'sensor16', 'sensor17', 'sensor18', 'sensor19', 'sensor20', 'sensor21']

    # Read space-delimited file
    pddf_RUL_FD00X = pd.read_csv(f'{file_path}/RUL_FD00{dataset_id}.txt', sep=r'\s+', header=None)
    pddf_train_FD00X = pd.read_csv(f'{file_path}/train_FD00{dataset_id}.txt', sep=r'\s+', header=None, names=col_names)
    pddf_test_FD00X = pd.read_csv(f'{file_path}/test_FD00{dataset_id}.txt', sep=r'\s+', header=None, names=col_names)

    # Convert pandas -> Polars
    df_RUL_FD00X = (
        pl.from_pandas(pddf_RUL_FD00X)
    .rename({"0": "RUL"})
    .with_row_index("unit_number", offset=1)
    )
    df_train_FD00X = pl.from_pandas(pddf_train_FD00X)
    df_test_FD00X = pl.from_pandas(pddf_test_FD00X)

    # Add split and dataset_id Column
    df_train_FD00X = df_train_FD00X.with_columns(pl.lit('train').alias('split'))
    df_test_FD00X = df_test_FD00X.with_columns(pl.lit('test').alias('split'))
    
    df_RUL_FD00X = df_RUL_FD00X.with_columns((pl.lit(f'FD00{dataset_id}').alias('dataset_id')))
    df_train_FD00X = df_train_FD00X.with_columns((pl.lit(f'FD00{dataset_id}').alias('dataset_id')))
    df_test_FD00X = df_test_FD00X.with_columns((pl.lit(f'FD00{dataset_id}').alias('dataset_id')))


    return df_RUL_FD00X, df_train_FD00X, df_test_FD00X


# %%
# Run Ingest Function
df_RUL_FD001, df_train_FD001, df_test_FD001 = ingest("data/raw", 1)

df_RUL_FD002, df_train_FD002, df_test_FD002 = ingest("data/raw", 2)

df_RUL_FD003, df_train_FD003, df_test_FD003 = ingest("data/raw", 3)

df_RUL_FD004, df_train_FD004, df_test_FD004 = ingest("data/raw", 4)


# %%
# Check Schema Uniformity
schemas = [df_train_FD001.columns, df_train_FD002.columns, df_train_FD003.columns, df_train_FD004.columns]
assert all(s == schemas[0] for s in schemas), "Schemas diverge across subsets"

schemas = [df_test_FD001.columns, df_test_FD002.columns, df_test_FD003.columns, df_test_FD004.columns]
assert all(s == schemas[0] for s in schemas), "Schemas diverge across subsets"

# %%
# Concatenate DataFrames
df_RUL = pl.concat([df_RUL_FD001, df_RUL_FD002, df_RUL_FD003, df_RUL_FD004])
df_train = pl.concat([df_train_FD001, df_train_FD002, df_train_FD003, df_train_FD004])
df_test = pl.concat([df_test_FD001, df_test_FD002, df_test_FD003, df_test_FD004])

# Write Parquets
df_RUL.write_parquet(f'data/clean/RUL.parquet')
df_train.write_parquet(f'data/clean/train.parquet')
df_test.write_parquet(f'data/clean/test.parquet')


