import polars as pl
from sqlalchemy import Engine
import util.s3 as s3
import util.sql_queries as sql

# Retrieve data from s3 and keep required data
def get_text(engine: Engine, table_name: str, batch: int | None = None, token: str | None = None) -> pl.LazyFrame:
    # Get all text columns
    text_cols = sql.get_text_cols(engine, table_name)
    
    # Download raw data from s3
    if batch is None:
        # Initial processing - read CSV from temp_uploads
        df_raw = s3.download("temp_uploads", table_name, batch, token)
    else:
        # Batch processing - read parquet from datasets
        df_raw = s3.download("datasets", table_name, batch, token)

    # Reformat data to prep for preprocessing
    df_list: list[pl.LazyFrame] = []
    for col in text_cols:
        df_list.append((
            df_raw
                .select([col, "record_id"])
                .rename({ f"{col}": "text" })
                .with_columns(col=pl.lit(col))
                .cast({ "text": pl.Utf8 })
        ))
    df = pl.concat(df_list)
    
    return df

# Get the values of a subset of columns for each record
def get_columns(table_name: str, columns: list[str], batch: int | None = None, token: str | None = None) -> pl.LazyFrame:
    if batch is None:
        df = s3.download("temp_uploads", table_name, batch, token)
    else:
        df = s3.download("datasets", table_name, batch, token)
    
    return df.select(["record_id"] + columns)
