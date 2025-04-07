import boto3
from boto3.s3.transfer import TransferConfig
import datetime as dt
import humanize
import jwt
import os
import pandas as pd
import polars as pl
from time import time

BASE_PATH = "files/s3"
BASE_PATH = "files/s3"

# Use TransferConfig to optimize the download
config = TransferConfig(
    multipart_threshold=1024 * 500,  # 100MB
    max_concurrency=10,
    multipart_chunksize=1024 * 500,  # 100MB
    use_threads=True
)

def get_creds(token: str | None = None) -> dict[str, str]:
    if token == None:
        return {
            "region": os.environ.get("S3_REGION"),
            "bucket": os.environ.get("S3_BUCKET"),
            "key_": os.environ.get("S3_KEY"),
            "secret": os.environ.get("S3_SECRET")
        }
        
    secret = os.environ.get("TOKEN_SECRET")
    return jwt.decode(token, secret, "HS256")

def upload(df: pl.DataFrame | pd.DataFrame, folder: str, name: str, batch: int | None = None, token: str | None = None) -> None:
    distributed = get_creds(token)
    
    # Convert file to parquet
    start_time = time()
    local_file = "{}/{}/{}.parquet".format(BASE_PATH, folder, name)
    if type(df) == pl.DataFrame:
        df.write_parquet(local_file, use_pyarrow=True, compression="zstd")
    elif type(df) == pd.DataFrame:
        df.to_parquet(local_file, "pyarrow", index = False, compression = "zstd")
    print("Conversion time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start_time))))
    
    # Upload file to s3
    if "key_" in distributed.keys() and "secret" in distributed.keys():
        s3_client = boto3.client(
            "s3",
            aws_access_key_id = distributed["key_"],
            aws_secret_access_key = distributed["secret"],
            region_name = distributed["region"]
        )
    else:
        s3_client = boto3.client(
            "s3",
            region_name = distributed["region"]
        )
        
    if batch is None:
        path = "tables/{}_{}/{}.parquet".format(folder, name, name)
    else:
        path = "tables/{}_{}/{}_{}.parquet".format(folder, name, name, batch)
        
    start_time = time()
    s3_client.upload_file(
        local_file,
        distributed["bucket"],
        path,
        Config = config
    )
    print("Upload time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start_time))))
    
def upload_file(local_folder: str, s3_folder: str, name: str, token: str | None = None) -> None:
    distributed = get_creds(token)

    # Upload file to s3
    if "key_" in distributed.keys() and "secret" in distributed.keys():
        s3_client = boto3.client(
            "s3",
            aws_access_key_id = distributed["key_"],
            aws_secret_access_key = distributed["secret"],
            region_name = distributed["region"]
        )
    else:
        s3_client = boto3.client(
            "s3",
            region_name = distributed["region"]
        )
        
    local_file = "{}/{}/{}".format(BASE_PATH, local_folder, name)
    path = "{}/{}".format(s3_folder, name)
      
    start_time = time()  
    s3_client.upload_file(
        local_file,
        distributed["bucket"],
        path,
        Config = config
    )
    print("Upload time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start_time))))
    
def download(folder: str, name: str, batch: int | None = None, token: str | None = None) -> pl.LazyFrame:
    distributed = get_creds(token)
    
    if folder == "temp_uploads":
        if batch is None:
            path = "temp_uploads/{}.csv".format( name)
        else:
            path = "temp_uploads/{}_{}.csv".format(name, batch)
    else:
        if batch is None:
            path = "tables/{}_{}/{}.csv".format(folder, name, name)
        else:
            path = "tables/{}_{}/{}_{}.csv".format(folder, name, name, batch)
        
    storage_options = {
        "aws_access_key_id": distributed["key_"],
        "aws_secret_access_key": distributed["secret"],
        "aws_region": distributed["region"],
    }
    s3_path = "s3://{}/{}".format(distributed["bucket"], path)
    df = pl.scan_csv(s3_path, storage_options=storage_options)
    if folder == "tokens":
        df = df.with_columns(
            record_id = pl.col("record_id").cast(pl.UInt32, strict = False)
        )
    elif folder == "temp_uploads":
        # Add record id column
        df = (
            df.with_row_index("record_id")
                .with_columns(
                    record_id = pl.col("record_id").cast(pl.UInt32, strict = False)
                )
        )
    
    return df

def download_file(local_file: str, folder: str, name: str, token: str | None = None):
    distributed = get_creds(token)
    
    if os.path.exists(local_file):
        # Do nothing if file already downloaded
        print("{} already exists".format(local_file))
    else:
        # Download file from s3
        if "key_" in distributed.keys() and "secret" in distributed.keys():
            s3_client = boto3.client(
                "s3",
                aws_access_key_id = distributed["key_"],
                aws_secret_access_key = distributed["secret"],
                region_name = distributed["region"]
            )
        else:
            s3_client = boto3.client(
                "s3",
                region_name = distributed["region"]
            )
        path = "{}/{}".format(folder, name)
        
        start_time = time()
        s3_client.download_file(
            distributed["bucket"],
            path,
            local_file,
            Config = config
        )
        print("Download time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start_time))))
        
def delete_temp_dataset(table_name: str, batch: int | None = None, token: str | None = None):
    distributed = get_creds(token)
    
    if "key_" in distributed.keys() and "secret" in distributed.keys():
        s3_client = boto3.client(
            "s3",
            aws_access_key_id = distributed["key_"],
            aws_secret_access_key = distributed["secret"],
            region_name = distributed["region"]
        )
    else:
        s3_client = boto3.client(
            "s3",
            region_name = distributed["region"]
        )
        
    if batch is None:
        path = f"temp_uploads/{ table_name }.csv"
    else:
        path = f"temp_uploads/{ table_name }_{ batch }.csv"
        
    s3_client.delete_object(Bucket = distributed["bucket"], Key = path)

def delete_stopwords(table_name: str, token: str | None = None):
    distributed = get_creds(token)
    
    if "key_" in distributed.keys() and "secret" in distributed.keys():
        s3_client = boto3.client(
            "s3",
            aws_access_key_id = distributed["key_"],
            aws_secret_access_key = distributed["secret"],
            region_name = distributed["region"]
        )
    else:
        s3_client = boto3.client(
            "s3",
            region_name = distributed["region"]
        )
        
    s3_client.delete_object(Bucket = distributed["bucket"], Key = f"stopwords/{ table_name }.txt")
    
def delete_embeddings(table_name: str, token: str | None = None):
    distributed = get_creds(token)
    
    if "key_" in distributed.keys() and "secret" in distributed.keys():
        s3_client = boto3.client(
            "s3",
            aws_access_key_id = distributed["key_"],
            aws_secret_access_key = distributed["secret"],
            region_name = distributed["region"]
        )
    else:
        s3_client = boto3.client(
            "s3",
            region_name = distributed["region"]
        )
        
    embed_directory = f"embeddings/{ table_name }/"
    # List and delete all objects under the embed_directory prefix
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=distributed["bucket"], Prefix=embed_directory):
        if "Contents" in page:
            # Prepare the list of objects to delete
            delete_objects = [{"Key": obj["Key"]} for obj in page["Contents"]]
            # Batch delete the objects
            s3_client.delete_objects(Bucket=distributed["bucket"], Delete={"Objects": delete_objects})
    
    print(f"All objects in the directory '{embed_directory}' have been deleted.")