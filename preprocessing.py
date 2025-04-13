from time import time, sleep, tzname
start_time = time()
from dotenv import load_dotenv
load_dotenv()
import boto3
import datetime as dt
import humanize
import os
import sys
# Database interaction
import util.s3 as s3
# SQL helpers
from util.sql_connect import sql_connect
import util.data_queries as data
import util.sql_queries as sql
# Word processing
from util.embeddings_save import compute_embeddings, update_embeddings
from util.processing_functions import DataProcessing
       
def main():  
    # Get table name from command line argument
    TABLE_NAME = sys.argv[1]

    # Get number of threads to use from second command line argument with default of 1 if not provided or not int
    try:
        NUM_THREADS = int(sys.argv[2])
    except:
        NUM_THREADS = 1

    # Get distributed token if defined
    try:
        BATCH_NUM = int(sys.argv[3])
    except:
        BATCH_NUM = None

    engine, meta = sql_connect()

    # Get metadata to determine preprocessing type
    metadata = sql.get_metadata(engine, meta, TABLE_NAME)
     
    # Load formatted data from s3
    print("Loading data...") 
    load_time = time()
    df = data.get_text(engine, TABLE_NAME, BATCH_NUM).drop_nulls().collect()
    print("Load time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - load_time))))
    # Create object with processing functions
    processor = DataProcessing(metadata, NUM_THREADS, TABLE_NAME)
    # Run token preprocessing
    print("Processing tokens...")   
    df_split, df_split_raw = processor.split_text(df)
    print("Tokens processed: {}".format(len(df_split)))
    # Upload results
    print("Uploading tokens...")
    sql.deactivate_processing(engine, TABLE_NAME, "tokens")
    upload_time = time()
    s3.upload(df_split, "tokens", TABLE_NAME, BATCH_NUM)
    upload_time = time() - upload_time
    # sql.complete_processing(engine, TABLE_NAME, "tokens")

    if metadata["embeddings"]:
        # Save data frame to output file in case of crash
        # df_split_raw.write_parquet("{}_split_raw.parquet".format(TABLE_NAME), use_pyarrow=True, compression="zstd")
        
        # Pause to avoid timeout if upload took too long
        if upload_time > 60:
            print("Waiting 5 minutes to avoid AWS timeout starting at {} {}".format(dt.datetime.now().strftime("%H:%M:%S"), tzname[0]))
            sleep(60 * 5) # 5 minutes
            print("5 minutes done. Resuming processing")
            
        # Delete old embeddings if they exist
        print("Deleting old embeddings (if necessary)...")
        s3.delete_embeddings(TABLE_NAME)
        sql.deactivate_processing(engine, TABLE_NAME, "embeddings")
            
        # Run embeddings
        print("Processing embeddings...")
        embed_time = time()
        embed_cols = sql.get_embed_cols(engine, meta, TABLE_NAME)
        if BATCH_NUM is None:
            compute_embeddings(df_split_raw.to_pandas(), embed_cols, TABLE_NAME, NUM_THREADS)
        else:
            update_embeddings(df_split_raw.to_pandas(), embed_cols, TABLE_NAME)
        embed_time = time() - embed_time
        sql.complete_processing(engine, TABLE_NAME, "embeddings")
    final_time = time() - start_time
    print("Total time: {}".format(humanize.precisedelta(dt.timedelta(seconds = final_time))))
    
    # Finish reprocessing
    sql.complete_reprocessing(engine, TABLE_NAME)

    # Update batches in sql
    if BATCH_NUM is None:
        sql.complete_batch(engine, TABLE_NAME, 1)
        
    else:
        sql.complete_batch(engine, TABLE_NAME, BATCH_NUM)
    
        # Start next batch if there is one
        if BATCH_NUM < metadata["num_batches"]:
            # Initialize the AWS Batch client
            batch_client = boto3.client('batch')
            
            # Setup input parameters
            name = f"table_name-{ BATCH_NUM }"
            params = {
                "table_name": TABLE_NAME,
                "num_threads": NUM_THREADS,
                "batch_num": BATCH_NUM
            }

            # Submit the job
            response = batch_client.submit_job(
                jobName=name,
                jobQueue=os.getenv('BATCH_QUEUE_LARGE'),
                jobDefinition=os.getenv('BATCH_DEF_LARGE'),
                parameters=params
            )
            
            print("Batch job submitted:")
            print(response)
    
if __name__ == "__main__":
    main()