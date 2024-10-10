from time import time, sleep, tzname
start_time = time()
from dotenv import load_dotenv
load_dotenv()
import datetime as dt
from functools import partial
import humanize
import json
import multiprocessing as mp
from nltk.corpus import stopwords
import os
import polars as pl
import re
import sys
# from tqdm import tqdm
from util.email import send_email
# Database interaction
from util.s3 import upload
# SQL helpers
from util.sql_connect import sql_connect
import util.data_queries as data
import util.sql_queries as sql
# Word processing
from util.spacy_models import load_spacy_model
import util.word_processing as wp
from util.embeddings_save import compute_embeddings

# Get table name from command line argument
TABLE_NAME = sys.argv[1]

# Get number of threads to use from second command line argument with default of 1 if not provided or not int
try:
    NUM_THREADS = int(sys.argv[2])
except:
    NUM_THREADS = 1

# Get distributed token if defined
try:
    TOKEN = sys.argv[3]
except:
    TOKEN = None

engine, meta = sql_connect()

# Get metadata to determine preprocessing type
metadata = sql.get_metadata(engine, meta, TABLE_NAME)

# Prep stop words and spacy model
language = metadata["language"].lower()
nlp = load_spacy_model(language)
if language in stopwords.fileids():
    stop_words = set(stopwords.words(language))
    if metadata["preprocessing_type"] == "stem":
        stop_words = list(map(lambda x: wp.stem(x, language), stop_words))
else:
    stop_words = []

# Extract lemmas, pos, and dependencies from tokens
def process_sentence(row, mode = "lemma"):
    text = str(row["text"])
    if mode == "lemma":
        doc = nlp(text)
        df = pl.DataFrame([{
                "record_id": row["record_id"], "col": row["col"], "word": re.sub("[^A-Za-z0-9 ]+", "", token.lemma_.lower()), 
                "pos": token.pos_.lower(), "tag": token.tag_.lower(), 
                "dep": token.dep_.lower(), "head": token.head.lemma_.lower()
            } for token in doc if not token.is_stop
        ])
        
        # Return empty data frame if empty
        if len(df) == 0:
            return pl.DataFrame(schema = [
                "record_id", "col", "word",
                "pos", "tag", "dep", "head"
            ])

        # Remove 1 character words
        df = df.filter(pl.col("word").str.strip_chars().str.len_chars() > 1)
    else:
        if mode == "stem":
            words = wp.stem(text, metadata["language"])
        else:
            words = wp.tokenize(text, metadata["language"])
        
        # Remove special characters
        words = [re.sub("[^A-Za-z0-9 ]+", "", word) for word in words]
        # Make lowercase, remove stop words and missing data
        words = [word.lower() for word in words if word.lower() not in stop_words and len(word.strip()) > 1]
        
        # Make data frame
        df = pl.DataFrame({
            "record_id": [row["record_id"]] * len(words),
            "col": [row["col"]] * len(words),
            "word": words
        })
        
    return df

def process_chunk(df: pl.DataFrame, mode = "lemma", i = 0) -> pl.DataFrame:
    df_list: list[pl.DataFrame] = []
    
    start_time = time()
    prev_time = start_time
    for j, row in enumerate(df.iter_rows(named = True)):
        df2 = process_sentence(row, mode)
        if len(df2) > 0:
            df_list.append(df2)
            
        curr_time = time()
        if curr_time - prev_time > NUM_THREADS:
            prev_time = curr_time
            total_time = curr_time - start_time
            its_sec = (j + 1) / total_time
            time_left = humanize.precisedelta(dt.timedelta(seconds = (len(df) - j + 1) / its_sec))
            print("Thread #{}: {}/{} = {}%. {} it/sec. Estimated {} remaining".format("{}".format(i+1).zfill(2), j + 1, len(df), round(100 * (j + 1) / len(df), 2), round(its_sec, 2), time_left))
        if j + 1 == len(df):
            print("Thread #{}: Done. Total time = {}".format("{}".format(i+1).zfill(2), humanize.precisedelta(dt.timedelta(seconds = time() - start_time))))
    return pl.concat(df_list)

def process_thread(arg: tuple[int, pl.DataFrame], mode = "lemma") -> pl.DataFrame:
    i, df = arg
    return process_chunk(df, mode, i)

# Split the text of the given data frame
def split_text(df: pl.DataFrame):
    start = time()
  
    # Multithread processing
    chunks = list(enumerate(df.iter_slices(len(df) // NUM_THREADS + 1)))
    parallel_function = partial(process_thread, mode=metadata["preprocessing_type"])
    with mp.get_context("spawn").Pool(NUM_THREADS) as pool:
        results = pool.map(parallel_function, chunks, 1)
        pool.terminate()
    split_data_list = [result for result in results]
            
    print("Text processing complete. Total time = {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
    print("Merging chunks...")
    split_data = pl.concat(split_data_list)
    # Delete list to free memory
    print("Cleaning memory...")
    del split_data_list
    del chunks
    
    # Create copy to use for embeddings
    print("Creating embeddings copy...")
    df_split_raw = split_data.clone()
    print("Finalizing tokens...")
    # Finish processing
    if metadata["preprocessing_type"] == "lemma":
        # Remove words with unwanted pos
        split_data = split_data.filter(~pl.col("pos").is_in(["num", "part", "punct", "sym", "x", "space"]))
        # Get counts of each word in each record
        split_data = split_data.group_by(["record_id", "word", "col", "pos", "tag", "dep", "head"]).agg(count = pl.len())
    else:
        # Get counts of each word in each record
        split_data = split_data.group_by(["record_id", "word", "col"]).agg(count = pl.len())
    
    print("Data processing: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
    return split_data, df_split_raw
       
def main():  
    print("Loading data...") 
    load_time = time()
    df = data.get_text(engine, TABLE_NAME, TOKEN).drop_nulls().collect()
    print("Load time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - load_time))))
    print("Processing tokens...")   
    df_split, df_split_raw = split_text(df)
    print("Tokens processed: {}".format(len(df_split)))
    print("Uploading tokens...")
    upload_time = time()
    upload(df_split, "tokens", TABLE_NAME, TOKEN)
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
            
        # Run embeddings
        print("Processing embeddings...")
        embed_time = time()
        compute_embeddings(df_split_raw.to_pandas(), metadata, TABLE_NAME, NUM_THREADS, TOKEN)
        embed_time = time() - embed_time
        sql.complete_processing(engine, TABLE_NAME, "embeddings")
    final_time = time() - start_time
    print("Total time: {}".format(humanize.precisedelta(dt.timedelta(seconds = final_time))))

    # Get user data for email
    # print("Sending confirmation email...")
    # user = sql.get_user(engine, meta, metadata["email"])
    # params = {
    #     "title": metadata["title"],
    #     "time": humanize.precisedelta(dt.timedelta(seconds = final_time))
    # }

    # send_email("processing_complete", params, "Processing Complete", user["email"])
    # print("Email sent to", user["email"])
    
if __name__ == "__main__":
    main()