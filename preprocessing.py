from time import time, sleep, tzname
start_time = time()
from dotenv import load_dotenv
load_dotenv()
import datetime as dt
import humanize
from nltk.corpus import stopwords
import polars as pl
import re
import sys
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

# Split the text of the given data frame
def split_text(df: pl.DataFrame):
    start = time()
    prev_time = time()
  
    split_data_list: list[pl.DataFrame] = []
    for i, row in enumerate(df.iter_rows(named = True)):
        df_tmp = process_sentence(row, metadata["preprocessing_type"])
        if len(df_tmp) > 0:
            split_data_list.append(df_tmp)
            
        curr_time = time()
        if curr_time - prev_time > 5:
            prev_time = curr_time
            total_time = curr_time - start
            its_sec = (i + 1) / total_time
            time_left = humanize.precisedelta(dt.timedelta(seconds = (len(df) - (i + 1)) / its_sec))
            print("{}/{} = {}%. {} it/sec. Estimated {} remaining".format(i + 1, len(df), round(100 * (i + 1) / len(df), 2), round(its_sec, 2), time_left))
            
    print("Text processing complete. Total time = {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
    print("Merging results...")
    split_data = pl.concat(split_data_list)
    # Delete list to free memory
    print("Cleaning memory...")
    del split_data_list
    
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

# Upload data to s3
def upload_result(df: pl.DataFrame):
    start_time = time()
    upload(df, "tokens", TABLE_NAME, TOKEN)
    print("Upload time: {} seconds".format(time() - start_time))
       
def main():  
    print("Loading data...") 
    load_time = time()
    df = data.get_text(engine, TABLE_NAME, TOKEN).drop_nulls().collect()
    print("Load time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - load_time))))
    print("Processing tokens...")   
    df_split, df_split_raw = split_text(df)
    print("Tokens processed: {}".format(len(df_split)))
    
    upload_time = time()
    upload(df_split, "tokens", TABLE_NAME, TOKEN)
    upload_time = time() - upload_time
    print("Upload time: {}".format(humanize.precisedelta(dt.timedelta(seconds = upload_time))))
    # Pause to avoid timeout if upload took too long
    if upload_time > 60:
        print("Waiting 5 minutes to avoid AWS timeout starting at {} {}".format(dt.datetime.now().strftime("%H:%M:%S"), tzname[0]))
        sleep(60 * 5) # 5 minutes
        print("5 minutes done. Resuming processing")

    if metadata["embeddings"]:
        print("Processing embeddings...")
        embed_time = time()
        compute_embeddings(df_split_raw.to_pandas(), metadata, TABLE_NAME, NUM_THREADS, TOKEN)
        embed_time = time() - embed_time
        sql.complete_processing(engine, TABLE_NAME, "embeddings")
    final_time = time() - start_time
    print("Total time: {}".format(humanize.precisedelta(dt.timedelta(seconds = final_time))))
    
if __name__ == "__main__":
    main()