from time import time
start_time = time()
from dotenv import load_dotenv
load_dotenv()
from multiprocessing import Pool, cpu_count, freeze_support, RLock
from nltk.corpus import stopwords
import polars as pl
import re
import sys
from tqdm import tqdm
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
    NUM_THREADS = cpu_count()

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
    if mode == "lemma":
        doc = nlp(row["text"])
        df = pl.DataFrame([{
                "record_id": row["record_id"], "col": row["col"], "word": token.lemma_.lower(), 
                "pos": token.pos_.lower(), "tag": token.tag_.lower(), 
                "dep": token.dep_.lower(), "head": token.head.lemma_.lower()
            } for token in doc if not token.is_stop
        ])
    else:
        if mode == "stem":
            words = wp.stem(row["text"], metadata["language"])
        else:
            words = wp.tokenize(row["text"], metadata["language"])
        
        # Remove special characters
        words = [re.sub("[^A-Za-z0-9 ]+", "", word) for word in words]
        # Make lowercase, remove stop words and missing data
        words = [word.lower() for word in words if word.lower() not in stop_words and word.strip()]
        
        # Make data frame
        df = pl.DataFrame({
            "record_id": [row["record_id"]] * len(words),
            "col": [row["col"]] * len(words),
            "word": words
        })
        
    return df
    
def process_chunk(df: pl.DataFrame, mode = "lemma", i = 0) -> pl.DataFrame:
    tqdm_text = "#" + "{}".format(i + 1).zfill(2)
    return pl.concat([process_sentence(row, mode) for row in tqdm(df.iter_rows(named = True), total = len(df), desc=tqdm_text, position=i+1)])

# Split the text of the given data frame
def split_text(df: pl.DataFrame):
    start = time()
    
    # Multithreaded processing
    freeze_support()
    chunks = df.iter_slices(len(df) // NUM_THREADS)
    pool = Pool(processes=NUM_THREADS, initargs=(RLock(),), initializer=tqdm.set_lock)
    jobs = [pool.apply_async(process_chunk, args=(chunk, metadata["preprocessing_type"], i)) for i, chunk in enumerate(chunks)]
    pool.close()
    split_data_list = [job.get() for job in jobs]
    split_data = pl.concat(split_data_list)
    # Print blank lines to adjust for progress bars
    print("\n" * (len(jobs) + 1))
    # Create copy to use for embeddings
    df_split_raw = split_data.clone()
    # Finish processing
    if metadata["preprocessing_type"] == "lemma":
        # Remove words with unwanted pos
        split_data = split_data.filter(~pl.col("pos").is_in(["num", "part", "punct", "sym", "x", "space"]))
        # Get counts of each word in each record
        split_data = split_data.group_by(["record_id", "word", "col", "pos", "tag", "dep", "head"]).agg(pl.len())
    else:
        # Get counts of each word in each record
        split_data = split_data.group_by(["record_id", "word", "col"]).agg(pl.len())
    
    print("Data processing: {} minutes".format((time() - start) / 60))
    return split_data, df_split_raw

# Upload data to s3
def upload_result(df: pl.DataFrame):
    start_time = time()
    upload(df, "tokens", TABLE_NAME, TOKEN)
    print("Upload time: {} seconds".format(time() - start_time))
       
def main():  
    print("Loading data...") 
    load_time = time()
    df = data.get_text(engine, TABLE_NAME, TOKEN).collect()
    print("Load time: {} minutes".format((time() - load_time) / 60))
    print("Processing tokens...")   
    df_split, df_split_raw = split_text(df)
    print("Tokens processed: {}".format(len(df)))
    print("Uploading tokens...")
    upload_result(df_split)
    sql.complete_processing(engine, TABLE_NAME, "tokens")

    if metadata["embeddings"]:
        print("Processing embeddings...")
        compute_embeddings(df_split_raw.to_pandas(), metadata, TABLE_NAME, NUM_THREADS, TOKEN)
        sql.complete_processing(engine, TABLE_NAME, "embeddings")
    final_time = (time() - start_time) / 60
    print("Total time: {} minutes".format(final_time))

    # Get user data for email
    print("Sending confirmation email...")
    user = sql.get_user(engine, meta, metadata["email"])
    params = {
        "title": metadata["title"],
        "time": round(final_time, 3)
    }

    send_email("processing_complete", params, "Processing Complete", user["email"])
    print("Email sent to", user["email"])
    
if __name__ == "__main__":
    main()