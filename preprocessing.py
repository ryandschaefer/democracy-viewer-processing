from time import time, sleep, tzname
start_time = time()
from dotenv import load_dotenv
load_dotenv()
import datetime as dt
import humanize
import json
from multiprocessing import Pool
from nltk.corpus import stopwords
import polars as pl
import re
import sys
# from tqdm import tqdm
# Word processing
from util.spacy_models import load_spacy_model
import util.word_processing as wp
from util.embeddings_save import compute_embeddings

# Get parameter file name from command line argument
PARAMS_FILE = sys.argv[1]

# Get number of threads to use from second command line argument with default of 1 if not provided or not int
try:
    NUM_THREADS = int(sys.argv[2])
except:
    NUM_THREADS = 1
    
# Read metadata file
with open(PARAMS_FILE) as f:
    params = json.load(f)
    
# Get data file from parameters
DATA_FILE = params["data_file"]
    
# Make dataset name
email_formatted = re.sub(r'\W+', '_', params['email'])
TABLE_NAME = f"{ email_formatted }_{ int(time.time() * 1000) }"

# Prep stop words and spacy model
language = params["language"].lower()
nlp = load_spacy_model(language)
if language in stopwords.fileids():
    stop_words = set(stopwords.words(language))
    if params["preprocessing_type"] == "stem":
        stop_words = list(map(lambda x: wp.stem(x, language), stop_words))
else:
    stop_words = []

# Extract lemmas, pos, and dependencies from tokens
def process_sentence(row, mode = "lemma"):
    text = row["text"]
    if mode == "lemma":
        doc = nlp(text)
        df = pl.DataFrame([{
                "record_id": row["record_id"], "col": row["col"], "word": token.lemma_.lower(), 
                "pos": token.pos_.lower(), "tag": token.tag_.lower(), 
                "dep": token.dep_.lower(), "head": token.head.lemma_.lower()
            } for token in doc if not token.is_stop
        ])
    else:
        if mode == "stem":
            words = wp.stem(text, params["language"])
        else:
            words = wp.tokenize(text, params["language"])
        
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
    start_time = time()
    prev_time = start_time
    df_list: list[pl.DataFrame] = []
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

# Split the text of the given data frame
def split_text(df: pl.DataFrame):
    start = time()
  
    # Multithread processing
    chunks = df.iter_slices(len(df) // NUM_THREADS + 1)
    pool = Pool(processes=NUM_THREADS)
    jobs = [pool.apply_async(process_chunk, args=(chunk, params["preprocessing_type"], i)) for i, chunk in enumerate(chunks)]
    pool.close()
    pool.join()
    split_data_list = [job.get() for job in jobs]
            
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
    if params["preprocessing_type"] == "lemma":
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
    df_raw = pl.read_csv(DATA_FILE).with_row_index("record_id")
    df_list: list[pl.DataFrame] = []
    for col in params["text"]:
        df_list.append(
            df_raw
                .select([col, "record_id"])
                .rename({ f"{col}": "text" })
                .with_columns(col=pl.lit(col))
                .cast({ "text": pl.Utf8 })
        )
    df = pl.concat(df_list).drop_nulls()
    
    print("Load time: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - load_time))))
    print("Processing tokens...")   
    df_split, df_split_raw = split_text(df)
    print("Tokens processed: {}".format(len(df_split)))
    df_split.write_parquet(f"files/output/{ TABLE_NAME }_split.parquet", use_pyarrow=True, compression="zstd")
    # sql.complete_processing(engine, TABLE_NAME, "tokens")

    if params["embeddings"]:
        # Save data frame to output file in case of crash
        df_split_raw.write_parquet("{}_split_raw.parquet".format(TABLE_NAME), use_pyarrow=True, compression="zstd")
        print("Processing embeddings...")
        embed_time = time()
        compute_embeddings(df_split_raw.to_pandas(), params, NUM_THREADS)
        embed_time = time() - embed_time
        # sql.complete_processing(engine, TABLE_NAME, "embeddings")
    final_time = time() - start_time
    print("Total time: {}".format(humanize.precisedelta(dt.timedelta(seconds = final_time))))
    
if __name__ == "__main__":
    main()