from time import time
start_time = time()
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import polars as pl
import sys
from tqdm import tqdm
tqdm.pandas()
from util.email import send_email
# Database interaction
from util.s3 import upload
# SQL helpers
from util.sql_connect import sql_connect
import util.data_queries as data
import util.sql_queries as sql
# Word processing
from util.spacy_models import load_spacy_model
from util.word_processing import stem, tokenize
from util.embeddings_save import compute_embeddings
print("Import time: {} seconds".format(time() - start_time))

# Get table name from command line argument
TABLE_NAME = sys.argv[1]

# Get distributed token if defined
try:
    TOKEN = sys.argv[2]
except:
    TOKEN = None

engine, meta = sql_connect()

# Get metadata to determine preprocessing type
metadata = sql.get_metadata(engine, meta, TABLE_NAME)

# Extract lemmas, pos, and dependencies from tokens
def process_sentence(row, nlp = load_spacy_model()):
    doc = nlp(row["text"])
    return [{
            "id": row["id"], "col": row["col"], "word": token.lemma_.lower(), 
            "pos": token.pos_.lower(), "tag": token.tag_.lower(), 
            "dep": token.dep_.lower(), "head": token.head.lemma_.lower()
        } for token in doc if not token.is_stop
    ]
    
def process_chunk(df: pd.DataFrame, nlp = load_spacy_model()):
    return pd.concat([pd.DataFrame(process_sentence(row, nlp)) for _, row in tqdm(df.iterrows(), total = len(df))], ignore_index=True)

# Split the text of the given data frame
def split_text(df: pd.DataFrame):
    start = time()
    
    # Create a deep copy of data
    split_data = deepcopy(df)
    # Lemmatize, stem, or split text based on preprocessing type
    if metadata["preprocessing_type"] == "lemma":
        # Load spacy language
        nlp = load_spacy_model(metadata["language"])
        
        # Multithreading
        num_threads = 8
        chunks = np.array_split(df, num_threads)
        with ThreadPoolExecutor(max_workers=num_threads) as executor: 
            split_data = list(executor.map(lambda x: process_chunk(x, nlp), chunks))
        split_data = pd.concat(split_data)
        
        # Remove words with unwanted pos
        split_data = split_data[~split_data["pos"].isin(["num", "part", "punct", "sym", "x", "space"])].dropna()
        # Create copy to use for embeddings
        df_split_raw = split_data.copy()
        # Get counts of each word in each record
        split_data = split_data.groupby(["id", "word", "col", "pos", "tag", "dep", "head"]).size().reset_index(name="count")
    else:
        language = metadata["language"].lower()
        if language in stopwords.fileids():
            stop_words = set(stopwords.words(language))
            if metadata["preprocessing_type"] == "stem":
                stop_words = list(map(lambda x: stem(x, language), stop_words))
        else:
            stop_words = []
        
        if metadata["preprocessing_type"] == "stem":
            split_data["text"] = split_data["text"].progress_apply(lambda x: stem(x, metadata["language"]))
        elif metadata["preprocessing_type"] == "none":
            split_data["text"] = split_data["text"].progress_apply(lambda x: tokenize(x, metadata["language"]))
            
        # Create a new row for each word
        split_data = split_data.explode("text")
        split_data["text"] = (
            split_data["text"]
                # Remove special characters
                .str.replace("\W", "", regex=True)
                # Make lowercase
                .str.lower()
            )
        
        # Remove empty values
        split_data = split_data[split_data["text"] != ""]
        # Create copy to use for embeddings
        df_split_raw = split_data.copy()
        # Get counts of each word in each record
        split_data = split_data.groupby(["id", "text", "col"]).size().reset_index(name="count")
        split_data = split_data.rename(columns = {"text": "word"})
        # Remove stop words and missing data
        split_data = split_data[~split_data["word"].isin(stop_words)].dropna()
        
    # Rename columns to match database names
    split_data = split_data.rename(columns = {"id": "record_id"})
    
    print("Data processing: {} minutes".format((time() - start) / 60))
    return split_data, df_split_raw

# Upload data to s3
def upload_result(df: pd.DataFrame):
    start_time = time()
    upload(df, "tokens", TABLE_NAME, TOKEN)
    print("Upload time: {} seconds".format(time() - start_time))
           
print("Processing tokens...")   
df = data.get_text(engine, TABLE_NAME, TOKEN)
df_split, df_split_raw = split_text(df)
print("Tokens processed: {}".format(len(df)))
print("Uploading tokens...")
upload_result(df_split)
sql.complete_processing(engine, TABLE_NAME, "tokens")

if metadata["embeddings"]:
    print("Processing embeddings...")
    compute_embeddings(df_split_raw, metadata, TABLE_NAME, TOKEN)
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