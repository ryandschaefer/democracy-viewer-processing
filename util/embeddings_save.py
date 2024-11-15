import datetime as dt
import numpy as np
import pickle
from gensim.models import Word2Vec
import humanize
import pandas as pd
from time import time
from tqdm import tqdm
# Database interaction
import util.data_queries as data
from util.s3 import upload_file, BASE_PATH

def prepare_text(df: pd.DataFrame) -> list[list[str]]:
    return df.groupby("record_id")["word"].apply(list).tolist()

def train_word2vec(texts: list[str], num_threads: int = 4):
    model = Word2Vec(
        vector_size=100, window=5, 
        min_count=1, workers=num_threads
    )
    print("Building vocabulary...")
    model.build_vocab(tqdm(texts))
    print("Training model...")
    model.train(tqdm(texts), total_examples = model.corpus_count, epochs = model.epochs)
    model.init_sims(replace=True)
    return model

def model_similar_words(df: pd.DataFrame, table_name: str, num_threads: int, token: str | None = None):
    cleaned_texts = prepare_text(df)
    model = train_word2vec(cleaned_texts, num_threads)
    
    local_folder = "embeddings"
    s3_folder = "embeddings/{}".format(table_name)
    name = "model.pkl".format(table_name)
    pkl_model_file_name = "{}/{}/{}".format(BASE_PATH, local_folder, name)
    # save models_per_year
    with open(pkl_model_file_name, 'wb') as f:
        pickle.dump(model, f)
    upload_file(local_folder, s3_folder, name, token)

def model_similar_words_over_group(df: pd.DataFrame, embed_cols: list[str], table_name: str, num_threads: int, token: str | None = None):
    col_times = []
    for j, group_col in enumerate(embed_cols):
        times = []
        time_values = sorted(df[group_col].astype(str).unique())
        print("Column {}/{}: {}".format(j + 1, len(embed_cols), group_col))
        if len(col_times) == 0:
            remaining_time = "unknown"
        else:
            remaining_time = humanize.precisedelta(dt.timedelta(seconds = (np.mean(col_times)) * (len(embed_cols) - i)))
        print("Estimated time remaining: {}".format(remaining_time))
        
        col_start_time = time()
        for i, time_value in enumerate(time_values):
            try:
                print("Group {}/{}: {}".format(i + 1, len(time_values), time_value))
                if len(times) == 0:
                    remaining_time = "unknown"
                else:
                    remaining_time = humanize.precisedelta(dt.timedelta(seconds = (np.mean(times)) * (len(time_values) - i)))
                print("Estimated time remaining: {}".format(remaining_time))
                
                start_time = time()
                
                cleaned_texts = prepare_text(df[df[group_col] == time_value])
                model = train_word2vec(cleaned_texts, num_threads)
                print("Exporting to output file...") 
                name = "model_{}_{}.pkl".format(group_col, time_value)
                pkl_model_file_name = "{}/{}/{}".format(BASE_PATH, "embeddings", name)
                # save models_per_year
                with open(pkl_model_file_name, 'wb') as f:
                    # save models as dictionary, where key is the group_col unique value AND value is the model
                    pickle.dump(model, f) 
                print("Uploading to S3...")
                upload_file("embeddings", "embeddings/{}".format(table_name), name, token)
                
                times.append(time() - start_time)
            except Exception:
                continue
            
        print()
        col_times.append(time() - col_start_time)

def compute_embeddings(df: pd.DataFrame, embed_cols: list[str], table_name: str, num_threads: int, token: str | None = None):
    start = time()

    if embed_cols is not None and len(embed_cols) > 0:
        # select top words over GROUP and save
        df_text = data.get_columns(table_name, embed_cols, token).collect().to_pandas()
        df_merged = pd.merge(df, df_text, left_on = "record_id", right_index = True)
        model_similar_words_over_group(df_merged, embed_cols, table_name, num_threads, token)
    else:
        model_similar_words(df, table_name, num_threads, token)
        
    print("Embeddings: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
        