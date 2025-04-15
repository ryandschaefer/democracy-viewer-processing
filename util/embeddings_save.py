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
from util.s3 import upload_file, BASE_PATH, download_file
import pickle as pkl

def prepare_text(df: pd.DataFrame) -> list[list[str]]:
    return df.groupby("record_id")["word"].apply(list).tolist()

# move top similar words with keywords requested here
def load_data_from_pkl(table_name: str, pkl_name: str, token: str | None = None) -> Word2Vec:
    local_file = "{}/embeddings/{}_{}.pkl".format(BASE_PATH, table_name, pkl_name.replace("/", "_"))
    
    try:
        download_file(local_file, "embeddings/{}".format(table_name), "{}.pkl".format(pkl_name), token)
    except Exception as err:
        print(err)
        if "No such file or directory" in str(err) or "Not Found" in str(err):
            raise Exception("Failed to load embedding")
        else:
            raise err

    with open(local_file, 'rb') as f:
        return pkl.load(f)

def train_word2vec(texts: list[str], num_threads: int = 4):
    model = Word2Vec(
        vector_size=100, window=5, 
        min_count=1, workers=num_threads
    )
    print("Building vocabulary...")
    model.build_vocab(tqdm(texts))
    print("Training model...")
    model.train(tqdm(texts), total_examples = len(texts), epochs = model.epochs)
    return model

def update_word2vec(model: Word2Vec, texts: list[str]):
    print("Building vocabulary...")
    model.build_vocab(tqdm(texts), update = True)
    print("Training model...")
    model.train(tqdm(texts), total_examples = len(texts), epochs = model.epochs)
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
            except Exception as err:
                print(err)
            
        print()
        col_times.append(time() - col_start_time)
        
def update_similar_words(df: pd.DataFrame, table_name: str, token: str | None = None):
    local_folder = "embeddings"
    s3_folder = "embeddings/{}".format(table_name)
    name = "model.pkl".format(table_name)
    pkl_model_file_name = "{}/{}/{}".format(BASE_PATH, local_folder, name)
    
    cleaned_texts = prepare_text(df)
    model = load_data_from_pkl(table_name, "model", token)
    model = update_word2vec(model, cleaned_texts)
    
    # save models_per_year
    with open(pkl_model_file_name, 'wb') as f:
        pickle.dump(model, f)
    upload_file(local_folder, s3_folder, name, token)

def update_similar_words_over_group(df: pd.DataFrame, embed_cols: list[str], table_name: str, token: str | None = None):
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
                
                name = "model_{}_{}.pkl".format(group_col, time_value)
                pkl_model_file_name = "{}/{}/{}".format(BASE_PATH, "embeddings", name)
                
                cleaned_texts = prepare_text(df[df[group_col] == time_value])
                try:
                    model = load_data_from_pkl(table_name, name, token)
                    model = update_word2vec(model, cleaned_texts)
                except Exception as err:
                    if str(err) == "Failed to load embedding":
                        print("Model does not exist. Training new model...")
                        model = train_word2vec(cleaned_texts)
                    else:
                        raise err
                    
                print("Exporting to output file...") 
                # save models_per_year
                with open(pkl_model_file_name, 'wb') as f:
                    # save models as dictionary, where key is the group_col unique value AND value is the model
                    pickle.dump(model, f) 
                print("Uploading to S3...")
                upload_file("embeddings", "embeddings/{}".format(table_name), name, token)
                
                times.append(time() - start_time)
            except Exception as err:
                print(err)
                continue
            
        print()
        col_times.append(time() - col_start_time)

def compute_embeddings(df: pd.DataFrame, embed_cols: list[str], table_name: str, num_threads: int, batch: int | None = None, token: str | None = None):
    start = time()

    if embed_cols is not None and len(embed_cols) > 0:
        # select top words over GROUP and save
        df_text = data.get_columns(table_name, embed_cols, batch, token).collect().to_pandas()
        df_merged = pd.merge(df, df_text, on = "record_id")
        model_similar_words_over_group(df_merged, embed_cols, table_name, num_threads, token)
    else:
        model_similar_words(df, table_name, num_threads, token)
        
    print("Embeddings: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
    
def update_embeddings(df: pd.DataFrame, embed_cols: list[str], table_name: str, batch: int | None, token: str | None = None):
    start = time()

    if embed_cols is not None and len(embed_cols) > 0:
        # select top words over GROUP and save
        df_text = data.get_columns(table_name, embed_cols, batch, token).collect().to_pandas()
        df_merged = pd.merge(df, df_text, on = "record_id")
        update_similar_words_over_group(df_merged, embed_cols, table_name, token)
    else:
        update_similar_words(df, table_name, token)
        
    print("Embeddings: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
        
def start_embeddings(df: pd.DataFrame, embed_cols: list[str], table_name: str, num_threads: int, batch: int | None, token: str | None = None):
    if batch is None or batch == 1:
        compute_embeddings(df, embed_cols, table_name, num_threads, batch, token)
    else:
        update_embeddings(df, embed_cols, table_name, batch, token)