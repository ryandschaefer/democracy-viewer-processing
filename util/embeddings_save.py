import datetime as dt
import numpy as np
import pickle
from gensim.models import Word2Vec
import humanize
import pandas as pd
from time import time
from tqdm import tqdm

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

def model_similar_words(df: pd.DataFrame, num_threads: int):
    cleaned_texts = prepare_text(df)
    model = train_word2vec(cleaned_texts, num_threads)
    
    name = "model.pkl"
    pkl_model_file_name = "files/output/embeddings/{}".format(name)
    # save models_per_year
    with open(pkl_model_file_name, 'wb') as f:
        pickle.dump(model, f)

def model_similar_words_over_group(df: pd.DataFrame, group_col: str, num_threads: int):
    time_values = sorted(df[group_col].unique())
    times = []

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
            pkl_model_file_name = "files/output/embeddings/{}".format(name)
            # save models_per_year
            with open(pkl_model_file_name, 'wb') as f:
                # save models as dictionary, where key is the group_col unique value AND value is the model
                pickle.dump(model, f) 
            print("Uploading to S3...")
            
            times.append(time() - start_time)
        except Exception:
            continue

def compute_embeddings(df: pd.DataFrame, metadata: dict, num_threads: int):
    start = time()
    # Get grouping column if defined
    column = metadata.get("embed_col", None)

    if column is not None:
        # select top words over GROUP and save
        df_text = pd.read_parquet(metadata["data_file"], "pyarrow")[[column]]
        df_merged = pd.merge(df, df_text, left_on = "record_id", right_index = True)
        model_similar_words_over_group(df_merged, column, num_threads)
    else:
        model_similar_words(df, num_threads)
        
    print("Embeddings: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
        