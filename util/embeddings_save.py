from pandas import DataFrame
from numpy import mean
import pickle
from gensim.models import Word2Vec
from pandas import DataFrame, merge
from time import time
from tqdm import tqdm
# Database interaction
import util.data_queries as data
from util.s3 import upload_file, BASE_PATH

def prepare_text(df: DataFrame) -> list[list[str]]:
    return df.groupby("id")["word"].apply(list).tolist()

def train_word2vec(texts):
    model = Word2Vec(
        vector_size=100, window=5, 
        min_count=1, workers=4
    )
    print("Building vocabulary...")
    model.build_vocab(tqdm(texts))
    print("Training model...")
    model.train(tqdm(texts), total_examples = model.corpus_count, epochs = model.epochs)
    model.init_sims(replace=True)
    return model

def model_similar_words(df: DataFrame, table_name: str, token: str | None = None):
    cleaned_texts = prepare_text(df)
    model = train_word2vec(cleaned_texts)
    
    folder = "embeddings"
    name = "model_{}.pkl".format(table_name)
    pkl_model_file_name = "{}/{}/{}".format(BASE_PATH, folder, name)
    # save models_per_year
    with open(pkl_model_file_name, 'wb') as f:
        pickle.dump(model, f)
    upload_file(folder, name, token)

def model_similar_words_over_group(df: DataFrame, group_col: str, table_name: str, token: str | None = None):
    time_values = sorted(df[group_col].unique())
    models_per_year = {}
    times = []

    for i, time_value in enumerate(time_values):
        try:
            print("Group {}/{}: {}".format(i + 1, len(time_values), time_value))
            if len(times) == 0:
                remaining_time = "unknown"
            else:
                remaining_time = "{} minutes".format((mean(times) / 60) * (len(time_values) - i))
            print("Estimated time remaining: {}".format(remaining_time))
            start_time = time()
            cleaned_texts = prepare_text(df[df[group_col] == time_value])
            model = train_word2vec(cleaned_texts)
            models_per_year[time_value] = model
            times.append(time() - start_time)
        except Exception:
            models_per_year[time_value] = []
            continue
        
    folder = "embeddings"
    name = "model_{}_{}.pkl".format(table_name, group_col)
    pkl_model_file_name = "{}/{}/{}".format(BASE_PATH, folder, name)
    # save models_per_year
    with open(pkl_model_file_name, 'wb') as f:
        # save models as dictionary, where key is the group_col unique value AND value is the model
        pickle.dump(models_per_year, f) 
    upload_file(folder, name, token)

# NOTE: we HAVE TO ask for the users' preferrence on stopwords at the very begining when they upload the file
# (for data cleaning purpose)
def compute_embeddings(df: DataFrame, metadata: dict, table_name: str, token: str | None = None):
    start = time()
    # Get grouping column if defined
    column = metadata.get("embed_col", None)

    if column is not None:
        # select top words over GROUP and save
        df_text = data.get_columns(table_name, [column], token)
        df_merged = merge(df, df_text, left_on = "id", right_index = True)
        model_similar_words_over_group(df_merged, column, table_name, token)
    else:
        model_similar_words(df, table_name, token)
    print("Embeddings: {} minutes".format((time() - start) / 60))
        