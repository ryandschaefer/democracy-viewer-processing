import datetime as dt
from functools import partial
import humanize
import multiprocessing as mp
from nltk.corpus import stopwords
import polars as pl
import re
from time import time
from util.spacy_models import load_spacy_model
import util.s3 as s3
import util.word_processing as wp

class DataProcessing:
    def __init__(self, metadata: dict, num_threads: int, table_name: str):
        self.metadata = metadata
        self.num_threads = num_threads
        self.table_name = table_name
        self.load_stop_words()
        
    # Load stop words list
    def load_stop_words(self):
        start = time()
        print("Loading stop words...")
        
        # Load stopwords file if it exists
        try:
            s3.download_file(f"files/{ self.table_name }.txt", "stopwords", f"{ self.table_name }.txt")
            custom_stopwords = True
        except:
            custom_stopwords = False

        # Prep stop words and spacy model
        language = self.metadata["language"].lower()
        nlp = load_spacy_model(language)
        nlp.max_length = 8000000
        if custom_stopwords:
            with open(f"files/{ self.table_name }.txt") as f:
                stop_words = f.readlines()
        else:
            if language in stopwords.fileids():
                stop_words = list(set(stopwords.words(language)))
            else:
                stop_words = []
                
        if self.metadata["preprocessing_type"] == "stem":
            stop_words2: list[str] = []
            for word in stop_words:
                for token in wp.stem(word, language):
                    tmp = re.sub("[^A-Za-z0-9 ]+", "", token).strip()
                    if len(tmp) > 1:
                        stop_words2.append(tmp)
            stop_words = stop_words2
        elif self.metadata["preprocessing_type"] == "lemma":
            stop_words2: list[str] = []
            for word in stop_words:
                doc = nlp(word)
                for token in doc:
                    tmp = re.sub("[^A-Za-z0-9 ]+", "", token.lemma_).strip()
                    if len(tmp) > 1:
                        stop_words2.append(tmp)
            stop_words = stop_words2
        else:
            stop_words2: list[str] = []
            for word in stop_words:
                for token in wp.tokenize(word, language):
                    tmp = re.sub("[^A-Za-z0-9 ]+", "", token).strip()
                    if len(tmp) > 1:
                        stop_words2.append(tmp)
            stop_words = stop_words2
            
        self.stop_words = stop_words
        self.nlp = nlp
        print("Load stop words: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))

    # Extract lemmas, pos, and dependencies from tokens
    def process_sentence(self, row, mode = "lemma"):
        text = str(row["text"])
        if mode == "lemma":
            doc = self.nlp(text)
            df = pl.DataFrame([{
                    "record_id": row["record_id"], "col": row["col"], "word": re.sub("[^A-Za-z0-9 ]+", "", token.lemma_.lower()), 
                    "pos": token.pos_.lower(), "tag": token.tag_.lower(), 
                    "dep": token.dep_.lower(), "head": token.head.lemma_.lower()
                } for token in doc if token.lemma_ not in self.stop_words
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
                words = wp.stem(text, self.metadata["language"])
            else:
                words = wp.tokenize(text, self.metadata["language"])
            
            # Remove special characters
            words = [re.sub("[^A-Za-z0-9 ]+", "", word).strip() for word in words]
            # Make lowercase, remove stop words and missing data
            words = [word.lower() for word in words if word.lower() not in self.stop_words and len(word) > 1]
            
            # Make data frame
            df = pl.DataFrame({
                "record_id": [row["record_id"]] * len(words),
                "col": [row["col"]] * len(words),
                "word": words
            })
            
        return df

    def process_chunk(self, df: pl.DataFrame, mode = "lemma", i = 0) -> pl.DataFrame:
        df_list: list[pl.DataFrame] = []
        
        start_time = time()
        prev_time = start_time
        for j, row in enumerate(df.iter_rows(named = True)):
            df2 = self.process_sentence(row, mode)
            if len(df2) > 0:
                df_list.append(df2)
                
            curr_time = time()
            if curr_time - prev_time > self.num_threads:
                prev_time = curr_time
                total_time = curr_time - start_time
                its_sec = (j + 1) / total_time
                time_left = humanize.precisedelta(dt.timedelta(seconds = (len(df) - j + 1) / its_sec))
                print("Thread #{}: {}/{} = {}%. {} it/sec. Estimated {} remaining".format("{}".format(i+1).zfill(2), j + 1, len(df), round(100 * (j + 1) / len(df), 2), round(its_sec, 2), time_left))
            if j + 1 == len(df):
                print("Thread #{}: Done. Total time = {}".format("{}".format(i+1).zfill(2), humanize.precisedelta(dt.timedelta(seconds = time() - start_time))))
        return pl.concat(df_list)

    def process_thread(self, arg: tuple[int, pl.DataFrame], mode = "lemma") -> pl.DataFrame:
        i, df = arg
        return self.process_chunk(df, mode, i)

    # Split the text of the given data frame
    def split_text(self, df: pl.DataFrame):
        start = time()

        # Multithread processing
        chunks = list(enumerate(df.iter_slices(len(df) // self.num_threads + 1)))
        parallel_function = partial(self.process_thread, mode=self.metadata["preprocessing_type"])
        with mp.get_context("spawn").Pool(self.num_threads) as pool:
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
        if self.metadata["preprocessing_type"] == "lemma":
            # Remove words with unwanted pos
            split_data = split_data.filter(~pl.col("pos").is_in(["num", "part", "punct", "sym", "x", "space"]))
            # Get counts of each word in each record
            split_data = split_data.group_by(["record_id", "word", "col", "pos", "tag", "dep", "head"]).agg(count = pl.len())
        else:
            # Get counts of each word in each record
            split_data = split_data.group_by(["record_id", "word", "col"]).agg(count = pl.len())
        
        print("Data processing: {}".format(humanize.precisedelta(dt.timedelta(seconds = time() - start))))
        return split_data, df_split_raw