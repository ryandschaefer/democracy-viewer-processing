from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from util.spacy_models import load_spacy_model

# Tokenizer
def tokenize(text: str, language: str = "English") -> list[str]:
    return word_tokenize(text, language.lower())

# Stemmer
def stem(text: str, language: str = "English") -> list[str]:
    if type(text) == str:
        stemmer = SnowballStemmer(language = language.lower())
        return list(map(lambda x: stemmer.stem(x), text.split()))
    else:
        return []

# Lemmatizer
def lemmatize(text: str, language: str = "English") -> list[str]:
    nlp = load_spacy_model(language)
    nlp.max_length = 2000000
    return [ token.lemma_ for token in nlp(text) ]