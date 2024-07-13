# Python dependencies
python3 -m pip install -r requirements.txt

# SpaCy language models
### Chinese
python3 -m spacy download zh_core_web_sm
### English
python3 -m spacy download en_core_web_sm
### French
python3 -m spacy download fr_core_news_sm
### German
python3 -m spacy download de_core_news_sm
### Greek
python3 -m spacy download el_core_news_sm
### Italian
python3 -m spacy download it_core_news_sm
### Portuguese
python3 -m spacy download pt_core_news_sm
### Russian
python3 -m spacy download ru_core_news_sm
### Spanish
python3 -m spacy download es_core_news_sm

# Third party language models
### Latin
python3 -m pip install https://huggingface.co/latincy/la_core_web_sm/resolve/main/la_core_web_sm-any-py3-none-any.whl