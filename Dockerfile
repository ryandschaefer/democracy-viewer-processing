FROM python:3.11

# Set the working directory within docker image
WORKDIR /usr/src/app

# Install Python dependencies
ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Install SpaCy language models
### Chinese
RUN python3 -m spacy download zh_core_web_sm
### English
RUN python3 -m spacy download en_core_web_sm
### French
RUN python3 -m spacy download fr_core_news_sm
### German
RUN python3 -m spacy download de_core_news_sm
### Greek
RUN python3 -m spacy download el_core_news_sm
### Italian
RUN python3 -m spacy download it_core_news_sm
### Portuguese
RUN python3 -m spacy download pt_core_news_sm
### Russian
RUN python3 -m spacy download ru_core_news_sm
### Spanish
RUN python3 -m spacy download es_core_news_sm
# Third party language models
### Latin
RUN python3 -m pip install https://huggingface.co/latincy/la_core_web_sm/resolve/main/la_core_web_sm-any-py3-none-any.whl
