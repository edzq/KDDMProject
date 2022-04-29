#NLP tools
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg
from tqdm import tqdm

import string
import pandas as pd
import numpy as np
import os
import json
import csv
from ast import literal_eval

df = pd.read_csv('./df_clustering_data_v2.csv')

#some preprocessing
#replace the nan with empty string
df.fillna('', inplace=True)

#adding word count columns for both abstract and body_text
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(x.strip().split()))  # word count for abstract
df['body_word_count'] = df['body_text'].apply(lambda x: len(x.strip().split()))  # word count for body text
df['body_unique_words']=df['body_text'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body text
# df.head()

#drop nan
df.dropna(inplace=True)
#df.info()


punctuations = string.punctuation
stopwords = list(STOP_WORDS)
custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]

for w in custom_stop_words:
    if w not in stopwords:
        stopwords.append(w)
        
# Parser
parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens



tqdm.pandas()
df["processed_text"] = df["body_text"].progress_apply(spacy_tokenizer)

df.to_csv('./tokenizer_processing_finished.cvs')
