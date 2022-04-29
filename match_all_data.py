import pandas as pd
import numpy as np
import os
import json
import csv
from ast import literal_eval
from tqdm import tqdm

df = pd.read_csv('./kddm_full_list_without.csv')

#match the meatadata of all the paper
METADATA_DIR = 'scripts/metadata/computerscience sub/' # data file path

num_paper_find = 0
#paper_list = []

df['abstract_meta'] = ''

df['doi'] = ''
df['authors'] = ''
df['title'] = ''
df['journal'] = ''
df['url'] = ''

# create the pandas dataframe

pbar = tqdm(total=len(df))
for metadata_file in os.listdir(METADATA_DIR):
    with open(os.path.join(METADATA_DIR, metadata_file)) as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = int(metadata_dict['paper_id'])
            #check whether in the df
            if df[df['paper_id'] == paper_id].empty is not True:
                #num_paper_find += 1
                pbar.update(1)
                idx = df[df['paper_id'] == paper_id].index.values[0]
                #print("Find #:", num_paper_find, ", data index:", idx)
                df.loc[idx,'doi'] = metadata_dict['doi']
                df.loc[idx,'authors'] = str(metadata_dict['authors'])
                df.loc[idx,'abstract_meta'] = metadata_dict['abstract']
                df.loc[idx,'title'] = metadata_dict['title']
                df.loc[idx,'journal'] = metadata_dict['venue']
                df.loc[idx,'url'] = metadata_dict['s2_url']
                #paper_list.append(idx)
                #pbar.update(1)
pbar.close()
df.to_csv('kddm_data_full_v2.csv')