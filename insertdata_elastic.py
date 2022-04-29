
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import numpy as np
from ast import literal_eval

def convert_array(x):
    return np.array(x)


es = Elasticsearch("http://localhost:9202")
df = pd.read_csv('data/df_final_2.csv', sep = '|', converters = {'doc_vertor': literal_eval})
#df.doc_vector = df.doc_vector.map(lambda x: np.array(x))

docs = []

for i in range(df.shape[0]):
    doc = {}
    doc["_index"] = "word_embed"
    doc["_id"] = i
    doc["product_key"] = df.loc[i, "product_key"]
    doc["full_name"] = df.loc[i, "full_name"]
    doc["rules"] = df.loc[i, "rules"]
    doc["doc_vector"] = np.fromstring(df.loc[i, "doc_vector"][1:-1], dtype = float, sep = ',')
    docs.append(doc)
print(type(doc["doc_vector"]))
helpers.bulk(es, docs, chunk_size = len(docs), request_timeout = 200)
