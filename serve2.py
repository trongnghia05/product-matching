from gensim.models.doc2vec import Doc2Vec
from keras.models import load_model
import pandas as pd
import re
import numpy as np
from ast import literal_eval
from flask import Flask
import json
from flask import request
from elasticsearch import Elasticsearch

from  utils import *

app = Flask(__name__)


model_gensim = Doc2Vec.load("model/gensim_model_2")
# df = pd.read_csv('data/df_final_2.csv', sep='|',converters={'doc_vector': literal_eval})
# df.doc_vector = df.doc_vector.map(lambda x: convert_array(x))
model_classification = load_model('model/model_final_2.h5')
es = Elasticsearch("http://localhost:9202")

@app.route("/matching/predict", methods=['GET'])
def home():
    title_exam = request.args.get("name")
    print("title_exam: ", title_exam)

    res = es.search(index="word_embed", sort = ["_score"], size=50, query={
        "match": {
            "full_name": {
                "query": title_exam
            }
        }
    })

    product_key = []
    full_name = []
    rules = []
    doc_vector = []

    for product in res["hits"]["hits"]:
        product_key.append(product["_source"]["product_key"])
        full_name.append(product["_source"]["full_name"])
        rules.append(product["_source"]["rules"])
        doc_vector.append(list(product["_source"]["doc_vector"]))


    df = pd.DataFrame({"product_key": product_key,
                       "rules": rules,
                       "full_name": full_name,
                       "doc_vector": doc_vector})

    df_predict = predict(title_exam, model_gensim, model_classification, df)
    df_return = df_predict[['product_key', 'rules', 'full_name', 'predict']].head(10)
    df_return.reset_index(drop=True, inplace=True)
    data = []
    print(df_return.head())
    for i in range(df_return.shape[0]):
        item = {}
        item['product_key'] = str(df_return.loc[i, "product_key"])
        item['rules'] = df_return.loc[i, "rules"]
        item['full_name'] = df_return.loc[i, "full_name"]
        item['predict'] = str(df_return.loc[i, "predict"])
        data.append(item)
    data = json.dumps(data, ensure_ascii=False)
    return data
app.run(host='localhost', port ='5112')
