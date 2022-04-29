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
model_cls = load_model('model/model_fianl.h5')
tfIdf = pickle.load(open("model/tfidf.pickle", "rb" ))
domain_label = load_file_label('label_cls/domain_label.json')
es = Elasticsearch("http://localhost:9200")


@app.route("/cls/multi-predict", methods=['GET'])
def predicts_domain():
    product_titles = request.args.get("name")
    result = []
    titles_vector = [text_pre_processing(txt) for txt in product_titles]
    titles_vector = [word_separation(txt) for txt in titles_vector]
    titles_vector = tfIdf.transform(titles_vector).toarray()
    y_preds = model_cls.predict(titles_vector)
    y_preds = np.argmax(y_preds, axis=1)
    for i, y_pred in enumerate(y_preds):
        s = {"product_title": product_titles[i], "label": domain_label[str(y_preds[i])]}
        result.append(s)
    data = json.dumps(result, ensure_ascii=False)
    return data

@app.route("/cls/predict", methods=['GET'])
def predict_domain():
    product_title = request.args.get("name")
    result = []
    title = text_pre_processing(product_title)
    title = word_separation(title)
    title = tfIdf.transform([title]).toarray()
    y_preds = model_cls.predict(title)
    y_preds = np.argmax(y_preds, axis=1)
    s = {"product_title": product_title, "label": domain_label[str(y_preds[0])]}
    result.append(s)
    data = json.dumps(result, ensure_ascii=False)
    return data

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
    result = {}
    similar_products = []
    print(df_return.head())
    for i in range(df_return.shape[0]):
        item = {}
        item['product_key'] = str(df_return.loc[i, "product_key"])
        item['label'] = df_return.loc[i, "rules"]
        item['full_name'] = df_return.loc[i, "full_name"]
        item['similarity'] = str(df_return.loc[i, "predict"])
        similar_products.append(item)
    category_predict = pd.DataFrame(df_return[df_return.predict > 0.5].groupby(['rules'])["product_key"].count())
    if len(category_predict) > 0:
        category_predict = category_predict.index[category_predict["product_key"].values.argmax()]
        result["category"] = category_predict
    else:
        result["category"] = ""
    result["similar product"] = similar_products
    data = json.dumps(result, ensure_ascii=False)
    return data
app.run(host='localhost', port ='5112')
