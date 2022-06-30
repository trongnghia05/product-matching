from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.models import load_model
import pandas as pd
import argparse
import re
import numpy as np
from ast import literal_eval
from flask import Flask
import json
from flask import request
from elasticsearch import Elasticsearch
from utils import *
import os

# IP_ELASTICSEARCH = "localhost"
# PORT_ELASTICSEARCH = 9200
IP_FLASK = "0.0.0.0"
PORT_FLASK = 5112
model_gensim = None
model_classification = None
model_cls = None
tfIdf = None
domain_label = None
es = None


app = Flask(__name__)


@app.route("/cls/multi-predict", methods=['GET'])
def predicts_domain():
    result = {}
    predicts = []
    product_titles = request.args.getlist("name")
    print("product_titles: ", product_titles)
    titles_vector = [text_process_cls_domain(txt) for txt in product_titles]
    titles_vector = [word_separation(txt) for txt in titles_vector]
    titles_vector = tfIdf.transform(titles_vector).toarray()
    y_preds = model_cls.predict(titles_vector)
    y_pro = np.argmax(y_preds, axis=1)
    y_preds = np.argmax(y_preds, axis=1)
    for i in range(y_preds.shape[0]):
        if y_preds[i] != 16 and y_pro[i] >= 0.85:
            label = y_preds[i]
        else:
            label = 16
        s = {"label": domain_label[str(label)]["name"], "id": domain_label[str(label)]["id"], "parent_id": domain_label[str(label)]["parent_id"]}
        predicts.append(s)
    result["result"] = predicts
    data = json.dumps(result, ensure_ascii=False)
    return data

@app.route("/cls/predict", methods=['GET'])
def predict_domain():
    product_title = request.args.get("name")
    result = {}
    title = text_pre_processing(product_title)
    title = word_separation(title)
    title = tfIdf.transform([title]).toarray()
    y_preds = model_cls.predict(title)
    y_pro = np.argmax(y_preds, axis = 1)
    y_preds = np.argmax(y_preds, axis=1)
    if y_preds[0] != 16 and y_pro[0] >= 0.85:
        label = y_preds[0]
    else:
        label = 16
    s = {"label": domain_label[str(label)]["name"], "id": domain_label[str(label)]["id"], "parent_id": domain_label[str(label)]["parent_id"]}
    result["result"] = [s]
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

if __name__ == "__main__":
    env = os.environ
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--ip_elas', type=str, default= "localhost", help='A required str positional argument')
    parser.add_argument('--port_elas', type=int, default=9200, help='An optional integer positional argument')
    parser.add_argument('--ip_flask', type=str, default= "0.0.0.0", help='An optional integer argument')
    parser.add_argument('--port_flask', type=int, default=5112, help='A boolean switch')

    args = parser.parse_args()

    # IP_ELASTICSEARCH = args.ip_elas
    # PORT_ELASTICSEARCH = args.port_elas
    # IP_FLASK = args.ip_flask
    # PORT_FLASK = args.port_flask

    # IP_ELASTICSEARCH = env["IP_ELASTICSEARCH"]
    # PORT_ELASTICSEARCH = env["PORT_ELASTICSEARCH"]


    # print(IP_ELASTICSEARCH)
    # print(PORT_ELASTICSEARCH)

    model_gensim = Doc2Vec.load("model/gensim_model_2")
    model_classification = load_model('model/model_final_2.h5')
    model_cls = load_model('model/cls_model.h5')
    tfIdf = pickle.load(open("model/tfidf.pickle", "rb"))
    domain_label = load_file_label('label_cls/domain_label_kiotpro_new.json')
    # es = Elasticsearch("http://" + IP_ELASTICSEARCH + ":" + str(PORT_ELASTICSEARCH))

    app.run(host="0.0.0.0", port=5112)
