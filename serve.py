from gensim.models.doc2vec import Doc2Vec
from keras.models import load_model
import pandas as pd
import numpy as np
from ast import literal_eval
from flask import Flask, request
import json
import utils


global model_gensim
global df
global model_classification


model_classification = None
model_gensim = None
df = None


app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello_world():
    return "Product Matching Service"


@app.route("/matching/predict", methods=['GET'])
def home():
    data = {}
    products = []
    name = request.args.get('name')
    df_predict = utils.predict(name, model_gensim, model_classification, df)
    category_predict = pd.DataFrame(df_predict[df.predict >= 0.5].groupby(['rules'])['product_key'].count())
    if len(category_predict) > 0:
        category_predict = category_predict.index[category_predict['product_key'].values.argmax()]
        data["category"] = category_predict
        df_predict = df_predict.head(10)[["product_key", "full_name", "predict", "rules"]].transpose()

        for product_id in df_predict.columns:
            products.append({"product_key:": df_predict.loc['product_key', product_id],
                             "full_name: ": df_predict.loc['full_name', product_id],
                             "similarity: ": df_predict.loc['predict', product_id],
                             "label: ": df_predict.loc['rules', product_id],
                             })
    data["similar product"] = products
    data = json.dumps(data, ensure_ascii=False)
    return data


if __name__ == "__main__":
    model_gensim = Doc2Vec.load("model/gensim_model_2")
    df = pd.read_csv('data/df_final_2_small.csv', sep='|',converters={'doc_vector': literal_eval})
    print(df.columns)
    df.doc_vector = df.doc_vector.map(lambda x: utils.convert_array(x))
    model_classification = load_model('model/model_final_3.h5')
    app.run(host='localhost', port='5112')
