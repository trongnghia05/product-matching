import pandas as pd
import numpy as np
import re

def convert_array(x):
    return np.array(x)

def predict(title, model_gensim, model_classification, df_product_vector):
    title = text_pre_processing(title)
    title = text_split(title)
    title_vector = model_gensim.infer_vector(title)
    data_predict = np.array([np.concatenate((title_vector, df_product_vector.loc[i, 'doc_vector'])) for i in range(df_product_vector.shape[0])])
    data_predict = data_predict.reshape(df_product_vector.shape[0],28,28)
    predict = model_classification.predict(data_predict)
    predict = predict.reshape(predict.shape[0],)
    df_product_vector['predict'] = predict
    df_product_vector = df_product_vector.sort_values(by='predict', ascending=False)
    return df_product_vector

def text_pre_processing(txt):
    # txt = "Apple iPhone 11 Pro Max – 2 Sim ( ZA/A)( Quốc"
    txt = txt.lower()
    txt = re.sub('[\-\'\\\.\,\#\$\%\^\&\*\(\)\!\+\-\_\"\:\?\<\>\{\}\[\];\(\)]','',txt)
    txt = re.sub('(\\r\\n)|[\/]',' ',txt)
    txt = re.sub('[^zxcvbnmasdfghjklqwertyuiop\d\sàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ]','',txt)
    txt = re.sub(' +',' ',txt)
    return txt

def text_split(txt):
    strs = txt.split(" ")
    return strs
