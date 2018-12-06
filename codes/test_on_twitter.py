import numpy as np
np.random.seed(42)
import pandas as pd
import re
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

aux = ""

categories = ['ASPERSION', 'VULGARITY', 'LYING', 'NAMECALLING', 'PEJORATIVE']

max_features = 30000
maxlen = 500

def clean_text(t):
    t = str(t)
    text = t.replace("\n", " ")

    return text.strip(' ')

def get_data(train_data_loc, test_data_loc):
	df = pd.read_csv(train_data_loc)
	df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

	df['text'] = df['text'].map(lambda com : clean_text(com))

	df2 = pd.read_csv(test_data_loc)
	df2 = df2.drop(df2.columns[df2.columns.str.contains('unnamed',case = False)],axis = 1)

	df2['text'] = df2['text'].map(lambda com : clean_text(com))

	return df, df2

def load_data(data_loc):
    embedding_file = '../crawl-300d-2M.vec'
    train_data_loc = '../data/train_data_with_tag_and_aux.csv'
    test_data_loc = '../data/val_data_with_tag_and_aux.csv'

    train, test = get_data(train_data_loc, test_data_loc)

    X_train = train.text
    X_test = test.text

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))

    ##################################################################################

    df = pd.read_csv(data_loc)
    df = df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1)

    test = df[df.language=='English']
    test = test[df.post_type != 'RETWEET']
    test = test[df.post_type != 'QUOTE_TWEET']

    #print(test.shape)

    test['content'] = test['content'].map(lambda com: clean_text(com))

    test_data = test.content

    X_test = tokenizer.texts_to_sequences(test_data)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    return x_test, test_data.values

def test_model_performance(task_id, x_test):
    print("Prediction started")
    #model = load_model("models/weights.pooledgru_fasttext_" + aux + str(maxlen) + ".hdf5")
    model = load_model("../models/weights.pooledgru_fasttext_" + task_id + "_" + aux + str(maxlen) + ".hdf5")

    y_pred = model.predict(x_test, batch_size=1024)

    return y_pred


def main():
    #task_id = sys.argv[1]
    cats = ['namecalling', 'vulgarity']
    for cat in cats:
        task_id = cat
        for ix in range(1,6):
            test_data, test_text = load_data('../data/russian-troll-tweets-master/IRAhandle_tweets_' + str(ix) + '.csv')
            y_pred = test_model_performance(task_id, test_data)

            print(y_pred.shape)

            tot_data = dict()
            tot_data["pred"] = []
            tot_data["text"] = []
            '''
            pos_data = dict()
            pos_data["pred"] = []
            pos_data["text"] = []

            neg_data = dict()
            neg_data["pred"] = []
            neg_data["text"] = []
            '''
            for i in range(y_pred.shape[0]):
                pred = y_pred[i][0]
                tot_data["pred"].append(pred)
                tot_data["text"].append(test_text[i])
                '''
                if pred >= 0.5:
                    pos_data["pred"].append(pred)
                    pos_data["text"].append(test_text[i])
                elif pred >=0.3 and pred < 0.4:
                    neg_data["pred"].append(pred)
                    neg_data["text"].append(test_text[i])
                #fw.write("pred_val: " + str(pred) + "\n--------------------\n" + test_text[i] + "\n--------------------\n")
            #fw.close()
            df = pd.DataFrame.from_dict(pos_data)
            df.to_csv('outputs/' + categories[j] + '_positive_troll_data.csv')

            df = pd.DataFrame.from_dict(neg_data)
            df.to_csv('outputs/' + categories[j] + '_confused_negative_troll_data.csv')
            '''
            df = pd.DataFrame.from_dict(tot_data)
            df.to_csv('../outputs/russ_troll_data/' + task_id + '_troll_data_' + str(ix) + '.csv')


if __name__ == "__main__":
    main()
