import numpy as np

np.random.seed(42)
import pandas as pd
import re
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras import Sequential
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
import pickle

import warnings

warnings.filterwarnings('ignore')

import os

os.environ['OMP_NUM_THREADS'] = '4'

max_features = 30000
maxlen = 100
embed_size = 300
gru_size = 80
batch_size = 32
epochs = 25
dense_out_shape = 5
pretraining_dense_out_shape = 2

embedding_file = 'crawl-300d-2M.vec'

categories = ['ASPERSION', 'VULGARITY', 'LYING', 'NAMECALLING', 'PEJORATIVE']

aux_cats = ['ThumbsUp', 'ThumbsDown']


def clean_text(t):
    t = str(t)
    text = t.replace("\n", " ")

    return text.strip(' ')


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict()


def get_model(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(gru_size, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(pretraining_dense_out_shape, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def pretraining_with_kaggle(embedding_file, kaggle_data_loc, X_train):
    pretrain = pd.read_csv(kaggle_data_loc)
    X_pretrain = pretrain["comment_text"].fillna("fillna").values
    y_pretrain = pretrain[["obscene", "insult"]].values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_pretrain))
    X_pretrain = tokenizer.texts_to_sequences(X_pretrain)
    x_pretrain = sequence.pad_sequences(X_pretrain, maxlen=maxlen)

    fpick = open("../models/tokenizer.obj", "wb")
    pickle.dump(tokenizer, fpick)
    # print(X_pretrain)

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    # print(nb_words)
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    print(embedding_matrix.shape)
    for word, i in word_index.items():
        # print(str(i), word)
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    print("Embeddings loaded")

    print("Started pretraining with kaggle")
    model = get_model(embedding_matrix)

    filepath = "../models/weights.pretrained_pooledgru_fasttext_" + str(maxlen) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger("../csv/pretrained_pooledgru_fasttext_" + str(maxlen) + ".csv", separator=',', append=True)
    callbacks_list = [checkpoint, csv_logger]

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=3, validation_split=0.3, callbacks=callbacks_list,
                     verbose=1)


def retraining_with_daily_star(pretrained_model_loc, x_train, y_train):
    print("started retraining with daily star data")

    model = load_model(pretrained_model_loc)

    print(model.summary())

    model.layers.pop()

    print(model.summary())

    x = model.layers[-1].output
    x = Dense(dense_out_shape, activation='sigmoid')(x)
    model = Model(inputs=model.input, outputs=x)

    print(model.summary())

    # model.add(Dense(dense_out_shape, activation='sigmoid'))
    # model = Dense(dense_out_shape, activation="sigmoid")(model)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    filepath = "../models/weights.pooledgru_fasttext_" + str(maxlen) + "_with_pretraining.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger("../csv/pooledgru_fasttext_" + str(maxlen) + "_with_pretraining.csv", separator=',',
                           append=True)
    callbacks_list = [checkpoint, csv_logger, EarlyStopping(monitor='val_loss', patience=10)]

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                     callbacks=callbacks_list, verbose=1)


def model_test(model_loc, x_test, x_aux=None):
    model = load_model(model_loc)
    if x_aux == None:
        y_pred = model.predict(x_test, batch_size=1024)
    else:
        y_pred = model.predict([x_test, x_aux], batch_size=1024)
    print(y_pred.shape)

    return y_pred


def calc_performance(y_pred, y_test, threshold, X_test, x_test_id):
    print(threshold)
    print("---------------------")
    for j in range(y_pred.shape[1]):
        if categories[j] != "NAMECALLING" and categories[j] != "VULGARITY": continue
        ftp = open("error_analysis/" + categories[j] + "_with_pretraining_TP.txt", "w")
        ftn = open("error_analysis/" + categories[j] + "_with_pretraining_TN.txt", "w")
        ffp = open("error_analysis/" + categories[j] + "_with_pretraining_FP.txt", "w")
        ffn = open("error_analysis/" + categories[j] + "_with_pretraining_FN.txt", "w")
        TP = TN = FP = FN = 0
        for i in range(y_pred.shape[0]):
            pred = y_pred[i][j]
            curr_pred = 0
            if pred >= threshold:
                curr_pred = pred
                pred = 1
            else:
                pred = 0
            if pred == 1 and y_test[i][j] == 1:
                ftp.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ftp.write(X_test[i] + "\n--------------------------\n--------------------------\n")
                TP += 1
            if pred == 0 and y_test[i][j] == 0:
                ftn.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ftn.write(X_test[i] + "\n--------------------------\n--------------------------\n")
                TN += 1
            if pred == 1 and y_test[i][j] == 0:
                ffp.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ffp.write(X_test[i] + "\n--------------------------\n--------------------------\n")
                FP += 1
            if pred == 0 and y_test[i][j] == 1:
                ffn.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ffn.write(X_test[i] + "\n--------------------------\n--------------------------\n")
                FN += 1
        print(categories[j])
        print("TP: " + str(TP) + ", TN: " + str(TN) + ", FP: " + str(FP) + ", FN: " + str(FN))
        print("accuracy: " + str(((TP + TN) * 1.0) / (TP + TN + FP + FN)))
        try:
            prec = (TP * 1.0) / (TP + FP)
            rec = (TP * 1.0) / (TP + FN)
        except:
            prec = 0.0
            rec = 0.0
        print("Precision: " + str(prec))
        print("Recall: " + str(rec))
        print("F1: " + str((2 * prec * rec) / (prec + rec + 1)))
        print("-----------------------------")
        ffp.close()
        ffn.close()


def get_data(train_data_loc, test_data_loc):
    df = pd.read_csv(train_data_loc)
    df = df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1)

    df['text'] = df['text'].map(lambda com: clean_text(com))

    df2 = pd.read_csv(test_data_loc)
    df2 = df2.drop(df2.columns[df2.columns.str.contains('unnamed', case=False)], axis=1)

    df2['text'] = df2['text'].map(lambda com: clean_text(com))

    return df, df2


def main():
    task = sys.argv[1]
    kaggle_data_loc = "../kaggle_data/train.csv"
    train_data_loc = "../data/train_data_with_tag_and_aux.csv"
    test_data_loc = "../data/val_data_with_tag_and_aux.csv"

    train, test = get_data(train_data_loc, test_data_loc)

    # print(train.shape)
    # print(test.shape)

    X_train = train.text
    X_train_aux = train[aux_cats].values
    y_train = train[categories].values
    X_test = test.text
    X_test_aux = test[aux_cats].values
    y_test = test[categories].values
    x_test_id = test.id

    # print(X_train_aux.shape, X_test_aux.shape)

    # class_weights = {0: 23.95, 1: 53.24, 2: 39.11, 3: 4.45, 4: 182.53, 5: 14.57, 6: 22.16, 7: 32.48, 8: 4.07}

    X_test_text = X_test

    ##########Pretraining####################################################################################################

    if task == "pretrain": pretraining_with_kaggle(embedding_file, kaggle_data_loc, X_train)

    ##########Retraining#####################################################################################################
    if task == "train":
        fpick = open("../models/tokenizer.obj", "rb")
        tokenizer = pickle.load(fpick)

        X_train = tokenizer.texts_to_sequences(X_train)
        x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

        retraining_with_daily_star("../models/weights.pretrained_pooledgru_fasttext_" + str(maxlen) + ".hdf5", x_train, y_train)

    ##########Testing#####################################################################################################
    if task == "test":
        fpick = open("../models/tokenizer.obj", "rb")
        tokenizer = pickle.load(fpick)

        X_test = tokenizer.texts_to_sequences(X_test)
        x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

        y_pred = model_test("../models/weights.pooledgru_fasttext_" + str(maxlen) + "_with_pretraining.hdf5", x_test)

        # y_pred = model_test("models/weights.pooledgru_fasttext_aux_"+ str(maxlen) + "_with_pretraining.hdf5", x_test, X_test_aux)

        thresholds = [0.5]

        for threshold in thresholds: calc_performance(y_pred, y_test, threshold, X_test_text.values, x_test_id.values)


if __name__ == "__main__":
    main()
