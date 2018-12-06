import numpy as np
np.random.seed(42)
import pandas as pd
import re
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

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

aux = ""

max_features = 30000
maxlen = 500
embed_size = 300
gru_size = 80
batch_size = 32
epochs = 500

class_weights = {0:1, 1:7}

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def get_metadata(file): #deprecated
    xl = pd.ExcelFile(file)
    df1 = xl.parse('complete_metadata')
    columns = df1.columns.values
    rowcount = df1.shape[0]
    section = dict()
    thumbsup = dict()
    thumbsdown = dict()
    name = dict()
    for i in range(rowcount):
        id = str(int(df.ix[i,"id"]))
        section[id] = str(int(df.ix[i,"Sections"]))
        thumbsup[id] = str(int(df.ix[i,"ThumbsUp"]))
        thumbsdown[id] = str(int(df.ix[i,"ThumbsDown"]))
        name[id] = str(int(df.ix[i,"Name"]))

    return section, thumbsup, thumbsdown, name
 

def get_model(embedding_matrix, categories):
    print("using text-only model")
    inp = Input(shape=(maxlen, ))
    x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(gru_size, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(len(categories), activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def get_model_with_aux_input(embedding_matrix, categories):
    print("using aux model")
    inp = Input(shape=(maxlen, ))
    print(inp.shape)
    x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix])(inp)
    print(x.shape)
    x = SpatialDropout1D(0.2)(x)
    print(x.shape)
    x = Bidirectional(GRU(gru_size, return_sequences=True))(x)
    print(x.shape)
    auxiliary_input = Input(shape=(2,), name='aux_input')
    aux = BatchNormalization()(auxiliary_input)
    print(auxiliary_input.shape)
    #x = concatenate([x, auxiliary_input])
    #print(x.shape)
    avg_pool = GlobalAveragePooling1D()(x)
    print(avg_pool.shape)
    max_pool = GlobalMaxPooling1D()(x)
    print(max_pool.shape)
    conc = concatenate([avg_pool, max_pool, aux])
    print(conc.shape)
    outp = Dense(len(categories), activation="sigmoid")(conc)
    print(outp.shape)
    
    model = Model(inputs=[inp, auxiliary_input], outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def clean_text(t):
    t = str(t)
    text = t.replace("\n", " ")

    return text.strip(' ')

def train_model(task_id, categories, embedding_matrix, x_train, y_train, X_train_aux = None):
    print("Training started")
    filepath="../models/weights.pooledgru_fasttext_" + task_id + "_" + aux + str(maxlen) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger("../csv/pooledgru_fasttext_aux_"+ aux + str(maxlen) + ".csv", separator=',', append=True)
    callbacks_list = [checkpoint, csv_logger, EarlyStopping(monitor='val_loss', patience = 5)]

    if X_train_aux == None:
        model = get_model(embedding_matrix, categories)
        hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, callbacks=callbacks_list, verbose=1, class_weight=class_weights)
    else:
        model = get_model_with_aux_input(embedding_matrix, categories)
        hist = model.fit([x_train, X_train_aux], y_train, batch_size=batch_size, epochs=epochs, validation_split = 0.1, callbacks=callbacks_list, verbose=1, class_weight=class_weights)


def test_model_performance(task_id, x_test, X_test_aux = None):
    model = load_model("../models/weights.pooledgru_fasttext_" + task_id + "_" + aux + str(maxlen) + ".hdf5")

    if X_test_aux == None: y_pred = model.predict(x_test, batch_size=1024)
    else: y_pred = model.predict([x_test, X_test_aux], batch_size=1024)

    return y_pred


def calc_performance(task_id, categories, y_pred, y_test, threshold, X_test, x_test_id):
    print(threshold)
    print("---------------------")
    for j in range(y_pred.shape[1]):
        if categories[j] != "NAMECALLING" and categories[j]!="VULGARITY": continue
        ftp = open("../error_analysis/" + categories[j] + "_TP_" + task_id + "_" + str(maxlen) + ".txt", "w")
        ftn = open("../error_analysis/" + categories[j] + "_TN_" + task_id + "_" + str(maxlen) + ".txt", "w")
        ffp = open("../error_analysis/" + categories[j] + "_FP_" + task_id + "_" + str(maxlen) + ".txt", "w")
        ffn = open("../error_analysis/" + categories[j] + "_FN_" + task_id + "_" + str(maxlen) + ".txt", "w")
        TP = TN = FP = FN = 0
        for i in range(y_pred.shape[0]):
            pred = y_pred[i][j]
            curr_pred = 0
            if pred >= threshold:
                curr_pred = pred
                pred = 1
            else: pred = 0
            if pred == 1 and y_test[i][j] == 1:
                ftp.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ftp.write(X_test[i]+"\n--------------------------\n--------------------------\n")
                TP+=1
            if pred == 0 and y_test[i][j] == 0:
                ftn.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ftn.write(X_test[i]+"\n--------------------------\n--------------------------\n")
                TN+=1
            if pred == 1 and y_test[i][j] == 0:
                ffp.write("pred_val: " + str(curr_pred) + " id: " + x_test_id[i] + "\n--------------------------\n")
                ffp.write(X_test[i]+"\n--------------------------\n--------------------------\n")
                FP+=1
            if pred == 0 and y_test[i][j] == 1:
                ffn.write("pred_val: " + str(curr_pred)  + " id: " + x_test_id[i] + "\n--------------------------\n")
                ffn.write(X_test[i]+"\n--------------------------\n--------------------------\n")
                FN+=1
        print(categories[j])
        print("TP: " + str(TP) + ", TN: " + str(TN) + ", FP: " + str(FP) + ", FN: " + str(FN))
        print("accuracy: " + str(((TP+TN)*1.0)/(TP+TN+FP+FN)))
        try:
            prec = (TP*1.0)/(TP+FP)
            rec = (TP*1.0)/(TP+FN)
        except:
            prec = 0.0
            rec = 0.0
        print("Precision: " + str(prec))
        print("Recall: " + str(rec))
        try:print("F1: " + str((2*prec*rec)/(prec+rec)))
        except: print("F1: 0")
        print("-----------------------------")
        ffp.close()
        ffn.close()

def get_data(train_data_loc, test_data_loc):
    df = pd.read_csv(train_data_loc)
    df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)

    df['text'] = df['text'].map(lambda com : clean_text(com))

    df2 = pd.read_csv(test_data_loc)
    df2 = df2.drop(df2.columns[df2.columns.str.contains('unnamed',case = False)],axis = 1)

    df2['text'] = df2['text'].map(lambda com : clean_text(com))

    return df, df2

def get_embedding_matrix(embedding_file, word_index):
    print("Creating embedding matrix")
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

    nb_words = min(max_features, len(word_index))
    #print(nb_words)
    embedding_matrix = np.zeros((nb_words+1, embed_size))
    for word, i in word_index.items():
        #print(str(i), word)
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix

def main():
    task = sys.argv[1]
    task_id = sys.argv[2]
    if task_id != "all":
        categories = [task_id.strip().upper()]
    else: categories = ['ASPERSION', 'VULGARITY', 'LYING', 'NAMECALLING', 'PEJORATIVE']

    embedding_file = '../crawl-300d-2M.vec'
    #train_data_loc = sys.argv[2]
    train_data_loc = '../data/train_data_with_tag_and_aux.csv'
    #test_data_loc = sys.argv[3]
    test_data_loc = '../data/val_data_with_tag_and_aux.csv'

    train, test = get_data(train_data_loc, test_data_loc)

    #print(train.shape)
    #print(test.shape)

    #aux_cats = ['Name', 'Section', 'ThumbsUp', 'ThumbsDown']

    aux_cats = ['ThumbsUp', 'ThumbsDown']

    X_train = train.text
    X_train_aux = train[aux_cats].values
    y_train = train[categories].values
    #y_train = train[['namecalling']].values
    X_test = test.text
    X_test_aux = test[aux_cats].values
    y_test = test[categories].values
    #y_test = test[['namecalling']].values
    x_test_id = test.id

    #print(X_train_aux.shape, X_test_aux.shape)

    #class_weights = {0: 23.95, 1: 53.24, 2: 39.11, 3: 4.45, 4: 182.53, 5: 14.57, 6: 22.16, 7: 32.48, 8: 4.07}


    #lens = train.text.str.len()
    #print lens.min(), lens.mean(), lens.std(), lens.max()

    #categories = ['namecalling']

    X_test_text = X_test

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    word_index = tokenizer.word_index

    ###########training################
    if task == "train":
        embedding_matrix = get_embedding_matrix(embedding_file, word_index)
        if aux == "aux_": train_model(task_id, categories, embedding_matrix, x_train, y_train, X_train_aux = X_train_aux)
        else: train_model(task_id, categories, embedding_matrix, x_train, y_train)

    ##########testing#################
    if task == "test":
        if aux == "aux_": y_pred = test_model_performance(task_id, x_test, X_test_aux = X_test_aux)
        else: y_pred = test_model_performance(task_id, x_test)

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7]

        for threshold in thresholds: calc_performance(task_id, categories, y_pred, y_test, threshold, X_test_text.values, x_test_id.values)


if __name__ == "__main__":
    main()



















