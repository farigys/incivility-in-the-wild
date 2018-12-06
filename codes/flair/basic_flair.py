from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer
from flair.data import Sentence
import pandas as pd
import re
import math	

def clean_text(t):
    t = str(t)
    text = t.replace("\n", " ")
    return text


def collect_and_convert_data():
    categories = ['ASPERSION', 'VULGARITY', 'LYING', 'NAMECALLING', 'PEJORATIVE']

    aux_cats = ['ThumbsUp', 'ThumbsDown']

    df = pd.read_csv('../data/train_data_with_tag_and_aux.csv')
    df = df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1)

    df['text'] = df['text'].map(lambda com: clean_text(com))

    train = df

    df2 = pd.read_csv('../data/val_data_with_tag_and_aux.csv')
    df2 = df2.drop(df2.columns[df2.columns.str.contains('unnamed', case=False)], axis=1)

    df2['text'] = df2['text'].map(lambda com: clean_text(com))

    test = df2

    X_train = train.text
    X_train_aux = train[aux_cats].values
    y_train = train[categories].values
    # y_train = train[['namecalling']].values
    X_test = test.text
    X_test_aux = test[aux_cats].values
    y_test = test[categories].values
    # y_test = test[['namecalling']].values
    x_test_id = test.id

    train_size = math.floor(X_train.shape[0] * 0.9)
    val_size = X_train.shape[0] - train_size
    test_size = X_test.shape[0]

    fw = open("resources/tasks/ag_news/train.txt", "w")

    for i in range(train_size):
        flag = 0
        for j in range(y_train.shape[1]):
            if y_train[i][j] == 1:
                flag = 1
                fw.write("__label__" + categories[j] + " ")
        if flag == 0:
            fw.write("__label__NONE ")
        fw.write(X_train[i] + "\n")

    fw = open("resources/tasks/ag_news/dev.txt", "w")

    for i in range(train_size,train_size+val_size):
        flag = 0
        for j in range(y_train.shape[1]):
            if y_train[i][j] == 1:
                flag = 1
                fw.write("__label__" + categories[j] + " ")
        if flag == 0:
            fw.write("__label__NONE ")
        fw.write(X_train[i] + "\n")

    fw = open("resources/tasks/ag_news/test.txt", "w")

    for i in range(test_size):
        flag = 0
        for j in range(y_test.shape[1]):
            if y_test[i][j] == 1:
                flag = 1
                fw.write("__label__" + categories[j] + " ")
        if flag == 0:
            fw.write("__label__NONE ")
        fw.write(X_test[i] + "\n")

def text_classification():
    corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.AG_NEWS)
    corpus.train = [sentence for sentence in corpus.train if len(sentence) > 0]
    corpus.test = [sentence for sentence in corpus.test if len(sentence) > 0]
    corpus.dev = [sentence for sentence in corpus.dev if len(sentence) > 0]
    print("corpus created")
    #print(corpus.get_all_sentences())
    label_dict = corpus.make_label_dictionary()
    print("created label dict")
    for sent in corpus.get_all_sentences():
        print(sent)
        print(sent.labels)
    word_embeddings = [WordEmbeddings('glove'), CharLMEmbeddings('news-forward'), CharLMEmbeddings('news-backward')]
    print("loaded word embeddings")
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,hidden_states=512,reproject_words=True,reproject_words_dimension=256, )

    print("loaded document embeddings")

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=True)

    print("created classifier")

    # 6. initialize the text classifier trainer
    #trainer = TextClassifierTrainer(classifier, corpus, label_dict)

    print("starting training")
    # 7. start the trainig
    #trainer.train('results', learning_rate=0.1, mini_batch_size=32, anneal_factor=0.5, patience=5, max_epochs=50)


    model = classifier.load_from_file("final-model.pt")
    print(model)
    sentences = model.predict(Sentence('France is the current world cup winner.'))

    print("training finished")


def main():
    #collect_and_convert_data()
    text_classification()


if __name__ == "__main__":
    main()
