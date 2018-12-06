from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import WordEmbeddings, CharLMEmbeddings, DocumentLSTMEmbeddings
from flair.models.text_classification_model import TextClassifier
from flair.trainers.text_classification_trainer import TextClassifierTrainer
import re
import math	

def clean_text(t):
    t = str(t)
    text = t.replace("\n", " ")
    return text

def text_classification():
    corpus: TaggedCorpus = NLPTaskDataFetcher.fetch_data(NLPTask.AG_NEWS)
    corpus.train = [sentence for sentence in corpus.train if len(sentence) > 0]
    corpus.test = [sentence for sentence in corpus.test if len(sentence) > 0]
    corpus.dev = [sentence for sentence in corpus.dev if len(sentence) > 0]
    print("corpus created")
    #print(corpus.get_all_sentences())
    label_dict = corpus.make_label_dictionary()
    print("created label dict")
    #for sent in corpus.get_all_sentences():
    #    print(sent.labels)
    word_embeddings = [WordEmbeddings('glove'), CharLMEmbeddings('news-forward'), CharLMEmbeddings('news-backward')]
    print("loaded word embeddings")
    document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,hidden_states=512,reproject_words=True,reproject_words_dimension=256, )

    print("loaded document embeddings")

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=True)

    print("created classifier")

    # 6. initialize the text classifier trainer
    trainer = TextClassifierTrainer(classifier, corpus, label_dict)

    print("starting training")

    # 7. start the trainig
    trainer.train('results', learning_rate=0.1, mini_batch_size=32, anneal_factor=0.5, patience=5, max_epochs=50)

    print("training finished")

    # 8. plot training curves (optional)
    #from flair.visual.training_curves import Plotter
    #plotter = Plotter()
    #plotter.plot_training_curves('resources/ag_news/results/loss.tsv')
    #plotter.plot_weights('resources/ag_news/results/weights.txt')



def main():
    #collect_and_convert_data()
    text_classification()

if __name__ == "__main__":
    main()
