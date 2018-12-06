# incivlity-in-the-wild

This project targets to identify incivilities in comments posted by users in social media. We trained our models on the AZ Daily Star comments collected by the department of Communications, University of Arizona. The models used in this project have their base in the Kaggle task to detect toxicity in Wikipedia comments (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/). 

The most common incivilities occuring in the annotated data are namecalling and vulagirity- we are currently focusing on detecting these.

We also use our trained models to detect uncivil comments from a set of previously unannotated tweets (https://github.com/fivethirtyeight/russian-troll-tweets).

# Codes

data_extractor.py: Extracts data from the pdfs containing contents of the comments collected from AZ Daily Star. Also extracts metadata and annotations from the metadata excel file.

baseline_learning.py: Creates a simple SVM model as a baseline from the Daily Star data.

bidir-gru-with-pooling.py: Creates RNNs with bidirectional GRUs from the Daily Star data. Uses Fasttext embeddings.

bidir-gru-with-pooling-and-pretraining.py: Creates RNNs with bidirectional GRUs from the Kaggle data, and retrains the model on the Daiyl Star data.

flair/basic_flair_trainer.py: Creates a text classification model using character embeddings in Flair.

text_on_twitter.py: Runs the models on unannotated Twitter data.
