import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns

def clean_text(t):
    text = str(t)
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

df = pd.read_csv('data/complete_data_with_tag.csv')
df = df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1)
#print df.head() 

print df.shape

df_toxic = df.drop(['text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
print df_stats

print('comments that are not labelled:')
print(len(df[(df['aspersion']==0) & (df['hyperbole']==0) & (df['lying']==0) & (df['namecalling']== 0) & (df['noncoop']==0) & (df['offtopic']==0) & (df['others']==0) & (df['pejorative']==0) & (df['sarcasm']==0)]))

df['text'] = df['text'].map(lambda com : clean_text(com))
#print df['text'][0]


categories = ['aspersion', 'hyperbole', 'lying', 'namecalling', 'noncoop', 'offtopic', 'others', 'pejorative', 'sarcasm']
train, test = train_test_split(df, random_state=42, test_size=0.25, shuffle=True)
X_train = train.text
X_test = test.text
#print(X_train.shape)
#print(X_test.shape)

# Define a pipeline combining a text feature extractor with multi lable classifier
#LogReg_pipeline = Pipeline([
 #               ('tfidf', TfidfVectorizer(stop_words=stop_words)),
  #              ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
   #         ])
SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
for category in categories:
    print(category)
    # train the model using X_dtm & y
    #LogReg_pipeline.fit(X_train, train[category])
    SVC_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    #prediction = LogReg_pipeline.predict(X_test)
    prediction = SVC_pipeline.predict(X_test)
    cat_count = 0
    pos_count = 0
    for t in test[category]:
	if t == 1:
	    cat_count += 1
    #for p in prediction:
	#if p == 1: pos_count += 1 
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    #print cat_count, X_test.shape[0]
    print("Majority class baseline is " + str(1.0 - (cat_count/(X_test.shape[0]*1.0))))
    TP = FP = TN = FN = 0
    for i,t in enumerate(test[category]):
	if t == 1 and prediction[i] == 1: TP+=1
	elif t == 1 and prediction[i] == 0: FN+=1
	elif t == 0 and prediction[i] == 1: FP+=1
	elif t == 0 and prediction[i] == 0: TN+=1
    print("TP: " + str(TP) + ", TN: " + str(TN) + ", FP: " + str(FP) + ", FN: " + str(FN))
    print("-----------------------------")

























