import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('stopwords')

def preProcess(row, stopWords):
    if type(row) is str:
        text = row
    else:
        text = row[DOCUMENT_FIELD]
    textArray = [w for w in word_tokenize(text)]
    noStopWordsTextArray = [w.lower() for w in textArray if not w.lower() in stopWords]
    lemma = WordNetLemmatizer()
    lemmaArray = [lemma.lemmatize(w) for w in noStopWordsTextArray]
    lemmaText =  ' '.join(lemmaArray)
    if type(row) is str:
        return lemmaText
    else:
        row[DOCUMENT_FIELD] = lemmaText
        return row

# https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb
class CapstonePreprocessingTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        self.stopwords2Use = set(stopwords.words('english'))
        pass
    
    def fit(self, X, y, **transform_params):
        return self

    def transform(self, X, **transform_params):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            X_copy = X_copy.apply(lambda row: preProcess(row=row, stopWords=self.stopwords2Use), axis=1)
        else:
            X_copy = X_copy.apply(lambda value: preProcess(row=value, stopWords=self.stopwords2Use))
        return X_copy
