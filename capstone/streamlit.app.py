import pandas as pd
import numpy as np
import streamlit as st
import requests
from io import StringIO

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
#nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('stopwords')


INPUT_FILE = 'https://raw.githubusercontent.com/gomesla/aimless/refs/heads/main/capstone/data/all_tickets_processed_improved_v3.csv'
DOCUMENT_FIELD = 'Original'
TARGET_FIELD = 'target'
TARGET_FIELD_ENCODED = 'target_encoded'

stop_words = set(stopwords.words('english'))
url = INPUT_FILE
print(f'Downloading: {url}')
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f'Unable to download: {url}')
rawDf = pd.read_csv(StringIO(response.text))
rawDf = rawDf.rename(columns={"Document": DOCUMENT_FIELD, "Topic_group": TARGET_FIELD})
rawDf.info()

experimentDf = rawDf.copy()
UNIQUE_TARGET_CLASS_COUNT = len(rawDf[TARGET_FIELD].value_counts().values)

labelEncoder = LabelEncoder()
labelEncoder.fit(experimentDf[TARGET_FIELD])
CLASS_MAPPING =  dict(zip(labelEncoder.transform(labelEncoder.classes_), labelEncoder.classes_))
experimentDf[TARGET_FIELD_ENCODED]= labelEncoder.transform(experimentDf[TARGET_FIELD]) 

def preProcess(row):
    if type(row) is str:
        text = row
    else:
        text = row[DOCUMENT_FIELD]
    textArray = [w for w in word_tokenize(text)]
    noStopWordsTextArray = [w.lower() for w in textArray if not w.lower() in stop_words]
    lemma = WordNetLemmatizer()
    lemmaArray = [lemma.lemmatize(w) for w in noStopWordsTextArray]
    lemmaText =  ' '.join(lemmaArray)
    if type(row) is str:
        return lemmaText
    else:
        row[DOCUMENT_FIELD] = lemmaText
        return row

# https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb
class PreprocessingTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y, **transform_params):
        return self

    def transform(self, X, **transform_params):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            X_copy = X_copy.apply(preProcess, axis=1)
        else:
            X_copy = X_copy.apply(preProcess)
        return X_copy

@st.cache_resource()
def trainModel():
    pipeline = Pipeline([
        ('preprocesssor', PreprocessingTransformer()),
        ('vectorizer', TfidfVectorizer(max_features=None)),
        ('model', XGBClassifier(n_jobs=-1, objective='multi:softmax', num_class=UNIQUE_TARGET_CLASS_COUNT, colsample_bytree= 0.5, max_depth= 10, subsample= 0.7))
    ])
    
    TEST_MODE = False
    X = experimentDf[[DOCUMENT_FIELD]]
    y = experimentDf[TARGET_FIELD_ENCODED]
    if TEST_MODE:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        pipeline.fit(X_train[DOCUMENT_FIELD], y_train)
        y_pred_train_final = pipeline.predict(X_train[DOCUMENT_FIELD])
        y_pred_test_final = pipeline.predict(X_test[DOCUMENT_FIELD])
        trainScore = classification_report(y_train, y_pred_train_final)
        testScore = classification_report(y_test, y_pred_test_final)
        print(f'Train Report:\n{trainScore}')
        print(f'Test Report:\n{testScore}')
    else:
        pipeline.fit(X[DOCUMENT_FIELD], y)

    return pipeline


# Set page title and favicon
st.set_page_config(page_title="Capstone: IT Ticket Classification", page_icon=":rocket:")
st.title("Capstone: IT Ticket Classification")
st.write(
    """
    This is a demo of the best model as determined by the Capstone Project
    """
)

model = trainModel()

# Input text box for the sentence to classify
inputString = st.text_input("Enter the sentence to classify:", key="input_sentence", 
                                     type="default", value="")


# Classification button
if st.button("Classify", key="classify_button"):
    if inputString:
        input = pd.Series()
        input.loc[0] = inputString
        result = model.predict(input)
        category = CLASS_MAPPING[result[0]]
        # Display classification results
        st.subheader("Classification Results:")
        st.markdown("---")
        st.markdown(f"{category}", unsafe_allow_html=True)
        st.markdown("---")