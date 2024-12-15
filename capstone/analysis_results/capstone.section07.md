## Deployment and Future Work
### Observations
- We got lucky with this dataset and the data seemed to have some level of preprocessing (unclear based on documentation) done for us but lowercasing and removing stop words is a must.
- Original field performed best for ROC AUC score. Surprisingly stemming and lemmatization did not have much of an impact on performance likely because the data had already been pre-processsed as noted earlier. More analysis may be needed, however the loss in ROC AUC score is not significant enough to be of concern right now. We would still prefer to do lemmatization in general.
- The TfidfVectorizer paired with various models performs best. We should definitely use TfidfVectorizer in our pipeline.
- It is very clear that to get the best performance from the XGBClassifier we have to pay for it inprocessing time.
### Model/Pipeline Selection
- Given the choice if we want to maximize ROC AUC score we would pick our best pipeline which had an ROC AUC score of 98.70%:
  - Preprocessing=Original
  - Vectorizer=TfidfVectorizer
  - Model=XGBClassifier ({"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null})

- In general if we want a model capable of dealing with larger sets I would go with:
  - XGBoost which is very efficent and scores highly vs our best model as well as is easy to train as it can make use of multiple cores. For training time it appears that a majority of the time is actually spent in the vectorization as the count vs TFIDF perform comparably for score but the TFIDF takes longer to train.
  - If we can trade off on ROC AUC score I would go with RandomForestClassifier which had good tradeoff for speed of training.
### Deployment
- The best model from above is deployed as a Streamlit application below:
  - **Streamlit Application**: [Capstone: IT Ticket Classification](https://aimless-capstone.streamlit.app/)

### Next Steps
Deep learning may be able get us even better results. This would be worth researching.

