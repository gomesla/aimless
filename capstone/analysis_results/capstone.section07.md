## Deployment and Future Work
### Observations
- We got lucky with this dataset and the data seemed to have some level of preprocessing (unclear based on documentation) done for us but lowercasing and removing stop words is a must.
- Original field performed best for accuracy. Surprisngly stemming and lemmatization had worse performance. More analysis may be needed, however the loss in accuracy is not significant enough. We would still prefer to do lemmatization which gives next best accuracy.
- The TfidfVectorizer paired with various models performs best. We should definitely use TfidfVectorizer in our pipeline.
- It is very clear that while Logistic Regression performed best the processing time is very high.
### Model/Pipeline Selection
- Given the choice if we want to maximize accuracy we would pick our best pipeline which had an accuracy score of 86.43%:
  - Preprocessing=Original
  - Vectorizer=TfidfVectorizer
  - Model=LogisticRegression ({"model__C": 1, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null})

- In general if we want a model capable of dealing with larger sets where training time is a concern and we can trade off on accuracy I would go with:
  - Complement Naive Bayes as industry expectation is the data will be imbalanced for IT tickets. Even though it was third best the kNN classifier was only marginally better
### Next Steps
Deep learning may be able get us even better results. This would be worth researching.

