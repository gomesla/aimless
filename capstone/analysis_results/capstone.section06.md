## Evaluation
We will now re-run our best pipeline discovered druing modeling which had an ROC AUC score of 98.70%:
- Preprocessing=Original
- Vectorizer=TfidfVectorizer
- Model=XGBClassifier ({"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null})

### Results
<table>
<tr>
<th></th>
<th>Train</th>
<th>Test</th>
</tr>
<tr>
<td>Classification Report</td>
<td>

```

              precision    recall  f1-score   support

           0       0.99      0.99      0.99      4987
           1       1.00      0.99      1.00      1232
           2       0.99      0.97      0.98      7640
           3       0.97      0.99      0.98      9532
           4       1.00      1.00      1.00      1483
           5       0.99      0.98      0.98      4942
           6       1.00      1.00      1.00      1725
           7       1.00      0.99      1.00      1944

    accuracy                           0.98     33485
   macro avg       0.99      0.99      0.99     33485
weighted avg       0.98      0.98      0.98     33485

```

</td>
<td>

```

              precision    recall  f1-score   support

           0       0.91      0.89      0.90      2138
           1       0.85      0.74      0.79       528
           2       0.89      0.87      0.88      3275
           3       0.80      0.89      0.84      4085
           4       0.91      0.82      0.86       636
           5       0.84      0.81      0.82      2118
           6       0.96      0.88      0.92       739
           7       0.93      0.86      0.89       833

    accuracy                           0.86     14352
   macro avg       0.89      0.84      0.86     14352
weighted avg       0.86      0.86      0.86     14352

```

</td>
</tr>
<tr>
<td>Confusion Matrix</td>
<td>

<a href="./analysis_results/capstone.evaluation.XGBClassifier.TfidfVectorizer.Original.confusion_matrix.train.png" target="_blank"><img src="./analysis_results/capstone.evaluation.XGBClassifier.TfidfVectorizer.Original.confusion_matrix.train.png"/></a></td>
<td>

<a href="./analysis_results/capstone.evaluation.XGBClassifier.TfidfVectorizer.Original.confusion_matrix.test.png" target="_blank"><img src="./analysis_results/capstone.evaluation.XGBClassifier.TfidfVectorizer.Original.confusion_matrix.test.png"/></a></td>
</tr>
</table>

