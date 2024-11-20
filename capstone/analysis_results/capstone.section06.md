## Evaluation
We will now run our best pipeline which had an accuracy score of 86.43%:
- Preprocessing=Original
- Vectorizer=TfidfVectorizer
- Model=LogisticRegression ({"model__C": 1, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null})

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

               Access       0.94      0.90      0.92      4987
Administrative rights       0.89      0.76      0.82      1232
           HR Support       0.89      0.89      0.89      7640
             Hardware       0.84      0.92      0.88      9532
     Internal Project       0.92      0.86      0.89      1483
        Miscellaneous       0.88      0.87      0.88      4942
             Purchase       0.97      0.91      0.94      1725
              Storage       0.94      0.89      0.92      1944

             accuracy                           0.89     33485
            macro avg       0.91      0.87      0.89     33485
         weighted avg       0.89      0.89      0.89     33485

```

</td>
<td>

```

                       precision    recall  f1-score   support

               Access       0.92      0.88      0.90      2138
Administrative rights       0.84      0.74      0.79       528
           HR Support       0.87      0.87      0.87      3275
             Hardware       0.81      0.88      0.85      4085
     Internal Project       0.92      0.83      0.87       636
        Miscellaneous       0.85      0.83      0.84      2118
             Purchase       0.97      0.89      0.93       739
              Storage       0.94      0.87      0.91       833

             accuracy                           0.86     14352
            macro avg       0.89      0.85      0.87     14352
         weighted avg       0.87      0.86      0.86     14352

```

</td>
</tr>
<tr>
<td>Confusion Matrix</td>
<td>

<a href="./analysis_results/capstone.evaluation.confusion_matrix.train.png" target="_blank"><img src="./analysis_results/capstone.evaluation.confusion_matrix.train.png"/></a></td>
<td>

<a href="./analysis_results/capstone.evaluation.confusion_matrix.test.png" target="_blank"><img src="./analysis_results/capstone.evaluation.confusion_matrix.test.png"/></a></td>
</tr>
</table>

