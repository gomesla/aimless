## CRISP DM: Modeling
- We will use the accuracy of the test score to evaluate the models against each other and select best model during Grid Search.
- However if the bank had an opinon for example one of the ones below we would adjust the scoring type.
  - Recall: If we wanted to make sure we didn't miss any customers that may sign up we would optimize for recall.
  - Precision: If we wanted to make sure we only spend time contacting customers that are likely sign up we would optimize for precision.
### Baseline Model
#### Confusion Matrix
<a href="./analysis_results/module_17_01.step07.baseline_model.confusion_matrix.png" target="_blank"><img src="./analysis_results/module_17_01.step07.baseline_model.confusion_matrix.png"/></a>

#### Performance Metrics (Tables)
<a href="./analysis_results/module_17_01.step07.baseline_model.model_comparison_report.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step07.baseline_model.model_comparison_report.dataFrame.png"/></a>

#### Analysis
The Baseline (DummyClassifier) has an accuracy test score of 88.61476767500653%
