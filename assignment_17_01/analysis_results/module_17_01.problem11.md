### Grid Search - Only Bank Client Data
Experiment using bank client data features including duration
#### Confusion Matrix
<a href="./analysis_results/module_17_01.step11.improving_model.experiment1.confusion_matrix.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment1.confusion_matrix.png"/></a>

#### Performance Metrics (Tables)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment1.model_comparison_report.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment1.model_comparison_report.dataFrame.png"/></a>

#### Performance Metrics (Visualized)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment1.model_comparison_graphs.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment1.model_comparison_graphs.png"/></a>

#### Analysis
- The best performing model is Decision Tree with a score of 90.21009502222998% which is better than our baseline score of 88.61476767500653%.
- The worst performing model is KNN with a score of 88.90244965565338% which is better than our baseline score of 88.61476767500653%.
- The fastest performing model is KNN with a score of 88.90244965565338% which is better than our baseline score of 88.61476767500653%.
- The slowest performing model is SVC with a score of 89.90497777002876% which is better than our baseline score of 88.61476767500653%.

### Grid Search - Bank Client Data + Other
Experiment using bank client data features, other features including duration
#### Confusion Matrix
<a href="./analysis_results/module_17_01.step11.improving_model.experiment2.confusion_matrix.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment2.confusion_matrix.png"/></a>

#### Performance Metrics (Tables)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment2.model_comparison_report.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment2.model_comparison_report.dataFrame.png"/></a>

#### Performance Metrics (Visualized)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment2.model_comparison_graphs.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment2.model_comparison_graphs.png"/></a>

#### Analysis
- The best performing model is Logistic Regression with a score of 90.27111847267021% which is better than our baseline score of 88.61476767500653%.
- The worst performing model is KNN with a score of 88.87629674832186% which is better than our baseline score of 88.61476767500653%.
- The fastest performing model is KNN with a score of 88.87629674832186% which is better than our baseline score of 88.61476767500653%.
- The slowest performing model is SVC with a score of 90.1229186644582% which is better than our baseline score of 88.61476767500653%.

### Grid Search - Bank Client Data + Other + Social and Economic
Experiment using bank client data features, other features, social and economic features including duration
#### Confusion Matrix
<a href="./analysis_results/module_17_01.step11.improving_model.experiment3.confusion_matrix.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment3.confusion_matrix.png"/></a>

#### Performance Metrics (Tables)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment3.model_comparison_report.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment3.model_comparison_report.dataFrame.png"/></a>

#### Performance Metrics (Visualized)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment3.model_comparison_graphs.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment3.model_comparison_graphs.png"/></a>

#### Analysis
- The best performing model is Decision Tree with a score of 91.39569348792608% which is better than our baseline score of 88.61476767500653%.
- The worst performing model is KNN with a score of 89.02449655653388% which is better than our baseline score of 88.61476767500653%.
- The fastest performing model is KNN with a score of 89.02449655653388% which is better than our baseline score of 88.61476767500653%.
- The slowest performing model is SVC with a score of 90.5413651817627% which is better than our baseline score of 88.61476767500653%.

### Grid Search - Bank Client Data + Other + Social and Economic without Duraton
Experiment using all features except duration.
As part of the data description above we are told:
```
duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model
```

#### Confusion Matrix
<a href="./analysis_results/module_17_01.step11.improving_model.experiment4.confusion_matrix.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment4.confusion_matrix.png"/></a>

#### Performance Metrics (Tables)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment4.model_comparison_report.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment4.model_comparison_report.dataFrame.png"/></a>

#### Performance Metrics (Visualized)
<a href="./analysis_results/module_17_01.step11.improving_model.experiment4.model_comparison_graphs.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment4.model_comparison_graphs.png"/></a>

#### Analysis
- The best performing model is Decision Tree with a score of 89.30346090140354% which is better than our baseline score of 88.61476767500653%.
- The worst performing model is KNN with a score of 88.68450876122395% which is better than our baseline score of 88.61476767500653%.
- The fastest performing model is KNN with a score of 88.68450876122395% which is better than our baseline score of 88.61476767500653%.
- The slowest performing model is SVC with a score of 89.25115508674047% which is better than our baseline score of 88.61476767500653%.

