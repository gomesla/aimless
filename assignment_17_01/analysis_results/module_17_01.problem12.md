## CRISP DM: Evaluation
Using a "voting" approach where each time a feature is deemed important by a model it increments the vote. Coefficents are including to evaluate direction and magnitude as well.
### Only Bank Client Data
#### Feature Importance

Important features using voting approach:

['duration', 'month', 'contact', 'job', 'age']

The specific values of these important features are:
<a href="./analysis_results/module_17_01.step11.improving_model.experiment1.important_features.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment1.important_features.dataFrame.png"/></a>

#### Decision Tree
The descision tree is more easily interpretable/explainable to help with the understanding how the model works and can help the bank make decisions about optimizing the campaign.
<a href="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png"/></a>
### Bank Client Data + Other
#### Feature Importance

Important features using voting approach:

['duration', 'previous', 'month', 'contact', 'job', 'age', 'campaign']

The specific values of these important features are:
<a href="./analysis_results/module_17_01.step11.improving_model.experiment2.important_features.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment2.important_features.dataFrame.png"/></a>

#### Decision Tree
The descision tree is more easily interpretable/explainable to help with the understanding how the model works and can help the bank make decisions about optimizing the campaign.
<a href="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png"/></a>
### Bank Client Data + Other + Social and Economic
#### Feature Importance

Important features using voting approach:

['duration', 'euribor3m', 'emp.var.rate', 'cons.price.idx', 'month', 'cons.conf.idx', 'contact']

The specific values of these important features are:
<a href="./analysis_results/module_17_01.step11.improving_model.experiment3.important_features.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment3.important_features.dataFrame.png"/></a>

#### Decision Tree
The descision tree is more easily interpretable/explainable to help with the understanding how the model works and can help the bank make decisions about optimizing the campaign.
<a href="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png"/></a>
### Bank Client Data + Other + Social and Economic without Duraton
#### Feature Importance

Important features using voting approach:

['euribor3m', 'month', 'emp.var.rate', 'cons.price.idx', 'contact']

The specific values of these important features are:
<a href="./analysis_results/module_17_01.step11.improving_model.experiment4.important_features.dataFrame.png" target="_blank"><img src="./analysis_results/module_17_01.step11.improving_model.experiment4.important_features.dataFrame.png"/></a>

#### Analysis
#### Data Distributions
**NOTE:** Important features highlighted using different palette
<a href="./analysis_results/module_17_01.step12.final_analysis.important.categorical.data.distribution.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis.important.categorical.data.distribution.png"/></a>
<a href="./analysis_results/module_17_01.step12.final_analysis.important.age.data.distribution.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis.important.age.data.distribution.png"/></a>

<a href="./analysis_results/module_17_01.step12.final_analysis.important.numeric.data.distribution.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis.important.numeric.data.distribution.png"/></a>

#### Decision Tree
The descision tree is more easily interpretable/explainable to help with the understanding how the model works and can help the bank make decisions about optimizing the campaign.
<a href="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png" target="_blank"><img src="./analysis_results/module_17_01.step12.final_analysis..decision_tree_final.png"/></a>
