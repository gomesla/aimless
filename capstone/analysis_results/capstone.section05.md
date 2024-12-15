## Modeling
### Feature Engineering Decisions
- No real feature engineering is needed as we have no missing data and nothing to impute. The preprocessing takes care of most of the needs for stemming and lemmatization before our modeling stage.
- When creating intial train and test sets (30% of data for testing) we will stratify the set so that both train and test sets contain similar percentages of the classes because they are imbalanced.
- When we do our grid search split because the target classes are imbalanced we will use StratifiedKFold so the splits are representative of the orginal set and class balances.
- We will try the following models:
  - LogisticRegression: We will use elasticnet since it intergates l1 and l2 penalties rather than having to pick between ridge and lasso.
  - Decision Tree: While not expecting the best performance a decision tree could help us in explainability of how a decision is made and derive important "features/tokens"
  - Naive Bayes: We will use Multinomail and Complement. Complement is meant to deal better with imbalanced classes but we shall see.
  - KNN: Will use k nearest neigbhours, one would expect that tickets in the same category would have similar words and so would cluster well together
- In addition we will try some ensemble techniques:
  - XGBoost: We will use the XGBClassifier with a multi:softmax objective since we have a non binary classification case.
  - Random Forest: We will use the RandomForestClassifier which will combine multiple Decision tree created using random subsets of data.
  - Bagging: We will use the BaggingClassifier which will combine multiple "weak" classifiers (In this case DecisionTreeClassifier but we have control of using other models should we want to try).

### Data Distribution
Checking to make sure our test and train datasets represent the class imbalances
<table>
<tr>
<td>
<a href="./analysis_results/capstone.model_results.y_train.targetField.distribution.png" target="_blank"><img src="./analysis_results/capstone.model_results.y_train.targetField.distribution.png"/></a></td>
<td>
<a href="./analysis_results/capstone.model_results.y_test.targetField.distribution.png" target="_blank"><img src="./analysis_results/capstone.model_results.y_test.targetField.distribution.png"/></a></td>
</tr>
</table>

### Model Results
<style type="text/css">
#T_29042 th {
  font-size: 8pt;
  font-family: Verdana;
}
#T_29042_row0_col0, #T_29042_row0_col1, #T_29042_row0_col2, #T_29042_row0_col3, #T_29042_row0_col4, #T_29042_row0_col5, #T_29042_row0_col6, #T_29042_row1_col0, #T_29042_row1_col1, #T_29042_row1_col2, #T_29042_row1_col3, #T_29042_row1_col4, #T_29042_row1_col5, #T_29042_row1_col6, #T_29042_row2_col0, #T_29042_row2_col1, #T_29042_row2_col2, #T_29042_row2_col3, #T_29042_row2_col4, #T_29042_row2_col5, #T_29042_row2_col6, #T_29042_row3_col0, #T_29042_row3_col1, #T_29042_row3_col2, #T_29042_row3_col3, #T_29042_row3_col4, #T_29042_row3_col5, #T_29042_row3_col6, #T_29042_row4_col0, #T_29042_row4_col1, #T_29042_row4_col2, #T_29042_row4_col3, #T_29042_row4_col4, #T_29042_row4_col5, #T_29042_row4_col6, #T_29042_row5_col0, #T_29042_row5_col1, #T_29042_row5_col2, #T_29042_row5_col4, #T_29042_row5_col5, #T_29042_row5_col6, #T_29042_row6_col0, #T_29042_row6_col1, #T_29042_row6_col2, #T_29042_row6_col3, #T_29042_row6_col4, #T_29042_row6_col5, #T_29042_row6_col6, #T_29042_row7_col0, #T_29042_row7_col1, #T_29042_row7_col2, #T_29042_row7_col3, #T_29042_row7_col4, #T_29042_row7_col5, #T_29042_row7_col6, #T_29042_row8_col0, #T_29042_row8_col1, #T_29042_row8_col2, #T_29042_row8_col3, #T_29042_row8_col4, #T_29042_row8_col5, #T_29042_row8_col6, #T_29042_row9_col0, #T_29042_row9_col1, #T_29042_row9_col2, #T_29042_row9_col3, #T_29042_row9_col4, #T_29042_row9_col5, #T_29042_row9_col6, #T_29042_row10_col0, #T_29042_row10_col1, #T_29042_row10_col2, #T_29042_row10_col3, #T_29042_row10_col4, #T_29042_row10_col5, #T_29042_row10_col6, #T_29042_row11_col0, #T_29042_row11_col1, #T_29042_row11_col2, #T_29042_row11_col3, #T_29042_row11_col4, #T_29042_row11_col5, #T_29042_row11_col6, #T_29042_row12_col0, #T_29042_row12_col1, #T_29042_row12_col2, #T_29042_row12_col3, #T_29042_row12_col4, #T_29042_row12_col5, #T_29042_row12_col6, #T_29042_row13_col0, #T_29042_row13_col1, #T_29042_row13_col2, #T_29042_row13_col3, #T_29042_row13_col4, #T_29042_row13_col5, #T_29042_row13_col6, #T_29042_row14_col0, #T_29042_row14_col1, #T_29042_row14_col2, #T_29042_row14_col3, #T_29042_row14_col4, #T_29042_row14_col5, #T_29042_row14_col6, #T_29042_row15_col0, #T_29042_row15_col1, #T_29042_row15_col2, #T_29042_row15_col3, #T_29042_row15_col4, #T_29042_row15_col5, #T_29042_row15_col6, #T_29042_row16_col0, #T_29042_row16_col1, #T_29042_row16_col2, #T_29042_row16_col3, #T_29042_row16_col4, #T_29042_row16_col5, #T_29042_row16_col6, #T_29042_row17_col0, #T_29042_row17_col1, #T_29042_row17_col2, #T_29042_row17_col3, #T_29042_row17_col4, #T_29042_row17_col5, #T_29042_row17_col6, #T_29042_row18_col0, #T_29042_row18_col1, #T_29042_row18_col2, #T_29042_row18_col3, #T_29042_row18_col4, #T_29042_row18_col5, #T_29042_row18_col6, #T_29042_row19_col0, #T_29042_row19_col1, #T_29042_row19_col2, #T_29042_row19_col3, #T_29042_row19_col4, #T_29042_row19_col5, #T_29042_row19_col6, #T_29042_row20_col0, #T_29042_row20_col1, #T_29042_row20_col2, #T_29042_row20_col3, #T_29042_row20_col4, #T_29042_row20_col5, #T_29042_row20_col6, #T_29042_row21_col0, #T_29042_row21_col1, #T_29042_row21_col2, #T_29042_row21_col3, #T_29042_row21_col4, #T_29042_row21_col5, #T_29042_row21_col6, #T_29042_row22_col0, #T_29042_row22_col1, #T_29042_row22_col2, #T_29042_row22_col3, #T_29042_row22_col4, #T_29042_row22_col5, #T_29042_row22_col6, #T_29042_row23_col0, #T_29042_row23_col1, #T_29042_row23_col2, #T_29042_row23_col3, #T_29042_row23_col4, #T_29042_row23_col5, #T_29042_row23_col6, #T_29042_row24_col0, #T_29042_row24_col1, #T_29042_row24_col2, #T_29042_row24_col3, #T_29042_row24_col4, #T_29042_row24_col5, #T_29042_row24_col6, #T_29042_row25_col0, #T_29042_row25_col1, #T_29042_row25_col2, #T_29042_row25_col3, #T_29042_row25_col4, #T_29042_row25_col5, #T_29042_row25_col6, #T_29042_row26_col0, #T_29042_row26_col1, #T_29042_row26_col2, #T_29042_row26_col3, #T_29042_row26_col4, #T_29042_row26_col5, #T_29042_row26_col6, #T_29042_row27_col0, #T_29042_row27_col1, #T_29042_row27_col2, #T_29042_row27_col3, #T_29042_row27_col5, #T_29042_row27_col6, #T_29042_row28_col0, #T_29042_row28_col1, #T_29042_row28_col2, #T_29042_row28_col3, #T_29042_row28_col4, #T_29042_row28_col5, #T_29042_row28_col6, #T_29042_row29_col0, #T_29042_row29_col1, #T_29042_row29_col2, #T_29042_row29_col3, #T_29042_row29_col4, #T_29042_row29_col5, #T_29042_row29_col6, #T_29042_row30_col0, #T_29042_row30_col1, #T_29042_row30_col2, #T_29042_row30_col3, #T_29042_row30_col4, #T_29042_row30_col5, #T_29042_row30_col6, #T_29042_row31_col0, #T_29042_row31_col1, #T_29042_row31_col2, #T_29042_row31_col3, #T_29042_row31_col4, #T_29042_row31_col5, #T_29042_row31_col6, #T_29042_row32_col0, #T_29042_row32_col1, #T_29042_row32_col2, #T_29042_row32_col3, #T_29042_row32_col5, #T_29042_row32_col6, #T_29042_row33_col0, #T_29042_row33_col1, #T_29042_row33_col2, #T_29042_row33_col3, #T_29042_row33_col4, #T_29042_row33_col5, #T_29042_row33_col6, #T_29042_row34_col0, #T_29042_row34_col1, #T_29042_row34_col2, #T_29042_row34_col3, #T_29042_row34_col4, #T_29042_row34_col5, #T_29042_row34_col6, #T_29042_row35_col0, #T_29042_row35_col1, #T_29042_row35_col2, #T_29042_row35_col3, #T_29042_row35_col4, #T_29042_row35_col5, #T_29042_row35_col6, #T_29042_row36_col0, #T_29042_row36_col1, #T_29042_row36_col2, #T_29042_row36_col3, #T_29042_row36_col4, #T_29042_row36_col5, #T_29042_row36_col6, #T_29042_row37_col0, #T_29042_row37_col1, #T_29042_row37_col2, #T_29042_row37_col3, #T_29042_row37_col4, #T_29042_row37_col5, #T_29042_row37_col6, #T_29042_row38_col0, #T_29042_row38_col1, #T_29042_row38_col2, #T_29042_row38_col3, #T_29042_row38_col4, #T_29042_row38_col5, #T_29042_row38_col6, #T_29042_row39_col0, #T_29042_row39_col1, #T_29042_row39_col2, #T_29042_row39_col3, #T_29042_row39_col4, #T_29042_row39_col5, #T_29042_row39_col6, #T_29042_row40_col0, #T_29042_row40_col1, #T_29042_row40_col2, #T_29042_row40_col3, #T_29042_row40_col4, #T_29042_row40_col5, #T_29042_row40_col6, #T_29042_row41_col0, #T_29042_row41_col1, #T_29042_row41_col2, #T_29042_row41_col3, #T_29042_row41_col4, #T_29042_row41_col5, #T_29042_row41_col6, #T_29042_row42_col0, #T_29042_row42_col1, #T_29042_row42_col2, #T_29042_row42_col3, #T_29042_row42_col4, #T_29042_row42_col5, #T_29042_row42_col6, #T_29042_row43_col0, #T_29042_row43_col1, #T_29042_row43_col2, #T_29042_row43_col3, #T_29042_row43_col4, #T_29042_row43_col5, #T_29042_row43_col6, #T_29042_row44_col0, #T_29042_row44_col1, #T_29042_row44_col2, #T_29042_row44_col4, #T_29042_row44_col5, #T_29042_row44_col6, #T_29042_row45_col0, #T_29042_row45_col1, #T_29042_row45_col2, #T_29042_row45_col3, #T_29042_row45_col4, #T_29042_row45_col5, #T_29042_row45_col6, #T_29042_row46_col0, #T_29042_row46_col1, #T_29042_row46_col2, #T_29042_row46_col3, #T_29042_row46_col4, #T_29042_row46_col5, #T_29042_row46_col6, #T_29042_row47_col0, #T_29042_row47_col1, #T_29042_row47_col2, #T_29042_row47_col3, #T_29042_row47_col4, #T_29042_row47_col5, #T_29042_row47_col6, #T_29042_row48_col0, #T_29042_row48_col1, #T_29042_row48_col2, #T_29042_row48_col3, #T_29042_row48_col4, #T_29042_row48_col5, #T_29042_row48_col6, #T_29042_row49_col0, #T_29042_row49_col1, #T_29042_row49_col2, #T_29042_row49_col3, #T_29042_row49_col4, #T_29042_row49_col5, #T_29042_row49_col6, #T_29042_row50_col0, #T_29042_row50_col1, #T_29042_row50_col2, #T_29042_row50_col3, #T_29042_row50_col4, #T_29042_row50_col5, #T_29042_row50_col6, #T_29042_row51_col0, #T_29042_row51_col1, #T_29042_row51_col2, #T_29042_row51_col3, #T_29042_row51_col4, #T_29042_row51_col5, #T_29042_row51_col6, #T_29042_row52_col0, #T_29042_row52_col1, #T_29042_row52_col2, #T_29042_row52_col3, #T_29042_row52_col4, #T_29042_row52_col5, #T_29042_row52_col6, #T_29042_row53_col0, #T_29042_row53_col1, #T_29042_row53_col2, #T_29042_row53_col3, #T_29042_row53_col4, #T_29042_row53_col5, #T_29042_row53_col6, #T_29042_row54_col0, #T_29042_row54_col1, #T_29042_row54_col2, #T_29042_row54_col3, #T_29042_row54_col4, #T_29042_row54_col5, #T_29042_row54_col6, #T_29042_row55_col0, #T_29042_row55_col1, #T_29042_row55_col2, #T_29042_row55_col3, #T_29042_row55_col4, #T_29042_row55_col5, #T_29042_row55_col6, #T_29042_row56_col0, #T_29042_row56_col1, #T_29042_row56_col2, #T_29042_row56_col3, #T_29042_row56_col4, #T_29042_row56_col5, #T_29042_row56_col6, #T_29042_row57_col0, #T_29042_row57_col1, #T_29042_row57_col2, #T_29042_row57_col3, #T_29042_row57_col4, #T_29042_row57_col5, #T_29042_row57_col6, #T_29042_row58_col0, #T_29042_row58_col1, #T_29042_row58_col2, #T_29042_row58_col3, #T_29042_row58_col4, #T_29042_row58_col5, #T_29042_row58_col6, #T_29042_row59_col0, #T_29042_row59_col1, #T_29042_row59_col2, #T_29042_row59_col3, #T_29042_row59_col4, #T_29042_row59_col5, #T_29042_row59_col6, #T_29042_row60_col0, #T_29042_row60_col1, #T_29042_row60_col2, #T_29042_row60_col3, #T_29042_row60_col4, #T_29042_row60_col5, #T_29042_row60_col6, #T_29042_row61_col0, #T_29042_row61_col1, #T_29042_row61_col2, #T_29042_row61_col3, #T_29042_row61_col4, #T_29042_row61_col5, #T_29042_row61_col6, #T_29042_row62_col0, #T_29042_row62_col1, #T_29042_row62_col2, #T_29042_row62_col3, #T_29042_row62_col4, #T_29042_row62_col5, #T_29042_row62_col6, #T_29042_row63_col0, #T_29042_row63_col1, #T_29042_row63_col2, #T_29042_row63_col3, #T_29042_row63_col4, #T_29042_row63_col5, #T_29042_row63_col6 {
  font-size: 8pt;
  font-family: Verdana;
}
#T_29042_row5_col3, #T_29042_row32_col4 {
  font-size: 8pt;
  font-family: Verdana;
  font-weight: bold;
  background-color: #FF8A8A;
}
#T_29042_row27_col4, #T_29042_row44_col3 {
  font-size: 8pt;
  font-family: Verdana;
  font-weight: bold;
  background-color: #CCE0AC;
}
</style>
<table id="T_29042">
  <thead>
    <tr>
      <th id="T_29042_level0_col0" class="col_heading level0 col0" >model</th>
      <th id="T_29042_level0_col1" class="col_heading level0 col1" >vectorizer</th>
      <th id="T_29042_level0_col2" class="col_heading level0 col2" >input_field</th>
      <th id="T_29042_level0_col3" class="col_heading level0 col3" >best_score</th>
      <th id="T_29042_level0_col4" class="col_heading level0 col4" >mean_fit_time</th>
      <th id="T_29042_level0_col5" class="col_heading level0 col5" >best_params</th>
      <th id="T_29042_level0_col6" class="col_heading level0 col6" >run_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_29042_row0_col0" class="data row0 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row0_col1" class="data row0 col1" >CountVectorizer</td>
      <td id="T_29042_row0_col2" class="data row0 col2" >Original</td>
      <td id="T_29042_row0_col3" class="data row0 col3" >0.880906</td>
      <td id="T_29042_row0_col4" class="data row0 col4" >1.193505</td>
      <td id="T_29042_row0_col5" class="data row0 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_29042_row0_col6" class="data row0 col6" >29.376820</td>
    </tr>
    <tr>
      <td id="T_29042_row1_col0" class="data row1 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row1_col1" class="data row1 col1" >CountVectorizer</td>
      <td id="T_29042_row1_col2" class="data row1 col2" >Stemmed</td>
      <td id="T_29042_row1_col3" class="data row1 col3" >0.883830</td>
      <td id="T_29042_row1_col4" class="data row1 col4" >1.059290</td>
      <td id="T_29042_row1_col5" class="data row1 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_29042_row1_col6" class="data row1 col6" >25.438417</td>
    </tr>
    <tr>
      <td id="T_29042_row2_col0" class="data row2 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row2_col1" class="data row2 col1" >CountVectorizer</td>
      <td id="T_29042_row2_col2" class="data row2 col2" >Lemmatized</td>
      <td id="T_29042_row2_col3" class="data row2 col3" >0.882731</td>
      <td id="T_29042_row2_col4" class="data row2 col4" >1.046649</td>
      <td id="T_29042_row2_col5" class="data row2 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_29042_row2_col6" class="data row2 col6" >25.314366</td>
    </tr>
    <tr>
      <td id="T_29042_row3_col0" class="data row3 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row3_col1" class="data row3 col1" >CountVectorizer</td>
      <td id="T_29042_row3_col2" class="data row3 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row3_col3" class="data row3 col3" >0.885090</td>
      <td id="T_29042_row3_col4" class="data row3 col4" >1.049817</td>
      <td id="T_29042_row3_col5" class="data row3 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_29042_row3_col6" class="data row3 col6" >25.283688</td>
    </tr>
    <tr>
      <td id="T_29042_row4_col0" class="data row4 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row4_col1" class="data row4 col1" >TfidfVectorizer</td>
      <td id="T_29042_row4_col2" class="data row4 col2" >Original</td>
      <td id="T_29042_row4_col3" class="data row4 col3" >0.883312</td>
      <td id="T_29042_row4_col4" class="data row4 col4" >2.203545</td>
      <td id="T_29042_row4_col5" class="data row4 col5" >{"model__criterion": "gini", "model__max_depth": 50, "vectorizer__max_features": null}</td>
      <td id="T_29042_row4_col6" class="data row4 col6" >51.766405</td>
    </tr>
    <tr>
      <td id="T_29042_row5_col0" class="data row5 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row5_col1" class="data row5 col1" >TfidfVectorizer</td>
      <td id="T_29042_row5_col2" class="data row5 col2" >Stemmed</td>
      <td id="T_29042_row5_col3" class="data row5 col3" >0.877467</td>
      <td id="T_29042_row5_col4" class="data row5 col4" >1.915424</td>
      <td id="T_29042_row5_col5" class="data row5 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_29042_row5_col6" class="data row5 col6" >44.117501</td>
    </tr>
    <tr>
      <td id="T_29042_row6_col0" class="data row6 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row6_col1" class="data row6 col1" >TfidfVectorizer</td>
      <td id="T_29042_row6_col2" class="data row6 col2" >Lemmatized</td>
      <td id="T_29042_row6_col3" class="data row6 col3" >0.880878</td>
      <td id="T_29042_row6_col4" class="data row6 col4" >1.896291</td>
      <td id="T_29042_row6_col5" class="data row6 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_29042_row6_col6" class="data row6 col6" >43.583669</td>
    </tr>
    <tr>
      <td id="T_29042_row7_col0" class="data row7 col0" >DecisionTreeClassifier</td>
      <td id="T_29042_row7_col1" class="data row7 col1" >TfidfVectorizer</td>
      <td id="T_29042_row7_col2" class="data row7 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row7_col3" class="data row7 col3" >0.881266</td>
      <td id="T_29042_row7_col4" class="data row7 col4" >1.911532</td>
      <td id="T_29042_row7_col5" class="data row7 col5" >{"model__criterion": "gini", "model__max_depth": 50, "vectorizer__max_features": null}</td>
      <td id="T_29042_row7_col6" class="data row7 col6" >44.550871</td>
    </tr>
    <tr>
      <td id="T_29042_row8_col0" class="data row8 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row8_col1" class="data row8 col1" >CountVectorizer</td>
      <td id="T_29042_row8_col2" class="data row8 col2" >Original</td>
      <td id="T_29042_row8_col3" class="data row8 col3" >0.931534</td>
      <td id="T_29042_row8_col4" class="data row8 col4" >0.312508</td>
      <td id="T_29042_row8_col5" class="data row8 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_29042_row8_col6" class="data row8 col6" >107.176686</td>
    </tr>
    <tr>
      <td id="T_29042_row9_col0" class="data row9 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row9_col1" class="data row9 col1" >CountVectorizer</td>
      <td id="T_29042_row9_col2" class="data row9 col2" >Stemmed</td>
      <td id="T_29042_row9_col3" class="data row9 col3" >0.942798</td>
      <td id="T_29042_row9_col4" class="data row9 col4" >0.255802</td>
      <td id="T_29042_row9_col5" class="data row9 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_29042_row9_col6" class="data row9 col6" >165.264951</td>
    </tr>
    <tr>
      <td id="T_29042_row10_col0" class="data row10 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row10_col1" class="data row10 col1" >CountVectorizer</td>
      <td id="T_29042_row10_col2" class="data row10 col2" >Lemmatized</td>
      <td id="T_29042_row10_col3" class="data row10 col3" >0.938484</td>
      <td id="T_29042_row10_col4" class="data row10 col4" >0.270951</td>
      <td id="T_29042_row10_col5" class="data row10 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_29042_row10_col6" class="data row10 col6" >104.686211</td>
    </tr>
    <tr>
      <td id="T_29042_row11_col0" class="data row11 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row11_col1" class="data row11 col1" >CountVectorizer</td>
      <td id="T_29042_row11_col2" class="data row11 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row11_col3" class="data row11 col3" >0.942643</td>
      <td id="T_29042_row11_col4" class="data row11 col4" >0.247672</td>
      <td id="T_29042_row11_col5" class="data row11 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_29042_row11_col6" class="data row11 col6" >165.267842</td>
    </tr>
    <tr>
      <td id="T_29042_row12_col0" class="data row12 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row12_col1" class="data row12 col1" >TfidfVectorizer</td>
      <td id="T_29042_row12_col2" class="data row12 col2" >Original</td>
      <td id="T_29042_row12_col3" class="data row12 col3" >0.973472</td>
      <td id="T_29042_row12_col4" class="data row12 col4" >0.328251</td>
      <td id="T_29042_row12_col5" class="data row12 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_29042_row12_col6" class="data row12 col6" >121.556900</td>
    </tr>
    <tr>
      <td id="T_29042_row13_col0" class="data row13 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row13_col1" class="data row13 col1" >TfidfVectorizer</td>
      <td id="T_29042_row13_col2" class="data row13 col2" >Stemmed</td>
      <td id="T_29042_row13_col3" class="data row13 col3" >0.971769</td>
      <td id="T_29042_row13_col4" class="data row13 col4" >0.270268</td>
      <td id="T_29042_row13_col5" class="data row13 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_29042_row13_col6" class="data row13 col6" >137.881482</td>
    </tr>
    <tr>
      <td id="T_29042_row14_col0" class="data row14 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row14_col1" class="data row14 col1" >TfidfVectorizer</td>
      <td id="T_29042_row14_col2" class="data row14 col2" >Lemmatized</td>
      <td id="T_29042_row14_col3" class="data row14 col3" >0.972244</td>
      <td id="T_29042_row14_col4" class="data row14 col4" >0.270533</td>
      <td id="T_29042_row14_col5" class="data row14 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_29042_row14_col6" class="data row14 col6" >106.810837</td>
    </tr>
    <tr>
      <td id="T_29042_row15_col0" class="data row15 col0" >KNeighborsClassifier</td>
      <td id="T_29042_row15_col1" class="data row15 col1" >TfidfVectorizer</td>
      <td id="T_29042_row15_col2" class="data row15 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row15_col3" class="data row15 col3" >0.971738</td>
      <td id="T_29042_row15_col4" class="data row15 col4" >0.258624</td>
      <td id="T_29042_row15_col5" class="data row15 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_29042_row15_col6" class="data row15 col6" >152.140115</td>
    </tr>
    <tr>
      <td id="T_29042_row16_col0" class="data row16 col0" >MultinomialNB</td>
      <td id="T_29042_row16_col1" class="data row16 col1" >CountVectorizer</td>
      <td id="T_29042_row16_col2" class="data row16 col2" >Original</td>
      <td id="T_29042_row16_col3" class="data row16 col3" >0.953175</td>
      <td id="T_29042_row16_col4" class="data row16 col4" >0.286969</td>
      <td id="T_29042_row16_col5" class="data row16 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row16_col6" class="data row16 col6" >5.061755</td>
    </tr>
    <tr>
      <td id="T_29042_row17_col0" class="data row17 col0" >MultinomialNB</td>
      <td id="T_29042_row17_col1" class="data row17 col1" >CountVectorizer</td>
      <td id="T_29042_row17_col2" class="data row17 col2" >Stemmed</td>
      <td id="T_29042_row17_col3" class="data row17 col3" >0.954360</td>
      <td id="T_29042_row17_col4" class="data row17 col4" >0.236176</td>
      <td id="T_29042_row17_col5" class="data row17 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row17_col6" class="data row17 col6" >4.384494</td>
    </tr>
    <tr>
      <td id="T_29042_row18_col0" class="data row18 col0" >MultinomialNB</td>
      <td id="T_29042_row18_col1" class="data row18 col1" >CountVectorizer</td>
      <td id="T_29042_row18_col2" class="data row18 col2" >Lemmatized</td>
      <td id="T_29042_row18_col3" class="data row18 col3" >0.955302</td>
      <td id="T_29042_row18_col4" class="data row18 col4" >0.247204</td>
      <td id="T_29042_row18_col5" class="data row18 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row18_col6" class="data row18 col6" >4.439699</td>
    </tr>
    <tr>
      <td id="T_29042_row19_col0" class="data row19 col0" >MultinomialNB</td>
      <td id="T_29042_row19_col1" class="data row19 col1" >CountVectorizer</td>
      <td id="T_29042_row19_col2" class="data row19 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row19_col3" class="data row19 col3" >0.954424</td>
      <td id="T_29042_row19_col4" class="data row19 col4" >0.238093</td>
      <td id="T_29042_row19_col5" class="data row19 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row19_col6" class="data row19 col6" >4.474550</td>
    </tr>
    <tr>
      <td id="T_29042_row20_col0" class="data row20 col0" >MultinomialNB</td>
      <td id="T_29042_row20_col1" class="data row20 col1" >TfidfVectorizer</td>
      <td id="T_29042_row20_col2" class="data row20 col2" >Original</td>
      <td id="T_29042_row20_col3" class="data row20 col3" >0.964878</td>
      <td id="T_29042_row20_col4" class="data row20 col4" >0.296716</td>
      <td id="T_29042_row20_col5" class="data row20 col5" >{"model__alpha": 0.1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row20_col6" class="data row20 col6" >5.114765</td>
    </tr>
    <tr>
      <td id="T_29042_row21_col0" class="data row21 col0" >MultinomialNB</td>
      <td id="T_29042_row21_col1" class="data row21 col1" >TfidfVectorizer</td>
      <td id="T_29042_row21_col2" class="data row21 col2" >Stemmed</td>
      <td id="T_29042_row21_col3" class="data row21 col3" >0.965847</td>
      <td id="T_29042_row21_col4" class="data row21 col4" >0.239766</td>
      <td id="T_29042_row21_col5" class="data row21 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_29042_row21_col6" class="data row21 col6" >4.524948</td>
    </tr>
    <tr>
      <td id="T_29042_row22_col0" class="data row22 col0" >MultinomialNB</td>
      <td id="T_29042_row22_col1" class="data row22 col1" >TfidfVectorizer</td>
      <td id="T_29042_row22_col2" class="data row22 col2" >Lemmatized</td>
      <td id="T_29042_row22_col3" class="data row22 col3" >0.965914</td>
      <td id="T_29042_row22_col4" class="data row22 col4" >0.254408</td>
      <td id="T_29042_row22_col5" class="data row22 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_29042_row22_col6" class="data row22 col6" >4.696984</td>
    </tr>
    <tr>
      <td id="T_29042_row23_col0" class="data row23 col0" >MultinomialNB</td>
      <td id="T_29042_row23_col1" class="data row23 col1" >TfidfVectorizer</td>
      <td id="T_29042_row23_col2" class="data row23 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row23_col3" class="data row23 col3" >0.965847</td>
      <td id="T_29042_row23_col4" class="data row23 col4" >0.239786</td>
      <td id="T_29042_row23_col5" class="data row23 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_29042_row23_col6" class="data row23 col6" >4.393879</td>
    </tr>
    <tr>
      <td id="T_29042_row24_col0" class="data row24 col0" >ComplementNB</td>
      <td id="T_29042_row24_col1" class="data row24 col1" >CountVectorizer</td>
      <td id="T_29042_row24_col2" class="data row24 col2" >Original</td>
      <td id="T_29042_row24_col3" class="data row24 col3" >0.944329</td>
      <td id="T_29042_row24_col4" class="data row24 col4" >0.291545</td>
      <td id="T_29042_row24_col5" class="data row24 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row24_col6" class="data row24 col6" >5.216317</td>
    </tr>
    <tr>
      <td id="T_29042_row25_col0" class="data row25 col0" >ComplementNB</td>
      <td id="T_29042_row25_col1" class="data row25 col1" >CountVectorizer</td>
      <td id="T_29042_row25_col2" class="data row25 col2" >Stemmed</td>
      <td id="T_29042_row25_col3" class="data row25 col3" >0.943821</td>
      <td id="T_29042_row25_col4" class="data row25 col4" >0.236715</td>
      <td id="T_29042_row25_col5" class="data row25 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row25_col6" class="data row25 col6" >4.530648</td>
    </tr>
    <tr>
      <td id="T_29042_row26_col0" class="data row26 col0" >ComplementNB</td>
      <td id="T_29042_row26_col1" class="data row26 col1" >CountVectorizer</td>
      <td id="T_29042_row26_col2" class="data row26 col2" >Lemmatized</td>
      <td id="T_29042_row26_col3" class="data row26 col3" >0.945060</td>
      <td id="T_29042_row26_col4" class="data row26 col4" >0.246268</td>
      <td id="T_29042_row26_col5" class="data row26 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row26_col6" class="data row26 col6" >4.464471</td>
    </tr>
    <tr>
      <td id="T_29042_row27_col0" class="data row27 col0" >ComplementNB</td>
      <td id="T_29042_row27_col1" class="data row27 col1" >CountVectorizer</td>
      <td id="T_29042_row27_col2" class="data row27 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row27_col3" class="data row27 col3" >0.943771</td>
      <td id="T_29042_row27_col4" class="data row27 col4" >0.234908</td>
      <td id="T_29042_row27_col5" class="data row27 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row27_col6" class="data row27 col6" >4.372359</td>
    </tr>
    <tr>
      <td id="T_29042_row28_col0" class="data row28 col0" >ComplementNB</td>
      <td id="T_29042_row28_col1" class="data row28 col1" >TfidfVectorizer</td>
      <td id="T_29042_row28_col2" class="data row28 col2" >Original</td>
      <td id="T_29042_row28_col3" class="data row28 col3" >0.971004</td>
      <td id="T_29042_row28_col4" class="data row28 col4" >0.305616</td>
      <td id="T_29042_row28_col5" class="data row28 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row28_col6" class="data row28 col6" >5.484758</td>
    </tr>
    <tr>
      <td id="T_29042_row29_col0" class="data row29 col0" >ComplementNB</td>
      <td id="T_29042_row29_col1" class="data row29 col1" >TfidfVectorizer</td>
      <td id="T_29042_row29_col2" class="data row29 col2" >Stemmed</td>
      <td id="T_29042_row29_col3" class="data row29 col3" >0.969519</td>
      <td id="T_29042_row29_col4" class="data row29 col4" >0.237895</td>
      <td id="T_29042_row29_col5" class="data row29 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row29_col6" class="data row29 col6" >4.435607</td>
    </tr>
    <tr>
      <td id="T_29042_row30_col0" class="data row30 col0" >ComplementNB</td>
      <td id="T_29042_row30_col1" class="data row30 col1" >TfidfVectorizer</td>
      <td id="T_29042_row30_col2" class="data row30 col2" >Lemmatized</td>
      <td id="T_29042_row30_col3" class="data row30 col3" >0.969854</td>
      <td id="T_29042_row30_col4" class="data row30 col4" >0.251077</td>
      <td id="T_29042_row30_col5" class="data row30 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row30_col6" class="data row30 col6" >4.600548</td>
    </tr>
    <tr>
      <td id="T_29042_row31_col0" class="data row31 col0" >ComplementNB</td>
      <td id="T_29042_row31_col1" class="data row31 col1" >TfidfVectorizer</td>
      <td id="T_29042_row31_col2" class="data row31 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row31_col3" class="data row31 col3" >0.969578</td>
      <td id="T_29042_row31_col4" class="data row31 col4" >0.237831</td>
      <td id="T_29042_row31_col5" class="data row31 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row31_col6" class="data row31 col6" >4.595905</td>
    </tr>
    <tr>
      <td id="T_29042_row32_col0" class="data row32 col0" >LogisticRegression</td>
      <td id="T_29042_row32_col1" class="data row32 col1" >CountVectorizer</td>
      <td id="T_29042_row32_col2" class="data row32 col2" >Original</td>
      <td id="T_29042_row32_col3" class="data row32 col3" >0.958691</td>
      <td id="T_29042_row32_col4" class="data row32 col4" >90.873420</td>
      <td id="T_29042_row32_col5" class="data row32 col5" >{"model__C": 100, "model__l1_ratio": 0.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row32_col6" class="data row32 col6" >5008.611961</td>
    </tr>
    <tr>
      <td id="T_29042_row33_col0" class="data row33 col0" >LogisticRegression</td>
      <td id="T_29042_row33_col1" class="data row33 col1" >CountVectorizer</td>
      <td id="T_29042_row33_col2" class="data row33 col2" >Stemmed</td>
      <td id="T_29042_row33_col3" class="data row33 col3" >0.960372</td>
      <td id="T_29042_row33_col4" class="data row33 col4" >57.457154</td>
      <td id="T_29042_row33_col5" class="data row33 col5" >{"model__C": 100, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row33_col6" class="data row33 col6" >3743.248646</td>
    </tr>
    <tr>
      <td id="T_29042_row34_col0" class="data row34 col0" >LogisticRegression</td>
      <td id="T_29042_row34_col1" class="data row34 col1" >CountVectorizer</td>
      <td id="T_29042_row34_col2" class="data row34 col2" >Lemmatized</td>
      <td id="T_29042_row34_col3" class="data row34 col3" >0.960110</td>
      <td id="T_29042_row34_col4" class="data row34 col4" >78.963146</td>
      <td id="T_29042_row34_col5" class="data row34 col5" >{"model__C": 100, "model__l1_ratio": 0.5, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row34_col6" class="data row34 col6" >5280.003322</td>
    </tr>
    <tr>
      <td id="T_29042_row35_col0" class="data row35 col0" >LogisticRegression</td>
      <td id="T_29042_row35_col1" class="data row35 col1" >CountVectorizer</td>
      <td id="T_29042_row35_col2" class="data row35 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row35_col3" class="data row35 col3" >0.960340</td>
      <td id="T_29042_row35_col4" class="data row35 col4" >57.338801</td>
      <td id="T_29042_row35_col5" class="data row35 col5" >{"model__C": 100, "model__l1_ratio": 0.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row35_col6" class="data row35 col6" >3101.679289</td>
    </tr>
    <tr>
      <td id="T_29042_row36_col0" class="data row36 col0" >LogisticRegression</td>
      <td id="T_29042_row36_col1" class="data row36 col1" >TfidfVectorizer</td>
      <td id="T_29042_row36_col2" class="data row36 col2" >Original</td>
      <td id="T_29042_row36_col3" class="data row36 col3" >0.986617</td>
      <td id="T_29042_row36_col4" class="data row36 col4" >25.333251</td>
      <td id="T_29042_row36_col5" class="data row36 col5" >{"model__C": 1, "model__l1_ratio": 0.5, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row36_col6" class="data row36 col6" >1380.814533</td>
    </tr>
    <tr>
      <td id="T_29042_row37_col0" class="data row37 col0" >LogisticRegression</td>
      <td id="T_29042_row37_col1" class="data row37 col1" >TfidfVectorizer</td>
      <td id="T_29042_row37_col2" class="data row37 col2" >Stemmed</td>
      <td id="T_29042_row37_col3" class="data row37 col3" >0.984014</td>
      <td id="T_29042_row37_col4" class="data row37 col4" >19.186127</td>
      <td id="T_29042_row37_col5" class="data row37 col5" >{"model__C": 1, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row37_col6" class="data row37 col6" >1046.609349</td>
    </tr>
    <tr>
      <td id="T_29042_row38_col0" class="data row38 col0" >LogisticRegression</td>
      <td id="T_29042_row38_col1" class="data row38 col1" >TfidfVectorizer</td>
      <td id="T_29042_row38_col2" class="data row38 col2" >Lemmatized</td>
      <td id="T_29042_row38_col3" class="data row38 col3" >0.984918</td>
      <td id="T_29042_row38_col4" class="data row38 col4" >23.884809</td>
      <td id="T_29042_row38_col5" class="data row38 col5" >{"model__C": 1, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row38_col6" class="data row38 col6" >1384.790200</td>
    </tr>
    <tr>
      <td id="T_29042_row39_col0" class="data row39 col0" >LogisticRegression</td>
      <td id="T_29042_row39_col1" class="data row39 col1" >TfidfVectorizer</td>
      <td id="T_29042_row39_col2" class="data row39 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row39_col3" class="data row39 col3" >0.984005</td>
      <td id="T_29042_row39_col4" class="data row39 col4" >19.118844</td>
      <td id="T_29042_row39_col5" class="data row39 col5" >{"model__C": 1, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_29042_row39_col6" class="data row39 col6" >1048.918884</td>
    </tr>
    <tr>
      <td id="T_29042_row40_col0" class="data row40 col0" >XGBClassifier</td>
      <td id="T_29042_row40_col1" class="data row40 col1" >CountVectorizer</td>
      <td id="T_29042_row40_col2" class="data row40 col2" >Original</td>
      <td id="T_29042_row40_col3" class="data row40 col3" >0.985959</td>
      <td id="T_29042_row40_col4" class="data row40 col4" >3.635338</td>
      <td id="T_29042_row40_col5" class="data row40 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row40_col6" class="data row40 col6" >249.117185</td>
    </tr>
    <tr>
      <td id="T_29042_row41_col0" class="data row41 col0" >XGBClassifier</td>
      <td id="T_29042_row41_col1" class="data row41 col1" >CountVectorizer</td>
      <td id="T_29042_row41_col2" class="data row41 col2" >Stemmed</td>
      <td id="T_29042_row41_col3" class="data row41 col3" >0.985284</td>
      <td id="T_29042_row41_col4" class="data row41 col4" >2.967854</td>
      <td id="T_29042_row41_col5" class="data row41 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row41_col6" class="data row41 col6" >204.539993</td>
    </tr>
    <tr>
      <td id="T_29042_row42_col0" class="data row42 col0" >XGBClassifier</td>
      <td id="T_29042_row42_col1" class="data row42 col1" >CountVectorizer</td>
      <td id="T_29042_row42_col2" class="data row42 col2" >Lemmatized</td>
      <td id="T_29042_row42_col3" class="data row42 col3" >0.985588</td>
      <td id="T_29042_row42_col4" class="data row42 col4" >3.237125</td>
      <td id="T_29042_row42_col5" class="data row42 col5" >{"model__colsample_bytree": 0.7, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row42_col6" class="data row42 col6" >222.565550</td>
    </tr>
    <tr>
      <td id="T_29042_row43_col0" class="data row43 col0" >XGBClassifier</td>
      <td id="T_29042_row43_col1" class="data row43 col1" >CountVectorizer</td>
      <td id="T_29042_row43_col2" class="data row43 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row43_col3" class="data row43 col3" >0.985297</td>
      <td id="T_29042_row43_col4" class="data row43 col4" >2.946874</td>
      <td id="T_29042_row43_col5" class="data row43 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row43_col6" class="data row43 col6" >203.569864</td>
    </tr>
    <tr>
      <td id="T_29042_row44_col0" class="data row44 col0" >XGBClassifier</td>
      <td id="T_29042_row44_col1" class="data row44 col1" >TfidfVectorizer</td>
      <td id="T_29042_row44_col2" class="data row44 col2" >Original</td>
      <td id="T_29042_row44_col3" class="data row44 col3" >0.986995</td>
      <td id="T_29042_row44_col4" class="data row44 col4" >19.613042</td>
      <td id="T_29042_row44_col5" class="data row44 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row44_col6" class="data row44 col6" >1247.920159</td>
    </tr>
    <tr>
      <td id="T_29042_row45_col0" class="data row45 col0" >XGBClassifier</td>
      <td id="T_29042_row45_col1" class="data row45 col1" >TfidfVectorizer</td>
      <td id="T_29042_row45_col2" class="data row45 col2" >Stemmed</td>
      <td id="T_29042_row45_col3" class="data row45 col3" >0.985848</td>
      <td id="T_29042_row45_col4" class="data row45 col4" >16.974539</td>
      <td id="T_29042_row45_col5" class="data row45 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row45_col6" class="data row45 col6" >1083.131984</td>
    </tr>
    <tr>
      <td id="T_29042_row46_col0" class="data row46 col0" >XGBClassifier</td>
      <td id="T_29042_row46_col1" class="data row46 col1" >TfidfVectorizer</td>
      <td id="T_29042_row46_col2" class="data row46 col2" >Lemmatized</td>
      <td id="T_29042_row46_col3" class="data row46 col3" >0.986182</td>
      <td id="T_29042_row46_col4" class="data row46 col4" >17.897716</td>
      <td id="T_29042_row46_col5" class="data row46 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_29042_row46_col6" class="data row46 col6" >1136.221328</td>
    </tr>
    <tr>
      <td id="T_29042_row47_col0" class="data row47 col0" >XGBClassifier</td>
      <td id="T_29042_row47_col1" class="data row47 col1" >TfidfVectorizer</td>
      <td id="T_29042_row47_col2" class="data row47 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row47_col3" class="data row47 col3" >0.985967</td>
      <td id="T_29042_row47_col4" class="data row47 col4" >16.910542</td>
      <td id="T_29042_row47_col5" class="data row47 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 1, "vectorizer__max_features": null}</td>
      <td id="T_29042_row47_col6" class="data row47 col6" >1079.451344</td>
    </tr>
    <tr>
      <td id="T_29042_row48_col0" class="data row48 col0" >BaggingClassifier</td>
      <td id="T_29042_row48_col1" class="data row48 col1" >CountVectorizer</td>
      <td id="T_29042_row48_col2" class="data row48 col2" >Original</td>
      <td id="T_29042_row48_col3" class="data row48 col3" >0.966127</td>
      <td id="T_29042_row48_col4" class="data row48 col4" >43.522997</td>
      <td id="T_29042_row48_col5" class="data row48 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row48_col6" class="data row48 col6" >523.237538</td>
    </tr>
    <tr>
      <td id="T_29042_row49_col0" class="data row49 col0" >BaggingClassifier</td>
      <td id="T_29042_row49_col1" class="data row49 col1" >CountVectorizer</td>
      <td id="T_29042_row49_col2" class="data row49 col2" >Stemmed</td>
      <td id="T_29042_row49_col3" class="data row49 col3" >0.965380</td>
      <td id="T_29042_row49_col4" class="data row49 col4" >38.062631</td>
      <td id="T_29042_row49_col5" class="data row49 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row49_col6" class="data row49 col6" >442.674253</td>
    </tr>
    <tr>
      <td id="T_29042_row50_col0" class="data row50 col0" >BaggingClassifier</td>
      <td id="T_29042_row50_col1" class="data row50 col1" >CountVectorizer</td>
      <td id="T_29042_row50_col2" class="data row50 col2" >Lemmatized</td>
      <td id="T_29042_row50_col3" class="data row50 col3" >0.964045</td>
      <td id="T_29042_row50_col4" class="data row50 col4" >37.804205</td>
      <td id="T_29042_row50_col5" class="data row50 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row50_col6" class="data row50 col6" >449.177688</td>
    </tr>
    <tr>
      <td id="T_29042_row51_col0" class="data row51 col0" >BaggingClassifier</td>
      <td id="T_29042_row51_col1" class="data row51 col1" >CountVectorizer</td>
      <td id="T_29042_row51_col2" class="data row51 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row51_col3" class="data row51 col3" >0.966269</td>
      <td id="T_29042_row51_col4" class="data row51 col4" >38.061219</td>
      <td id="T_29042_row51_col5" class="data row51 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row51_col6" class="data row51 col6" >441.230571</td>
    </tr>
    <tr>
      <td id="T_29042_row52_col0" class="data row52 col0" >BaggingClassifier</td>
      <td id="T_29042_row52_col1" class="data row52 col1" >TfidfVectorizer</td>
      <td id="T_29042_row52_col2" class="data row52 col2" >Original</td>
      <td id="T_29042_row52_col3" class="data row52 col3" >0.970145</td>
      <td id="T_29042_row52_col4" class="data row52 col4" >79.973978</td>
      <td id="T_29042_row52_col5" class="data row52 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row52_col6" class="data row52 col6" >953.591994</td>
    </tr>
    <tr>
      <td id="T_29042_row53_col0" class="data row53 col0" >BaggingClassifier</td>
      <td id="T_29042_row53_col1" class="data row53 col1" >TfidfVectorizer</td>
      <td id="T_29042_row53_col2" class="data row53 col2" >Stemmed</td>
      <td id="T_29042_row53_col3" class="data row53 col3" >0.970279</td>
      <td id="T_29042_row53_col4" class="data row53 col4" >68.076920</td>
      <td id="T_29042_row53_col5" class="data row53 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row53_col6" class="data row53 col6" >785.487087</td>
    </tr>
    <tr>
      <td id="T_29042_row54_col0" class="data row54 col0" >BaggingClassifier</td>
      <td id="T_29042_row54_col1" class="data row54 col1" >TfidfVectorizer</td>
      <td id="T_29042_row54_col2" class="data row54 col2" >Lemmatized</td>
      <td id="T_29042_row54_col3" class="data row54 col3" >0.970210</td>
      <td id="T_29042_row54_col4" class="data row54 col4" >67.954085</td>
      <td id="T_29042_row54_col5" class="data row54 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row54_col6" class="data row54 col6" >804.888307</td>
    </tr>
    <tr>
      <td id="T_29042_row55_col0" class="data row55 col0" >BaggingClassifier</td>
      <td id="T_29042_row55_col1" class="data row55 col1" >TfidfVectorizer</td>
      <td id="T_29042_row55_col2" class="data row55 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row55_col3" class="data row55 col3" >0.970610</td>
      <td id="T_29042_row55_col4" class="data row55 col4" >67.898102</td>
      <td id="T_29042_row55_col5" class="data row55 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row55_col6" class="data row55 col6" >782.555564</td>
    </tr>
    <tr>
      <td id="T_29042_row56_col0" class="data row56 col0" >RandomForestClassifier</td>
      <td id="T_29042_row56_col1" class="data row56 col1" >CountVectorizer</td>
      <td id="T_29042_row56_col2" class="data row56 col2" >Original</td>
      <td id="T_29042_row56_col3" class="data row56 col3" >0.975731</td>
      <td id="T_29042_row56_col4" class="data row56 col4" >2.260491</td>
      <td id="T_29042_row56_col5" class="data row56 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row56_col6" class="data row56 col6" >215.116901</td>
    </tr>
    <tr>
      <td id="T_29042_row57_col0" class="data row57 col0" >RandomForestClassifier</td>
      <td id="T_29042_row57_col1" class="data row57 col1" >CountVectorizer</td>
      <td id="T_29042_row57_col2" class="data row57 col2" >Stemmed</td>
      <td id="T_29042_row57_col3" class="data row57 col3" >0.976997</td>
      <td id="T_29042_row57_col4" class="data row57 col4" >2.130489</td>
      <td id="T_29042_row57_col5" class="data row57 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row57_col6" class="data row57 col6" >202.034706</td>
    </tr>
    <tr>
      <td id="T_29042_row58_col0" class="data row58 col0" >RandomForestClassifier</td>
      <td id="T_29042_row58_col1" class="data row58 col1" >CountVectorizer</td>
      <td id="T_29042_row58_col2" class="data row58 col2" >Lemmatized</td>
      <td id="T_29042_row58_col3" class="data row58 col3" >0.977167</td>
      <td id="T_29042_row58_col4" class="data row58 col4" >2.053881</td>
      <td id="T_29042_row58_col5" class="data row58 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row58_col6" class="data row58 col6" >195.739830</td>
    </tr>
    <tr>
      <td id="T_29042_row59_col0" class="data row59 col0" >RandomForestClassifier</td>
      <td id="T_29042_row59_col1" class="data row59 col1" >CountVectorizer</td>
      <td id="T_29042_row59_col2" class="data row59 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row59_col3" class="data row59 col3" >0.977309</td>
      <td id="T_29042_row59_col4" class="data row59 col4" >2.131661</td>
      <td id="T_29042_row59_col5" class="data row59 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row59_col6" class="data row59 col6" >202.741420</td>
    </tr>
    <tr>
      <td id="T_29042_row60_col0" class="data row60 col0" >RandomForestClassifier</td>
      <td id="T_29042_row60_col1" class="data row60 col1" >TfidfVectorizer</td>
      <td id="T_29042_row60_col2" class="data row60 col2" >Original</td>
      <td id="T_29042_row60_col3" class="data row60 col3" >0.977435</td>
      <td id="T_29042_row60_col4" class="data row60 col4" >2.760853</td>
      <td id="T_29042_row60_col5" class="data row60 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row60_col6" class="data row60 col6" >255.591368</td>
    </tr>
    <tr>
      <td id="T_29042_row61_col0" class="data row61 col0" >RandomForestClassifier</td>
      <td id="T_29042_row61_col1" class="data row61 col1" >TfidfVectorizer</td>
      <td id="T_29042_row61_col2" class="data row61 col2" >Stemmed</td>
      <td id="T_29042_row61_col3" class="data row61 col3" >0.978388</td>
      <td id="T_29042_row61_col4" class="data row61 col4" >2.622887</td>
      <td id="T_29042_row61_col5" class="data row61 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row61_col6" class="data row61 col6" >241.573630</td>
    </tr>
    <tr>
      <td id="T_29042_row62_col0" class="data row62 col0" >RandomForestClassifier</td>
      <td id="T_29042_row62_col1" class="data row62 col1" >TfidfVectorizer</td>
      <td id="T_29042_row62_col2" class="data row62 col2" >Lemmatized</td>
      <td id="T_29042_row62_col3" class="data row62 col3" >0.978780</td>
      <td id="T_29042_row62_col4" class="data row62 col4" >2.519313</td>
      <td id="T_29042_row62_col5" class="data row62 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row62_col6" class="data row62 col6" >233.885730</td>
    </tr>
    <tr>
      <td id="T_29042_row63_col0" class="data row63 col0" >RandomForestClassifier</td>
      <td id="T_29042_row63_col1" class="data row63 col1" >TfidfVectorizer</td>
      <td id="T_29042_row63_col2" class="data row63 col2" >Stemmed and Lemmatized</td>
      <td id="T_29042_row63_col3" class="data row63 col3" >0.979124</td>
      <td id="T_29042_row63_col4" class="data row63 col4" >2.617862</td>
      <td id="T_29042_row63_col5" class="data row63 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_29042_row63_col6" class="data row63 col6" >241.511404</td>
    </tr>
  </tbody>
</table>


### Analysis
<table>
<tr>
<th>ROC AUC score</th>
<th>Fit Time</th>
</tr>
<tr>
<td><a href="./analysis_results/capstone.model_results.roc_auc_ovr.png" target="_blank"><img src="./analysis_results/capstone.model_results.roc_auc_ovr.png"/></a></td>
<td><a href="./analysis_results/capstone.model_results.fit_time.png" target="_blank"><img src="./analysis_results/capstone.model_results.fit_time.png"/></a></td>
</tr>
<tr>
<td><ul>
<li>The best model was:
 XGBClassifier with vectorizer TfidfVectorizer with input pre-processing Original had ROC AUC score 98.69954725563986% and mean fit time of 19.61304181714853 seconds.</li><li>The worst model was:
 DecisionTreeClassifier with vectorizer TfidfVectorizer with input pre-processing Stemmed had ROC AUC score 87.74671564412645% and mean fit time of 1.915424230694771 seconds.</li></ul></td>
<td><ul>
<li>The fastest model was:
 ComplementNB with vectorizer CountVectorizer with input pre-processing Stemmed and Lemmatized had ROC AUC score 94.3771473528919% and mean fit time of 0.2349078516165416 seconds.</li><li>The slowest model was
 LogisticRegression with vectorizer CountVectorizer with input pre-processing Original had ROC AUC score 95.86914032273492% and mean fit time of 90.87342044671377 seconds.</li></ul></td>
</tr>
</table>

