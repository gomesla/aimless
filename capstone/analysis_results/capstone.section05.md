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
#T_0cdf1 th {
  font-size: 8pt;
  font-family: Verdana;
}
#T_0cdf1_row0_col0, #T_0cdf1_row0_col1, #T_0cdf1_row0_col2, #T_0cdf1_row0_col3, #T_0cdf1_row0_col4, #T_0cdf1_row0_col5, #T_0cdf1_row0_col6, #T_0cdf1_row1_col0, #T_0cdf1_row1_col1, #T_0cdf1_row1_col2, #T_0cdf1_row1_col3, #T_0cdf1_row1_col4, #T_0cdf1_row1_col5, #T_0cdf1_row1_col6, #T_0cdf1_row2_col0, #T_0cdf1_row2_col1, #T_0cdf1_row2_col2, #T_0cdf1_row2_col3, #T_0cdf1_row2_col4, #T_0cdf1_row2_col5, #T_0cdf1_row2_col6, #T_0cdf1_row3_col0, #T_0cdf1_row3_col1, #T_0cdf1_row3_col2, #T_0cdf1_row3_col3, #T_0cdf1_row3_col4, #T_0cdf1_row3_col5, #T_0cdf1_row3_col6, #T_0cdf1_row4_col0, #T_0cdf1_row4_col1, #T_0cdf1_row4_col2, #T_0cdf1_row4_col3, #T_0cdf1_row4_col4, #T_0cdf1_row4_col5, #T_0cdf1_row4_col6, #T_0cdf1_row5_col0, #T_0cdf1_row5_col1, #T_0cdf1_row5_col2, #T_0cdf1_row5_col3, #T_0cdf1_row5_col4, #T_0cdf1_row5_col5, #T_0cdf1_row5_col6, #T_0cdf1_row6_col0, #T_0cdf1_row6_col1, #T_0cdf1_row6_col2, #T_0cdf1_row6_col3, #T_0cdf1_row6_col4, #T_0cdf1_row6_col5, #T_0cdf1_row6_col6, #T_0cdf1_row7_col0, #T_0cdf1_row7_col1, #T_0cdf1_row7_col2, #T_0cdf1_row7_col3, #T_0cdf1_row7_col4, #T_0cdf1_row7_col5, #T_0cdf1_row7_col6, #T_0cdf1_row8_col0, #T_0cdf1_row8_col1, #T_0cdf1_row8_col2, #T_0cdf1_row8_col4, #T_0cdf1_row8_col5, #T_0cdf1_row8_col6, #T_0cdf1_row9_col0, #T_0cdf1_row9_col1, #T_0cdf1_row9_col2, #T_0cdf1_row9_col3, #T_0cdf1_row9_col4, #T_0cdf1_row9_col5, #T_0cdf1_row9_col6, #T_0cdf1_row10_col0, #T_0cdf1_row10_col1, #T_0cdf1_row10_col2, #T_0cdf1_row10_col3, #T_0cdf1_row10_col4, #T_0cdf1_row10_col5, #T_0cdf1_row10_col6, #T_0cdf1_row11_col0, #T_0cdf1_row11_col1, #T_0cdf1_row11_col2, #T_0cdf1_row11_col3, #T_0cdf1_row11_col5, #T_0cdf1_row11_col6, #T_0cdf1_row12_col0, #T_0cdf1_row12_col1, #T_0cdf1_row12_col2, #T_0cdf1_row12_col3, #T_0cdf1_row12_col4, #T_0cdf1_row12_col5, #T_0cdf1_row12_col6, #T_0cdf1_row13_col0, #T_0cdf1_row13_col1, #T_0cdf1_row13_col2, #T_0cdf1_row13_col3, #T_0cdf1_row13_col4, #T_0cdf1_row13_col5, #T_0cdf1_row13_col6, #T_0cdf1_row14_col0, #T_0cdf1_row14_col1, #T_0cdf1_row14_col2, #T_0cdf1_row14_col3, #T_0cdf1_row14_col4, #T_0cdf1_row14_col5, #T_0cdf1_row14_col6, #T_0cdf1_row15_col0, #T_0cdf1_row15_col1, #T_0cdf1_row15_col2, #T_0cdf1_row15_col3, #T_0cdf1_row15_col4, #T_0cdf1_row15_col5, #T_0cdf1_row15_col6, #T_0cdf1_row16_col0, #T_0cdf1_row16_col1, #T_0cdf1_row16_col2, #T_0cdf1_row16_col3, #T_0cdf1_row16_col4, #T_0cdf1_row16_col5, #T_0cdf1_row16_col6, #T_0cdf1_row17_col0, #T_0cdf1_row17_col1, #T_0cdf1_row17_col2, #T_0cdf1_row17_col3, #T_0cdf1_row17_col4, #T_0cdf1_row17_col5, #T_0cdf1_row17_col6, #T_0cdf1_row18_col0, #T_0cdf1_row18_col1, #T_0cdf1_row18_col2, #T_0cdf1_row18_col3, #T_0cdf1_row18_col4, #T_0cdf1_row18_col5, #T_0cdf1_row18_col6, #T_0cdf1_row19_col0, #T_0cdf1_row19_col1, #T_0cdf1_row19_col2, #T_0cdf1_row19_col3, #T_0cdf1_row19_col4, #T_0cdf1_row19_col5, #T_0cdf1_row19_col6, #T_0cdf1_row20_col0, #T_0cdf1_row20_col1, #T_0cdf1_row20_col2, #T_0cdf1_row20_col3, #T_0cdf1_row20_col4, #T_0cdf1_row20_col5, #T_0cdf1_row20_col6, #T_0cdf1_row21_col0, #T_0cdf1_row21_col1, #T_0cdf1_row21_col2, #T_0cdf1_row21_col3, #T_0cdf1_row21_col4, #T_0cdf1_row21_col5, #T_0cdf1_row21_col6, #T_0cdf1_row22_col0, #T_0cdf1_row22_col1, #T_0cdf1_row22_col2, #T_0cdf1_row22_col3, #T_0cdf1_row22_col4, #T_0cdf1_row22_col5, #T_0cdf1_row22_col6, #T_0cdf1_row23_col0, #T_0cdf1_row23_col1, #T_0cdf1_row23_col2, #T_0cdf1_row23_col3, #T_0cdf1_row23_col4, #T_0cdf1_row23_col5, #T_0cdf1_row23_col6, #T_0cdf1_row24_col0, #T_0cdf1_row24_col1, #T_0cdf1_row24_col2, #T_0cdf1_row24_col3, #T_0cdf1_row24_col4, #T_0cdf1_row24_col5, #T_0cdf1_row24_col6, #T_0cdf1_row25_col0, #T_0cdf1_row25_col1, #T_0cdf1_row25_col2, #T_0cdf1_row25_col3, #T_0cdf1_row25_col4, #T_0cdf1_row25_col5, #T_0cdf1_row25_col6, #T_0cdf1_row26_col0, #T_0cdf1_row26_col1, #T_0cdf1_row26_col2, #T_0cdf1_row26_col3, #T_0cdf1_row26_col4, #T_0cdf1_row26_col5, #T_0cdf1_row26_col6, #T_0cdf1_row27_col0, #T_0cdf1_row27_col1, #T_0cdf1_row27_col2, #T_0cdf1_row27_col3, #T_0cdf1_row27_col4, #T_0cdf1_row27_col5, #T_0cdf1_row27_col6, #T_0cdf1_row28_col0, #T_0cdf1_row28_col1, #T_0cdf1_row28_col2, #T_0cdf1_row28_col3, #T_0cdf1_row28_col4, #T_0cdf1_row28_col5, #T_0cdf1_row28_col6, #T_0cdf1_row29_col0, #T_0cdf1_row29_col1, #T_0cdf1_row29_col2, #T_0cdf1_row29_col3, #T_0cdf1_row29_col4, #T_0cdf1_row29_col5, #T_0cdf1_row29_col6, #T_0cdf1_row30_col0, #T_0cdf1_row30_col1, #T_0cdf1_row30_col2, #T_0cdf1_row30_col3, #T_0cdf1_row30_col4, #T_0cdf1_row30_col5, #T_0cdf1_row30_col6, #T_0cdf1_row31_col0, #T_0cdf1_row31_col1, #T_0cdf1_row31_col2, #T_0cdf1_row31_col3, #T_0cdf1_row31_col4, #T_0cdf1_row31_col5, #T_0cdf1_row31_col6, #T_0cdf1_row32_col0, #T_0cdf1_row32_col1, #T_0cdf1_row32_col2, #T_0cdf1_row32_col3, #T_0cdf1_row32_col5, #T_0cdf1_row32_col6, #T_0cdf1_row33_col0, #T_0cdf1_row33_col1, #T_0cdf1_row33_col2, #T_0cdf1_row33_col3, #T_0cdf1_row33_col4, #T_0cdf1_row33_col5, #T_0cdf1_row33_col6, #T_0cdf1_row34_col0, #T_0cdf1_row34_col1, #T_0cdf1_row34_col2, #T_0cdf1_row34_col3, #T_0cdf1_row34_col4, #T_0cdf1_row34_col5, #T_0cdf1_row34_col6, #T_0cdf1_row35_col0, #T_0cdf1_row35_col1, #T_0cdf1_row35_col2, #T_0cdf1_row35_col3, #T_0cdf1_row35_col4, #T_0cdf1_row35_col5, #T_0cdf1_row35_col6, #T_0cdf1_row36_col0, #T_0cdf1_row36_col1, #T_0cdf1_row36_col2, #T_0cdf1_row36_col4, #T_0cdf1_row36_col5, #T_0cdf1_row36_col6, #T_0cdf1_row37_col0, #T_0cdf1_row37_col1, #T_0cdf1_row37_col2, #T_0cdf1_row37_col3, #T_0cdf1_row37_col4, #T_0cdf1_row37_col5, #T_0cdf1_row37_col6, #T_0cdf1_row38_col0, #T_0cdf1_row38_col1, #T_0cdf1_row38_col2, #T_0cdf1_row38_col3, #T_0cdf1_row38_col4, #T_0cdf1_row38_col5, #T_0cdf1_row38_col6, #T_0cdf1_row39_col0, #T_0cdf1_row39_col1, #T_0cdf1_row39_col2, #T_0cdf1_row39_col3, #T_0cdf1_row39_col4, #T_0cdf1_row39_col5, #T_0cdf1_row39_col6 {
  font-size: 8pt;
  font-family: Verdana;
}
#T_0cdf1_row8_col3, #T_0cdf1_row32_col4 {
  font-size: 8pt;
  font-family: Verdana;
  font-weight: bold;
  background-color: #FF8A8A;
}
#T_0cdf1_row11_col4, #T_0cdf1_row36_col3 {
  font-size: 8pt;
  font-family: Verdana;
  font-weight: bold;
  background-color: #CCE0AC;
}
</style>
<table id="T_0cdf1">
  <thead>
    <tr>
      <th id="T_0cdf1_level0_col0" class="col_heading level0 col0" >model</th>
      <th id="T_0cdf1_level0_col1" class="col_heading level0 col1" >vectorizer</th>
      <th id="T_0cdf1_level0_col2" class="col_heading level0 col2" >input_field</th>
      <th id="T_0cdf1_level0_col3" class="col_heading level0 col3" >best_score</th>
      <th id="T_0cdf1_level0_col4" class="col_heading level0 col4" >mean_fit_time</th>
      <th id="T_0cdf1_level0_col5" class="col_heading level0 col5" >best_params</th>
      <th id="T_0cdf1_level0_col6" class="col_heading level0 col6" >run_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_0cdf1_row0_col0" class="data row0 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row0_col1" class="data row0 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row0_col2" class="data row0 col2" >Original</td>
      <td id="T_0cdf1_row0_col3" class="data row0 col3" >0.755644</td>
      <td id="T_0cdf1_row0_col4" class="data row0 col4" >1.348733</td>
      <td id="T_0cdf1_row0_col5" class="data row0 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row0_col6" class="data row0 col6" >92.699219</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row1_col0" class="data row1 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row1_col1" class="data row1 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row1_col2" class="data row1 col2" >Stemmed</td>
      <td id="T_0cdf1_row1_col3" class="data row1 col3" >0.755783</td>
      <td id="T_0cdf1_row1_col4" class="data row1 col4" >1.189605</td>
      <td id="T_0cdf1_row1_col5" class="data row1 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row1_col6" class="data row1 col6" >80.458790</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row2_col0" class="data row2 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row2_col1" class="data row2 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row2_col2" class="data row2 col2" >Lemmatized</td>
      <td id="T_0cdf1_row2_col3" class="data row2 col3" >0.755783</td>
      <td id="T_0cdf1_row2_col4" class="data row2 col4" >1.182780</td>
      <td id="T_0cdf1_row2_col5" class="data row2 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row2_col6" class="data row2 col6" >80.108658</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row3_col0" class="data row3 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row3_col1" class="data row3 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row3_col2" class="data row3 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row3_col3" class="data row3 col3" >0.755017</td>
      <td id="T_0cdf1_row3_col4" class="data row3 col4" >1.188148</td>
      <td id="T_0cdf1_row3_col5" class="data row3 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row3_col6" class="data row3 col6" >80.232608</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row4_col0" class="data row4 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row4_col1" class="data row4 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row4_col2" class="data row4 col2" >Original</td>
      <td id="T_0cdf1_row4_col3" class="data row4 col3" >0.760382</td>
      <td id="T_0cdf1_row4_col4" class="data row4 col4" >2.608903</td>
      <td id="T_0cdf1_row4_col5" class="data row4 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row4_col6" class="data row4 col6" >170.182084</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row5_col0" class="data row5 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row5_col1" class="data row5 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row5_col2" class="data row5 col2" >Stemmed</td>
      <td id="T_0cdf1_row5_col3" class="data row5 col3" >0.765329</td>
      <td id="T_0cdf1_row5_col4" class="data row5 col4" >2.227073</td>
      <td id="T_0cdf1_row5_col5" class="data row5 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row5_col6" class="data row5 col6" >145.253776</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row6_col0" class="data row6 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row6_col1" class="data row6 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row6_col2" class="data row6 col2" >Lemmatized</td>
      <td id="T_0cdf1_row6_col3" class="data row6 col3" >0.764911</td>
      <td id="T_0cdf1_row6_col4" class="data row6 col4" >2.233900</td>
      <td id="T_0cdf1_row6_col5" class="data row6 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row6_col6" class="data row6 col6" >145.801716</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row7_col0" class="data row7 col0" >DecisionTreeClassifier</td>
      <td id="T_0cdf1_row7_col1" class="data row7 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row7_col2" class="data row7 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row7_col3" class="data row7 col3" >0.759476</td>
      <td id="T_0cdf1_row7_col4" class="data row7 col4" >2.248466</td>
      <td id="T_0cdf1_row7_col5" class="data row7 col5" >{"model__criterion": "gini", "model__max_depth": 100, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row7_col6" class="data row7 col6" >146.368993</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row8_col0" class="data row8 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row8_col1" class="data row8 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row8_col2" class="data row8 col2" >Original</td>
      <td id="T_0cdf1_row8_col3" class="data row8 col3" >0.707358</td>
      <td id="T_0cdf1_row8_col4" class="data row8 col4" >0.319048</td>
      <td id="T_0cdf1_row8_col5" class="data row8 col5" >{"model__n_neighbors": 10, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row8_col6" class="data row8 col6" >295.930288</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row9_col0" class="data row9 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row9_col1" class="data row9 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row9_col2" class="data row9 col2" >Stemmed</td>
      <td id="T_0cdf1_row9_col3" class="data row9 col3" >0.731954</td>
      <td id="T_0cdf1_row9_col4" class="data row9 col4" >0.270328</td>
      <td id="T_0cdf1_row9_col5" class="data row9 col5" >{"model__n_neighbors": 10, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row9_col6" class="data row9 col6" >355.711935</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row10_col0" class="data row10 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row10_col1" class="data row10 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row10_col2" class="data row10 col2" >Lemmatized</td>
      <td id="T_0cdf1_row10_col3" class="data row10 col3" >0.727076</td>
      <td id="T_0cdf1_row10_col4" class="data row10 col4" >0.277265</td>
      <td id="T_0cdf1_row10_col5" class="data row10 col5" >{"model__n_neighbors": 10, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row10_col6" class="data row10 col6" >277.196101</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row11_col0" class="data row11 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row11_col1" class="data row11 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row11_col2" class="data row11 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row11_col3" class="data row11 col3" >0.732651</td>
      <td id="T_0cdf1_row11_col4" class="data row11 col4" >0.267950</td>
      <td id="T_0cdf1_row11_col5" class="data row11 col5" >{"model__n_neighbors": 10, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row11_col6" class="data row11 col6" >282.346461</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row12_col0" class="data row12 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row12_col1" class="data row12 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row12_col2" class="data row12 col2" >Original</td>
      <td id="T_0cdf1_row12_col3" class="data row12 col3" >0.790831</td>
      <td id="T_0cdf1_row12_col4" class="data row12 col4" >0.375930</td>
      <td id="T_0cdf1_row12_col5" class="data row12 col5" >{"model__n_neighbors": 250, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row12_col6" class="data row12 col6" >440.481341</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row13_col0" class="data row13 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row13_col1" class="data row13 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row13_col2" class="data row13 col2" >Stemmed</td>
      <td id="T_0cdf1_row13_col3" class="data row13 col3" >0.784211</td>
      <td id="T_0cdf1_row13_col4" class="data row13 col4" >0.300361</td>
      <td id="T_0cdf1_row13_col5" class="data row13 col5" >{"model__n_neighbors": 250, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row13_col6" class="data row13 col6" >390.767082</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row14_col0" class="data row14 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row14_col1" class="data row14 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row14_col2" class="data row14 col2" >Lemmatized</td>
      <td id="T_0cdf1_row14_col3" class="data row14 col3" >0.786650</td>
      <td id="T_0cdf1_row14_col4" class="data row14 col4" >0.302879</td>
      <td id="T_0cdf1_row14_col5" class="data row14 col5" >{"model__n_neighbors": 250, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row14_col6" class="data row14 col6" >365.603145</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row15_col0" class="data row15 col0" >KNeighborsClassifier</td>
      <td id="T_0cdf1_row15_col1" class="data row15 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row15_col2" class="data row15 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row15_col3" class="data row15 col3" >0.784351</td>
      <td id="T_0cdf1_row15_col4" class="data row15 col4" >0.285741</td>
      <td id="T_0cdf1_row15_col5" class="data row15 col5" >{"model__n_neighbors": 250, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row15_col6" class="data row15 col6" >323.230748</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row16_col0" class="data row16 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row16_col1" class="data row16 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row16_col2" class="data row16 col2" >Original</td>
      <td id="T_0cdf1_row16_col3" class="data row16 col3" >0.767349</td>
      <td id="T_0cdf1_row16_col4" class="data row16 col4" >0.364887</td>
      <td id="T_0cdf1_row16_col5" class="data row16 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row16_col6" class="data row16 col6" >15.043329</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row17_col0" class="data row17 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row17_col1" class="data row17 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row17_col2" class="data row17 col2" >Stemmed</td>
      <td id="T_0cdf1_row17_col3" class="data row17 col3" >0.763239</td>
      <td id="T_0cdf1_row17_col4" class="data row17 col4" >0.308300</td>
      <td id="T_0cdf1_row17_col5" class="data row17 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row17_col6" class="data row17 col6" >13.657276</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row18_col0" class="data row18 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row18_col1" class="data row18 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row18_col2" class="data row18 col2" >Lemmatized</td>
      <td id="T_0cdf1_row18_col3" class="data row18 col3" >0.769997</td>
      <td id="T_0cdf1_row18_col4" class="data row18 col4" >0.325901</td>
      <td id="T_0cdf1_row18_col5" class="data row18 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row18_col6" class="data row18 col6" >13.634323</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row19_col0" class="data row19 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row19_col1" class="data row19 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row19_col2" class="data row19 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row19_col3" class="data row19 col3" >0.763866</td>
      <td id="T_0cdf1_row19_col4" class="data row19 col4" >0.297423</td>
      <td id="T_0cdf1_row19_col5" class="data row19 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row19_col6" class="data row19 col6" >12.196160</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row20_col0" class="data row20 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row20_col1" class="data row20 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row20_col2" class="data row20 col2" >Original</td>
      <td id="T_0cdf1_row20_col3" class="data row20 col3" >0.772297</td>
      <td id="T_0cdf1_row20_col4" class="data row20 col4" >0.374451</td>
      <td id="T_0cdf1_row20_col5" class="data row20 col5" >{"model__alpha": 0.1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row20_col6" class="data row20 col6" >15.366932</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row21_col0" class="data row21 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row21_col1" class="data row21 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row21_col2" class="data row21 col2" >Stemmed</td>
      <td id="T_0cdf1_row21_col3" class="data row21 col3" >0.769440</td>
      <td id="T_0cdf1_row21_col4" class="data row21 col4" >0.311455</td>
      <td id="T_0cdf1_row21_col5" class="data row21 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row21_col6" class="data row21 col6" >12.726944</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row22_col0" class="data row22 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row22_col1" class="data row22 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row22_col2" class="data row22 col2" >Lemmatized</td>
      <td id="T_0cdf1_row22_col3" class="data row22 col3" >0.772993</td>
      <td id="T_0cdf1_row22_col4" class="data row22 col4" >0.311178</td>
      <td id="T_0cdf1_row22_col5" class="data row22 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row22_col6" class="data row22 col6" >12.730943</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row23_col0" class="data row23 col0" >MultinomialNB</td>
      <td id="T_0cdf1_row23_col1" class="data row23 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row23_col2" class="data row23 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row23_col3" class="data row23 col3" >0.769440</td>
      <td id="T_0cdf1_row23_col4" class="data row23 col4" >0.298116</td>
      <td id="T_0cdf1_row23_col5" class="data row23 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_0cdf1_row23_col6" class="data row23 col6" >12.216650</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row24_col0" class="data row24 col0" >ComplementNB</td>
      <td id="T_0cdf1_row24_col1" class="data row24 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row24_col2" class="data row24 col2" >Original</td>
      <td id="T_0cdf1_row24_col3" class="data row24 col3" >0.784560</td>
      <td id="T_0cdf1_row24_col4" class="data row24 col4" >0.346640</td>
      <td id="T_0cdf1_row24_col5" class="data row24 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row24_col6" class="data row24 col6" >14.137654</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row25_col0" class="data row25 col0" >ComplementNB</td>
      <td id="T_0cdf1_row25_col1" class="data row25 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row25_col2" class="data row25 col2" >Stemmed</td>
      <td id="T_0cdf1_row25_col3" class="data row25 col3" >0.777383</td>
      <td id="T_0cdf1_row25_col4" class="data row25 col4" >0.336892</td>
      <td id="T_0cdf1_row25_col5" class="data row25 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row25_col6" class="data row25 col6" >13.698543</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row26_col0" class="data row26 col0" >ComplementNB</td>
      <td id="T_0cdf1_row26_col1" class="data row26 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row26_col2" class="data row26 col2" >Lemmatized</td>
      <td id="T_0cdf1_row26_col3" class="data row26 col3" >0.781006</td>
      <td id="T_0cdf1_row26_col4" class="data row26 col4" >0.323680</td>
      <td id="T_0cdf1_row26_col5" class="data row26 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row26_col6" class="data row26 col6" >13.424998</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row27_col0" class="data row27 col0" >ComplementNB</td>
      <td id="T_0cdf1_row27_col1" class="data row27 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row27_col2" class="data row27 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row27_col3" class="data row27 col3" >0.776547</td>
      <td id="T_0cdf1_row27_col4" class="data row27 col4" >0.349950</td>
      <td id="T_0cdf1_row27_col5" class="data row27 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row27_col6" class="data row27 col6" >14.284199</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row28_col0" class="data row28 col0" >ComplementNB</td>
      <td id="T_0cdf1_row28_col1" class="data row28 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row28_col2" class="data row28 col2" >Original</td>
      <td id="T_0cdf1_row28_col3" class="data row28 col3" >0.789785</td>
      <td id="T_0cdf1_row28_col4" class="data row28 col4" >0.404248</td>
      <td id="T_0cdf1_row28_col5" class="data row28 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row28_col6" class="data row28 col6" >16.728385</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row29_col0" class="data row29 col0" >ComplementNB</td>
      <td id="T_0cdf1_row29_col1" class="data row29 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row29_col2" class="data row29 col2" >Stemmed</td>
      <td id="T_0cdf1_row29_col3" class="data row29 col3" >0.779125</td>
      <td id="T_0cdf1_row29_col4" class="data row29 col4" >0.350548</td>
      <td id="T_0cdf1_row29_col5" class="data row29 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row29_col6" class="data row29 col6" >14.407861</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row30_col0" class="data row30 col0" >ComplementNB</td>
      <td id="T_0cdf1_row30_col1" class="data row30 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row30_col2" class="data row30 col2" >Lemmatized</td>
      <td id="T_0cdf1_row30_col3" class="data row30 col3" >0.787347</td>
      <td id="T_0cdf1_row30_col4" class="data row30 col4" >0.319173</td>
      <td id="T_0cdf1_row30_col5" class="data row30 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row30_col6" class="data row30 col6" >13.073108</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row31_col0" class="data row31 col0" >ComplementNB</td>
      <td id="T_0cdf1_row31_col1" class="data row31 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row31_col2" class="data row31 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row31_col3" class="data row31 col3" >0.779125</td>
      <td id="T_0cdf1_row31_col4" class="data row31 col4" >0.314119</td>
      <td id="T_0cdf1_row31_col5" class="data row31 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row31_col6" class="data row31 col6" >12.948845</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row32_col0" class="data row32 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row32_col1" class="data row32 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row32_col2" class="data row32 col2" >Original</td>
      <td id="T_0cdf1_row32_col3" class="data row32 col3" >0.786998</td>
      <td id="T_0cdf1_row32_col4" class="data row32 col4" >121.754986</td>
      <td id="T_0cdf1_row32_col5" class="data row32 col5" >{"model__C": 10, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row32_col6" class="data row32 col6" >18838.972722</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row33_col0" class="data row33 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row33_col1" class="data row33 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row33_col2" class="data row33 col2" >Stemmed</td>
      <td id="T_0cdf1_row33_col3" class="data row33 col3" >0.789228</td>
      <td id="T_0cdf1_row33_col4" class="data row33 col4" >71.073977</td>
      <td id="T_0cdf1_row33_col5" class="data row33 col5" >{"model__C": 10, "model__l1_ratio": 0.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row33_col6" class="data row33 col6" >10933.584777</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row34_col0" class="data row34 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row34_col1" class="data row34 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row34_col2" class="data row34 col2" >Lemmatized</td>
      <td id="T_0cdf1_row34_col3" class="data row34 col3" >0.788740</td>
      <td id="T_0cdf1_row34_col4" class="data row34 col4" >97.957483</td>
      <td id="T_0cdf1_row34_col5" class="data row34 col5" >{"model__C": 10, "model__l1_ratio": 0.5, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row34_col6" class="data row34 col6" >16368.772253</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row35_col0" class="data row35 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row35_col1" class="data row35 col1" >CountVectorizer</td>
      <td id="T_0cdf1_row35_col2" class="data row35 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row35_col3" class="data row35 col3" >0.789298</td>
      <td id="T_0cdf1_row35_col4" class="data row35 col4" >70.680310</td>
      <td id="T_0cdf1_row35_col5" class="data row35 col5" >{"model__C": 100, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row35_col6" class="data row35 col6" >11806.473677</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row36_col0" class="data row36 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row36_col1" class="data row36 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row36_col2" class="data row36 col2" >Original</td>
      <td id="T_0cdf1_row36_col3" class="data row36 col3" >0.864339</td>
      <td id="T_0cdf1_row36_col4" class="data row36 col4" >30.583491</td>
      <td id="T_0cdf1_row36_col5" class="data row36 col5" >{"model__C": 1, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row36_col6" class="data row36 col6" >4798.800576</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row37_col0" class="data row37 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row37_col1" class="data row37 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row37_col2" class="data row37 col2" >Stemmed</td>
      <td id="T_0cdf1_row37_col3" class="data row37 col3" >0.845805</td>
      <td id="T_0cdf1_row37_col4" class="data row37 col4" >22.991875</td>
      <td id="T_0cdf1_row37_col5" class="data row37 col5" >{"model__C": 1, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row37_col6" class="data row37 col6" >3579.840115</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row38_col0" class="data row38 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row38_col1" class="data row38 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row38_col2" class="data row38 col2" >Lemmatized</td>
      <td id="T_0cdf1_row38_col3" class="data row38 col3" >0.849080</td>
      <td id="T_0cdf1_row38_col4" class="data row38 col4" >40.536143</td>
      <td id="T_0cdf1_row38_col5" class="data row38 col5" >{"model__C": 1, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row38_col6" class="data row38 col6" >4464.918524</td>
    </tr>
    <tr>
      <td id="T_0cdf1_row39_col0" class="data row39 col0" >LogisticRegression</td>
      <td id="T_0cdf1_row39_col1" class="data row39 col1" >TfidfVectorizer</td>
      <td id="T_0cdf1_row39_col2" class="data row39 col2" >Stemmed and Lemmatized</td>
      <td id="T_0cdf1_row39_col3" class="data row39 col3" >0.846084</td>
      <td id="T_0cdf1_row39_col4" class="data row39 col4" >23.099856</td>
      <td id="T_0cdf1_row39_col5" class="data row39 col5" >{"model__C": 1, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_0cdf1_row39_col6" class="data row39 col6" >3597.431579</td>
    </tr>
  </tbody>
</table>


### Analysis
<table>
<tr>
<th>Accuracy</th>
<th>Fit Time</th>
</tr>
<tr>
<td><a href="./analysis_results/capstone.model_results.accuracy.png" target="_blank"><img src="./analysis_results/capstone.model_results.accuracy.png"/></a></td>
<td><a href="./analysis_results/capstone.model_results.fit_time.png" target="_blank"><img src="./analysis_results/capstone.model_results.fit_time.png"/></a></td>
</tr>
<tr>
<td><ul>
<li>The best model was:
 LogisticRegression with vectorizer TfidfVectorizer with input pre-processing Original had accuracy score 86.43394648829431% and mean fit time of 30.58349098086357 seconds.</li><li>The worst model was:
 KNeighborsClassifier with vectorizer CountVectorizer with input pre-processing Original had accuracy score 70.73578595317726% and mean fit time of 0.3190477561950684 seconds.</li></ul></td>
<td><ul>
<li>The fastest model was:
 KNeighborsClassifier with vectorizer CountVectorizer with input pre-processing Stemmed and Lemmatized had accuracy score 73.26505016722408% and mean fit time of 0.2679497766494751 seconds.</li><li>The slowest model was
 LogisticRegression with vectorizer CountVectorizer with input pre-processing Original had accuracy score 78.69983277591973% and mean fit time of 121.7549864761034 seconds.</li></ul></td>
</tr>
</table>

