# Capstone: IT Service Ticket Classification
<b color="red">NOTE: All images are clickable and higher resolution images will load in new browser window</b>
## Background
Most companies have a service desk to help their user or customers with their IT problems.
Customers log and track their ticket in some digital platform and these get routed to Agents to solve.
Routing these tickets to the correct person/team to solve is very important:
- A common approach is to "tag" the ticket with some classification and there is a mapping that routes based on the tag to the correct team to solve it.
- Tickets incorrectly classified result in them "bouncing around" and wasting both agents and customer time.
- There is a financial impact resulting in wasted internal resources and lower customer satisfaction costing the company in real world dollars as well as reputation.
- The scale of the problem grows as you service more users as you need more and more agents to help.


This project aims to address this task by finding and building a model to aid in the classification of these tickets into appropriate categories. We will experiment with various types of pre-processing, vectorizers and regression models to help us find the best one to solve for this.

**Dataset**: [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
- This dataset contains 47,837 rows of data and 2 features.

We will follow the CRISP-DM process model consisting of Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment stages.


## Research Code
**Code:** [Data Analysis Workbook](./capstone.out.ipynb)

**NOTE:** *The processing of the juypter notebook take a long time and often disconnects from the session. In order to run it without monitoring it all the time use the wokraround below from command line.*
```
jupyter nbconvert --to notebook --execute {input_notebook} --output={output_notebook} --ExecutePreprocessor.timeout=-1
```
<sub>Source: [screen-and-jupyter-a-way-to-run-long-notebooks-headles](https://www.maksimeren.com/post/screen-and-jupyter-a-way-to-run-long-notebooks-headless/)</sub>



## Business Understanding
Our goal is to:
- Come up with the best classification model to correctly classify the tickets based solely on the description.
- Provide the business with the best model and hyperparameters to drive the best ROC AUC score.
  - We choose ROC AUC score because:
    - We are doing classification and it is particulary subject to the [Accuracy Paradox](https://en.wikipedia.org/wiki/Accuracy_paradox) because of class imbalance in this set.
    - It is generally preferred for imbalanced data. In addition we should make sure to use OVR during grid search for this (imbalance) reason.
- Provide the business with alternative models trading ROC AUC score for performance in case that may be a concern or to scale for a much larger dataset that may need re-training.


## Data Understanding
### Data Shape
### Raw

<table><tr><th>info()</th><tr><td><pre><class 'pandas.core.frame.DataFrame'>
RangeIndex: 47837 entries, 0 to 47836
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   Original  47837 non-null  object
 1   target    47837 non-null  object
dtypes: object(2)
memory usage: 747.6+ KB
</pre></td></tr></table>

### Data Sample
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Original</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>permission to create new projects needed tuesday october pm permission create needed hello please log assign directly queue has done thanks engineer</td>
      <td>Access</td>
    </tr>
    <tr>
      <td>please enable for on account thursday july pm please enable please reopen provide solution users converted reference needs doing after created please let disable jabber thank thursday july pm re please enable hi update misunderstanding he become user please create him notify delete jabber re please enable hi had chat he wanted explained him problem wait everybody please enable hello guys possible thank wednesday july re please enable dears date july pm please enable hello please advise he requested extension proceed entity allocate extension him looking forward hearing thanks kind analyst ext</td>
      <td>Access</td>
    </tr>
    <tr>
      <td>outlook and network ticket be issued regarding pc since end last week resolved thank you best regards senior software developer</td>
      <td>Administrative rights</td>
    </tr>
    <tr>
      <td>update the right info nexus wednesday pm update right nexus hi figure out weird thing nexus nexus nexus content repositories releases found module has version greater tagged image version tag stable version nexus has greater version which bad version greater than wrong please help delete wrong version nexus wrong version created by fail job run created bad versions thank nexus nexus module image design lead</td>
      <td>Administrative rights</td>
    </tr>
    <tr>
      <td>pas access july pm hello maternity leave years back kind provide thank</td>
      <td>HR Support</td>
    </tr>
    <tr>
      <td>new starter thursday july starter form hi please fill thank regards</td>
      <td>HR Support</td>
    </tr>
    <tr>
      <td>laptop audio problem laptop recognize headphones please assign ticket</td>
      <td>Hardware</td>
    </tr>
    <tr>
      <td>cable cat cat va pc</td>
      <td>Hardware</td>
    </tr>
    <tr>
      <td>us oracle codes codes hi please codes want every code other countries applicable thank best regards specialist nj id image upcoming out dates october rd</td>
      <td>Internal Project</td>
    </tr>
    <tr>
      <td>new project codes spare re codes spare hi please code spare manage chart accounts values manage values spare redesign template thanks senior accountant ext</td>
      <td>Internal Project</td>
    </tr>
    <tr>
      <td>opportunity sent friday march re opportunity thanks sent friday march re opportunity hi sorry for everything sorted out contracts opportunities properly linked could you please have look confirm hi could you please log ticket for below thank you kind regards applications analyst en sent re opportunity hi able contract level opportunity id add if going opportunity level things looks fine would ask for your link also another opportunity opportunity id with below contract contract id thanks much sent friday march re opportunity hi we have successfully linked mentioned opportunity provided contract could you please check let know if thank you kind regards applications analyst en sent opportunity hi can you please help link below opportunity opportunity id add with below contract contract id thanks upcoming holiday th th th floor blvd district</td>
      <td>Miscellaneous</td>
    </tr>
    <tr>
      <td>value field for spent report spent report hi guys please kindly add spent report hi please kindly log thanks thanks kind regards</td>
      <td>Miscellaneous</td>
    </tr>
    <tr>
      <td>new purchase po purchase po dear purchased scanner symbol cs sr link brother flexible tape roll please log allocation please call regarding scanner kind regards administrator</td>
      <td>Purchase</td>
    </tr>
    <tr>
      <td>reception new delivery hello received items po po po po po po please advise please log thank engineer</td>
      <td>Purchase</td>
    </tr>
    <tr>
      <td>joins project joins hi guys joined please him shared folder thanks</td>
      <td>Storage</td>
    </tr>
    <tr>
      <td>wants to share with wednesday pm wants share want share accept decline</td>
      <td>Storage</td>
    </tr>
  </tbody>
</table>

### Features
- There is only one feature Original which is free form text
- There are no missing values
- All data appears to be in english
- All data appears to be lowercased
- The data set is somewhat large and the classes are imbalanced.
- The classification field target has 8 distinct values
<a href="./analysis_results/capstone.raw.targetField.distribution.png" target="_blank"><img src="./analysis_results/capstone.raw.targetField.distribution.png"/></a>



## Data Preparation
### Pre-Processing Decisions
- We will need to:
  - lowercase the data
  - remove stop words
  - stem, lemmatize or combination of both
  - certain models only expect integer target classes (e.g. XGBoost) so we will LabelEncode and create a new encoded column for this.
    - Access = 0
    - Administrative rights = 1
    - HR Support = 2
    - Hardware = 3
    - Internal Project = 4
    - Miscellaneous = 5
    - Purchase = 6
    - Storage = 7
- We will create three columns for variations of stem, lemmatize, stem+lemmatize and then run through models to evaluate which is best
  - I am opting to do this here because while normally we would bake this into the Pipeline since we are experimenting I want to optmize the process and just process these once instead of each time we switch the model during cross validation stage.<a href="./analysis_results/capstone.pre_process.token_count_distribution.png" target="_blank"><img src="./analysis_results/capstone.pre_process.token_count_distribution.png"/></a>

### Analysis
- It is interesting to note both looking at the data and the distribution plot of tokens which doesn't appear to shift much after pre-processing that the data has been through some level of pre-processing already.
- Once pre-processing is done a majority of the tokens fall below 200 we can use this to set the upper bound for some of the model hyperparameters.


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
#T_698e5 th {
  font-size: 8pt;
  font-family: Verdana;
}
#T_698e5_row0_col0, #T_698e5_row0_col1, #T_698e5_row0_col2, #T_698e5_row0_col3, #T_698e5_row0_col4, #T_698e5_row0_col5, #T_698e5_row0_col6, #T_698e5_row1_col0, #T_698e5_row1_col1, #T_698e5_row1_col2, #T_698e5_row1_col3, #T_698e5_row1_col4, #T_698e5_row1_col5, #T_698e5_row1_col6, #T_698e5_row2_col0, #T_698e5_row2_col1, #T_698e5_row2_col2, #T_698e5_row2_col3, #T_698e5_row2_col4, #T_698e5_row2_col5, #T_698e5_row2_col6, #T_698e5_row3_col0, #T_698e5_row3_col1, #T_698e5_row3_col2, #T_698e5_row3_col3, #T_698e5_row3_col4, #T_698e5_row3_col5, #T_698e5_row3_col6, #T_698e5_row4_col0, #T_698e5_row4_col1, #T_698e5_row4_col2, #T_698e5_row4_col3, #T_698e5_row4_col4, #T_698e5_row4_col5, #T_698e5_row4_col6, #T_698e5_row5_col0, #T_698e5_row5_col1, #T_698e5_row5_col2, #T_698e5_row5_col4, #T_698e5_row5_col5, #T_698e5_row5_col6, #T_698e5_row6_col0, #T_698e5_row6_col1, #T_698e5_row6_col2, #T_698e5_row6_col3, #T_698e5_row6_col4, #T_698e5_row6_col5, #T_698e5_row6_col6, #T_698e5_row7_col0, #T_698e5_row7_col1, #T_698e5_row7_col2, #T_698e5_row7_col3, #T_698e5_row7_col4, #T_698e5_row7_col5, #T_698e5_row7_col6, #T_698e5_row8_col0, #T_698e5_row8_col1, #T_698e5_row8_col2, #T_698e5_row8_col3, #T_698e5_row8_col4, #T_698e5_row8_col5, #T_698e5_row8_col6, #T_698e5_row9_col0, #T_698e5_row9_col1, #T_698e5_row9_col2, #T_698e5_row9_col3, #T_698e5_row9_col4, #T_698e5_row9_col5, #T_698e5_row9_col6, #T_698e5_row10_col0, #T_698e5_row10_col1, #T_698e5_row10_col2, #T_698e5_row10_col3, #T_698e5_row10_col4, #T_698e5_row10_col5, #T_698e5_row10_col6, #T_698e5_row11_col0, #T_698e5_row11_col1, #T_698e5_row11_col2, #T_698e5_row11_col3, #T_698e5_row11_col4, #T_698e5_row11_col5, #T_698e5_row11_col6, #T_698e5_row12_col0, #T_698e5_row12_col1, #T_698e5_row12_col2, #T_698e5_row12_col3, #T_698e5_row12_col4, #T_698e5_row12_col5, #T_698e5_row12_col6, #T_698e5_row13_col0, #T_698e5_row13_col1, #T_698e5_row13_col2, #T_698e5_row13_col3, #T_698e5_row13_col4, #T_698e5_row13_col5, #T_698e5_row13_col6, #T_698e5_row14_col0, #T_698e5_row14_col1, #T_698e5_row14_col2, #T_698e5_row14_col3, #T_698e5_row14_col4, #T_698e5_row14_col5, #T_698e5_row14_col6, #T_698e5_row15_col0, #T_698e5_row15_col1, #T_698e5_row15_col2, #T_698e5_row15_col3, #T_698e5_row15_col4, #T_698e5_row15_col5, #T_698e5_row15_col6, #T_698e5_row16_col0, #T_698e5_row16_col1, #T_698e5_row16_col2, #T_698e5_row16_col3, #T_698e5_row16_col4, #T_698e5_row16_col5, #T_698e5_row16_col6, #T_698e5_row17_col0, #T_698e5_row17_col1, #T_698e5_row17_col2, #T_698e5_row17_col3, #T_698e5_row17_col4, #T_698e5_row17_col5, #T_698e5_row17_col6, #T_698e5_row18_col0, #T_698e5_row18_col1, #T_698e5_row18_col2, #T_698e5_row18_col3, #T_698e5_row18_col4, #T_698e5_row18_col5, #T_698e5_row18_col6, #T_698e5_row19_col0, #T_698e5_row19_col1, #T_698e5_row19_col2, #T_698e5_row19_col3, #T_698e5_row19_col4, #T_698e5_row19_col5, #T_698e5_row19_col6, #T_698e5_row20_col0, #T_698e5_row20_col1, #T_698e5_row20_col2, #T_698e5_row20_col3, #T_698e5_row20_col4, #T_698e5_row20_col5, #T_698e5_row20_col6, #T_698e5_row21_col0, #T_698e5_row21_col1, #T_698e5_row21_col2, #T_698e5_row21_col3, #T_698e5_row21_col4, #T_698e5_row21_col5, #T_698e5_row21_col6, #T_698e5_row22_col0, #T_698e5_row22_col1, #T_698e5_row22_col2, #T_698e5_row22_col3, #T_698e5_row22_col4, #T_698e5_row22_col5, #T_698e5_row22_col6, #T_698e5_row23_col0, #T_698e5_row23_col1, #T_698e5_row23_col2, #T_698e5_row23_col3, #T_698e5_row23_col4, #T_698e5_row23_col5, #T_698e5_row23_col6, #T_698e5_row24_col0, #T_698e5_row24_col1, #T_698e5_row24_col2, #T_698e5_row24_col3, #T_698e5_row24_col4, #T_698e5_row24_col5, #T_698e5_row24_col6, #T_698e5_row25_col0, #T_698e5_row25_col1, #T_698e5_row25_col2, #T_698e5_row25_col3, #T_698e5_row25_col4, #T_698e5_row25_col5, #T_698e5_row25_col6, #T_698e5_row26_col0, #T_698e5_row26_col1, #T_698e5_row26_col2, #T_698e5_row26_col3, #T_698e5_row26_col4, #T_698e5_row26_col5, #T_698e5_row26_col6, #T_698e5_row27_col0, #T_698e5_row27_col1, #T_698e5_row27_col2, #T_698e5_row27_col3, #T_698e5_row27_col5, #T_698e5_row27_col6, #T_698e5_row28_col0, #T_698e5_row28_col1, #T_698e5_row28_col2, #T_698e5_row28_col3, #T_698e5_row28_col4, #T_698e5_row28_col5, #T_698e5_row28_col6, #T_698e5_row29_col0, #T_698e5_row29_col1, #T_698e5_row29_col2, #T_698e5_row29_col3, #T_698e5_row29_col4, #T_698e5_row29_col5, #T_698e5_row29_col6, #T_698e5_row30_col0, #T_698e5_row30_col1, #T_698e5_row30_col2, #T_698e5_row30_col3, #T_698e5_row30_col4, #T_698e5_row30_col5, #T_698e5_row30_col6, #T_698e5_row31_col0, #T_698e5_row31_col1, #T_698e5_row31_col2, #T_698e5_row31_col3, #T_698e5_row31_col4, #T_698e5_row31_col5, #T_698e5_row31_col6, #T_698e5_row32_col0, #T_698e5_row32_col1, #T_698e5_row32_col2, #T_698e5_row32_col3, #T_698e5_row32_col4, #T_698e5_row32_col5, #T_698e5_row32_col6, #T_698e5_row33_col0, #T_698e5_row33_col1, #T_698e5_row33_col2, #T_698e5_row33_col3, #T_698e5_row33_col4, #T_698e5_row33_col5, #T_698e5_row33_col6, #T_698e5_row34_col0, #T_698e5_row34_col1, #T_698e5_row34_col2, #T_698e5_row34_col3, #T_698e5_row34_col4, #T_698e5_row34_col5, #T_698e5_row34_col6, #T_698e5_row35_col0, #T_698e5_row35_col1, #T_698e5_row35_col2, #T_698e5_row35_col3, #T_698e5_row35_col4, #T_698e5_row35_col5, #T_698e5_row35_col6, #T_698e5_row36_col0, #T_698e5_row36_col1, #T_698e5_row36_col2, #T_698e5_row36_col3, #T_698e5_row36_col4, #T_698e5_row36_col5, #T_698e5_row36_col6, #T_698e5_row37_col0, #T_698e5_row37_col1, #T_698e5_row37_col2, #T_698e5_row37_col3, #T_698e5_row37_col4, #T_698e5_row37_col5, #T_698e5_row37_col6, #T_698e5_row38_col0, #T_698e5_row38_col1, #T_698e5_row38_col2, #T_698e5_row38_col3, #T_698e5_row38_col4, #T_698e5_row38_col5, #T_698e5_row38_col6, #T_698e5_row39_col0, #T_698e5_row39_col1, #T_698e5_row39_col2, #T_698e5_row39_col3, #T_698e5_row39_col4, #T_698e5_row39_col5, #T_698e5_row39_col6, #T_698e5_row40_col0, #T_698e5_row40_col1, #T_698e5_row40_col2, #T_698e5_row40_col3, #T_698e5_row40_col4, #T_698e5_row40_col5, #T_698e5_row40_col6, #T_698e5_row41_col0, #T_698e5_row41_col1, #T_698e5_row41_col2, #T_698e5_row41_col3, #T_698e5_row41_col4, #T_698e5_row41_col5, #T_698e5_row41_col6, #T_698e5_row42_col0, #T_698e5_row42_col1, #T_698e5_row42_col2, #T_698e5_row42_col3, #T_698e5_row42_col4, #T_698e5_row42_col5, #T_698e5_row42_col6, #T_698e5_row43_col0, #T_698e5_row43_col1, #T_698e5_row43_col2, #T_698e5_row43_col3, #T_698e5_row43_col4, #T_698e5_row43_col5, #T_698e5_row43_col6, #T_698e5_row44_col0, #T_698e5_row44_col1, #T_698e5_row44_col2, #T_698e5_row44_col4, #T_698e5_row44_col5, #T_698e5_row44_col6, #T_698e5_row45_col0, #T_698e5_row45_col1, #T_698e5_row45_col2, #T_698e5_row45_col3, #T_698e5_row45_col4, #T_698e5_row45_col5, #T_698e5_row45_col6, #T_698e5_row46_col0, #T_698e5_row46_col1, #T_698e5_row46_col2, #T_698e5_row46_col3, #T_698e5_row46_col4, #T_698e5_row46_col5, #T_698e5_row46_col6, #T_698e5_row47_col0, #T_698e5_row47_col1, #T_698e5_row47_col2, #T_698e5_row47_col3, #T_698e5_row47_col4, #T_698e5_row47_col5, #T_698e5_row47_col6, #T_698e5_row48_col0, #T_698e5_row48_col1, #T_698e5_row48_col2, #T_698e5_row48_col3, #T_698e5_row48_col4, #T_698e5_row48_col5, #T_698e5_row48_col6, #T_698e5_row49_col0, #T_698e5_row49_col1, #T_698e5_row49_col2, #T_698e5_row49_col3, #T_698e5_row49_col4, #T_698e5_row49_col5, #T_698e5_row49_col6, #T_698e5_row50_col0, #T_698e5_row50_col1, #T_698e5_row50_col2, #T_698e5_row50_col3, #T_698e5_row50_col4, #T_698e5_row50_col5, #T_698e5_row50_col6, #T_698e5_row51_col0, #T_698e5_row51_col1, #T_698e5_row51_col2, #T_698e5_row51_col3, #T_698e5_row51_col4, #T_698e5_row51_col5, #T_698e5_row51_col6, #T_698e5_row52_col0, #T_698e5_row52_col1, #T_698e5_row52_col2, #T_698e5_row52_col3, #T_698e5_row52_col5, #T_698e5_row52_col6, #T_698e5_row53_col0, #T_698e5_row53_col1, #T_698e5_row53_col2, #T_698e5_row53_col3, #T_698e5_row53_col4, #T_698e5_row53_col5, #T_698e5_row53_col6, #T_698e5_row54_col0, #T_698e5_row54_col1, #T_698e5_row54_col2, #T_698e5_row54_col3, #T_698e5_row54_col4, #T_698e5_row54_col5, #T_698e5_row54_col6, #T_698e5_row55_col0, #T_698e5_row55_col1, #T_698e5_row55_col2, #T_698e5_row55_col3, #T_698e5_row55_col4, #T_698e5_row55_col5, #T_698e5_row55_col6, #T_698e5_row56_col0, #T_698e5_row56_col1, #T_698e5_row56_col2, #T_698e5_row56_col3, #T_698e5_row56_col4, #T_698e5_row56_col5, #T_698e5_row56_col6, #T_698e5_row57_col0, #T_698e5_row57_col1, #T_698e5_row57_col2, #T_698e5_row57_col3, #T_698e5_row57_col4, #T_698e5_row57_col5, #T_698e5_row57_col6, #T_698e5_row58_col0, #T_698e5_row58_col1, #T_698e5_row58_col2, #T_698e5_row58_col3, #T_698e5_row58_col4, #T_698e5_row58_col5, #T_698e5_row58_col6, #T_698e5_row59_col0, #T_698e5_row59_col1, #T_698e5_row59_col2, #T_698e5_row59_col3, #T_698e5_row59_col4, #T_698e5_row59_col5, #T_698e5_row59_col6, #T_698e5_row60_col0, #T_698e5_row60_col1, #T_698e5_row60_col2, #T_698e5_row60_col3, #T_698e5_row60_col4, #T_698e5_row60_col5, #T_698e5_row60_col6, #T_698e5_row61_col0, #T_698e5_row61_col1, #T_698e5_row61_col2, #T_698e5_row61_col3, #T_698e5_row61_col4, #T_698e5_row61_col5, #T_698e5_row61_col6, #T_698e5_row62_col0, #T_698e5_row62_col1, #T_698e5_row62_col2, #T_698e5_row62_col3, #T_698e5_row62_col4, #T_698e5_row62_col5, #T_698e5_row62_col6, #T_698e5_row63_col0, #T_698e5_row63_col1, #T_698e5_row63_col2, #T_698e5_row63_col3, #T_698e5_row63_col4, #T_698e5_row63_col5, #T_698e5_row63_col6 {
  font-size: 8pt;
  font-family: Verdana;
}
#T_698e5_row5_col3, #T_698e5_row52_col4 {
  font-size: 8pt;
  font-family: Verdana;
  font-weight: bold;
  background-color: #FF8A8A;
}
#T_698e5_row27_col4, #T_698e5_row44_col3 {
  font-size: 8pt;
  font-family: Verdana;
  font-weight: bold;
  background-color: #CCE0AC;
}
</style>
<table id="T_698e5">
  <thead>
    <tr>
      <th id="T_698e5_level0_col0" class="col_heading level0 col0" >model</th>
      <th id="T_698e5_level0_col1" class="col_heading level0 col1" >vectorizer</th>
      <th id="T_698e5_level0_col2" class="col_heading level0 col2" >input_field</th>
      <th id="T_698e5_level0_col3" class="col_heading level0 col3" >best_score</th>
      <th id="T_698e5_level0_col4" class="col_heading level0 col4" >mean_fit_time</th>
      <th id="T_698e5_level0_col5" class="col_heading level0 col5" >best_params</th>
      <th id="T_698e5_level0_col6" class="col_heading level0 col6" >run_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_698e5_row0_col0" class="data row0 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row0_col1" class="data row0 col1" >CountVectorizer</td>
      <td id="T_698e5_row0_col2" class="data row0 col2" >Original</td>
      <td id="T_698e5_row0_col3" class="data row0 col3" >0.879045</td>
      <td id="T_698e5_row0_col4" class="data row0 col4" >1.194810</td>
      <td id="T_698e5_row0_col5" class="data row0 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row0_col6" class="data row0 col6" >29.435410</td>
    </tr>
    <tr>
      <td id="T_698e5_row1_col0" class="data row1 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row1_col1" class="data row1 col1" >CountVectorizer</td>
      <td id="T_698e5_row1_col2" class="data row1 col2" >Stemmed</td>
      <td id="T_698e5_row1_col3" class="data row1 col3" >0.885074</td>
      <td id="T_698e5_row1_col4" class="data row1 col4" >1.063429</td>
      <td id="T_698e5_row1_col5" class="data row1 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row1_col6" class="data row1 col6" >25.612033</td>
    </tr>
    <tr>
      <td id="T_698e5_row2_col0" class="data row2 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row2_col1" class="data row2 col1" >CountVectorizer</td>
      <td id="T_698e5_row2_col2" class="data row2 col2" >Lemmatized</td>
      <td id="T_698e5_row2_col3" class="data row2 col3" >0.885321</td>
      <td id="T_698e5_row2_col4" class="data row2 col4" >1.042587</td>
      <td id="T_698e5_row2_col5" class="data row2 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row2_col6" class="data row2 col6" >25.303553</td>
    </tr>
    <tr>
      <td id="T_698e5_row3_col0" class="data row3 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row3_col1" class="data row3 col1" >CountVectorizer</td>
      <td id="T_698e5_row3_col2" class="data row3 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row3_col3" class="data row3 col3" >0.884301</td>
      <td id="T_698e5_row3_col4" class="data row3 col4" >1.053470</td>
      <td id="T_698e5_row3_col5" class="data row3 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row3_col6" class="data row3 col6" >25.305413</td>
    </tr>
    <tr>
      <td id="T_698e5_row4_col0" class="data row4 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row4_col1" class="data row4 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row4_col2" class="data row4 col2" >Original</td>
      <td id="T_698e5_row4_col3" class="data row4 col3" >0.880319</td>
      <td id="T_698e5_row4_col4" class="data row4 col4" >2.206945</td>
      <td id="T_698e5_row4_col5" class="data row4 col5" >{"model__criterion": "gini", "model__max_depth": 50, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row4_col6" class="data row4 col6" >52.073663</td>
    </tr>
    <tr>
      <td id="T_698e5_row5_col0" class="data row5 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row5_col1" class="data row5 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row5_col2" class="data row5 col2" >Stemmed</td>
      <td id="T_698e5_row5_col3" class="data row5 col3" >0.877355</td>
      <td id="T_698e5_row5_col4" class="data row5 col4" >1.921986</td>
      <td id="T_698e5_row5_col5" class="data row5 col5" >{"model__criterion": "gini", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row5_col6" class="data row5 col6" >43.795878</td>
    </tr>
    <tr>
      <td id="T_698e5_row6_col0" class="data row6 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row6_col1" class="data row6 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row6_col2" class="data row6 col2" >Lemmatized</td>
      <td id="T_698e5_row6_col3" class="data row6 col3" >0.881238</td>
      <td id="T_698e5_row6_col4" class="data row6 col4" >1.896561</td>
      <td id="T_698e5_row6_col5" class="data row6 col5" >{"model__criterion": "entropy", "model__max_depth": 25, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row6_col6" class="data row6 col6" >43.733084</td>
    </tr>
    <tr>
      <td id="T_698e5_row7_col0" class="data row7 col0" >DecisionTreeClassifier</td>
      <td id="T_698e5_row7_col1" class="data row7 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row7_col2" class="data row7 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row7_col3" class="data row7 col3" >0.879329</td>
      <td id="T_698e5_row7_col4" class="data row7 col4" >1.913801</td>
      <td id="T_698e5_row7_col5" class="data row7 col5" >{"model__criterion": "gini", "model__max_depth": 50, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row7_col6" class="data row7 col6" >44.665188</td>
    </tr>
    <tr>
      <td id="T_698e5_row8_col0" class="data row8 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row8_col1" class="data row8 col1" >CountVectorizer</td>
      <td id="T_698e5_row8_col2" class="data row8 col2" >Original</td>
      <td id="T_698e5_row8_col3" class="data row8 col3" >0.931534</td>
      <td id="T_698e5_row8_col4" class="data row8 col4" >0.322254</td>
      <td id="T_698e5_row8_col5" class="data row8 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row8_col6" class="data row8 col6" >108.423324</td>
    </tr>
    <tr>
      <td id="T_698e5_row9_col0" class="data row9 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row9_col1" class="data row9 col1" >CountVectorizer</td>
      <td id="T_698e5_row9_col2" class="data row9 col2" >Stemmed</td>
      <td id="T_698e5_row9_col3" class="data row9 col3" >0.942798</td>
      <td id="T_698e5_row9_col4" class="data row9 col4" >0.263932</td>
      <td id="T_698e5_row9_col5" class="data row9 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_698e5_row9_col6" class="data row9 col6" >108.718411</td>
    </tr>
    <tr>
      <td id="T_698e5_row10_col0" class="data row10 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row10_col1" class="data row10 col1" >CountVectorizer</td>
      <td id="T_698e5_row10_col2" class="data row10 col2" >Lemmatized</td>
      <td id="T_698e5_row10_col3" class="data row10 col3" >0.938484</td>
      <td id="T_698e5_row10_col4" class="data row10 col4" >0.270462</td>
      <td id="T_698e5_row10_col5" class="data row10 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_698e5_row10_col6" class="data row10 col6" >98.608885</td>
    </tr>
    <tr>
      <td id="T_698e5_row11_col0" class="data row11 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row11_col1" class="data row11 col1" >CountVectorizer</td>
      <td id="T_698e5_row11_col2" class="data row11 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row11_col3" class="data row11 col3" >0.942643</td>
      <td id="T_698e5_row11_col4" class="data row11 col4" >0.264656</td>
      <td id="T_698e5_row11_col5" class="data row11 col5" >{"model__n_neighbors": 100, "model__weights": "distance", "vectorizer__max_features": 500}</td>
      <td id="T_698e5_row11_col6" class="data row11 col6" >106.864122</td>
    </tr>
    <tr>
      <td id="T_698e5_row12_col0" class="data row12 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row12_col1" class="data row12 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row12_col2" class="data row12 col2" >Original</td>
      <td id="T_698e5_row12_col3" class="data row12 col3" >0.973472</td>
      <td id="T_698e5_row12_col4" class="data row12 col4" >0.316468</td>
      <td id="T_698e5_row12_col5" class="data row12 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row12_col6" class="data row12 col6" >115.396045</td>
    </tr>
    <tr>
      <td id="T_698e5_row13_col0" class="data row13 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row13_col1" class="data row13 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row13_col2" class="data row13 col2" >Stemmed</td>
      <td id="T_698e5_row13_col3" class="data row13 col3" >0.971769</td>
      <td id="T_698e5_row13_col4" class="data row13 col4" >0.261051</td>
      <td id="T_698e5_row13_col5" class="data row13 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row13_col6" class="data row13 col6" >152.728229</td>
    </tr>
    <tr>
      <td id="T_698e5_row14_col0" class="data row14 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row14_col1" class="data row14 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row14_col2" class="data row14 col2" >Lemmatized</td>
      <td id="T_698e5_row14_col3" class="data row14 col3" >0.972244</td>
      <td id="T_698e5_row14_col4" class="data row14 col4" >0.269687</td>
      <td id="T_698e5_row14_col5" class="data row14 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row14_col6" class="data row14 col6" >99.705519</td>
    </tr>
    <tr>
      <td id="T_698e5_row15_col0" class="data row15 col0" >KNeighborsClassifier</td>
      <td id="T_698e5_row15_col1" class="data row15 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row15_col2" class="data row15 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row15_col3" class="data row15 col3" >0.971738</td>
      <td id="T_698e5_row15_col4" class="data row15 col4" >0.247447</td>
      <td id="T_698e5_row15_col5" class="data row15 col5" >{"model__n_neighbors": 500, "model__weights": "distance", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row15_col6" class="data row15 col6" >180.872807</td>
    </tr>
    <tr>
      <td id="T_698e5_row16_col0" class="data row16 col0" >MultinomialNB</td>
      <td id="T_698e5_row16_col1" class="data row16 col1" >CountVectorizer</td>
      <td id="T_698e5_row16_col2" class="data row16 col2" >Original</td>
      <td id="T_698e5_row16_col3" class="data row16 col3" >0.953175</td>
      <td id="T_698e5_row16_col4" class="data row16 col4" >0.302751</td>
      <td id="T_698e5_row16_col5" class="data row16 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row16_col6" class="data row16 col6" >5.602990</td>
    </tr>
    <tr>
      <td id="T_698e5_row17_col0" class="data row17 col0" >MultinomialNB</td>
      <td id="T_698e5_row17_col1" class="data row17 col1" >CountVectorizer</td>
      <td id="T_698e5_row17_col2" class="data row17 col2" >Stemmed</td>
      <td id="T_698e5_row17_col3" class="data row17 col3" >0.954360</td>
      <td id="T_698e5_row17_col4" class="data row17 col4" >0.246380</td>
      <td id="T_698e5_row17_col5" class="data row17 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row17_col6" class="data row17 col6" >4.624883</td>
    </tr>
    <tr>
      <td id="T_698e5_row18_col0" class="data row18 col0" >MultinomialNB</td>
      <td id="T_698e5_row18_col1" class="data row18 col1" >CountVectorizer</td>
      <td id="T_698e5_row18_col2" class="data row18 col2" >Lemmatized</td>
      <td id="T_698e5_row18_col3" class="data row18 col3" >0.955302</td>
      <td id="T_698e5_row18_col4" class="data row18 col4" >0.263014</td>
      <td id="T_698e5_row18_col5" class="data row18 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row18_col6" class="data row18 col6" >4.775329</td>
    </tr>
    <tr>
      <td id="T_698e5_row19_col0" class="data row19 col0" >MultinomialNB</td>
      <td id="T_698e5_row19_col1" class="data row19 col1" >CountVectorizer</td>
      <td id="T_698e5_row19_col2" class="data row19 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row19_col3" class="data row19 col3" >0.954424</td>
      <td id="T_698e5_row19_col4" class="data row19 col4" >0.254128</td>
      <td id="T_698e5_row19_col5" class="data row19 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row19_col6" class="data row19 col6" >4.769068</td>
    </tr>
    <tr>
      <td id="T_698e5_row20_col0" class="data row20 col0" >MultinomialNB</td>
      <td id="T_698e5_row20_col1" class="data row20 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row20_col2" class="data row20 col2" >Original</td>
      <td id="T_698e5_row20_col3" class="data row20 col3" >0.964878</td>
      <td id="T_698e5_row20_col4" class="data row20 col4" >0.322025</td>
      <td id="T_698e5_row20_col5" class="data row20 col5" >{"model__alpha": 0.1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row20_col6" class="data row20 col6" >5.823270</td>
    </tr>
    <tr>
      <td id="T_698e5_row21_col0" class="data row21 col0" >MultinomialNB</td>
      <td id="T_698e5_row21_col1" class="data row21 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row21_col2" class="data row21 col2" >Stemmed</td>
      <td id="T_698e5_row21_col3" class="data row21 col3" >0.965847</td>
      <td id="T_698e5_row21_col4" class="data row21 col4" >0.252212</td>
      <td id="T_698e5_row21_col5" class="data row21 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_698e5_row21_col6" class="data row21 col6" >4.761241</td>
    </tr>
    <tr>
      <td id="T_698e5_row22_col0" class="data row22 col0" >MultinomialNB</td>
      <td id="T_698e5_row22_col1" class="data row22 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row22_col2" class="data row22 col2" >Lemmatized</td>
      <td id="T_698e5_row22_col3" class="data row22 col3" >0.965914</td>
      <td id="T_698e5_row22_col4" class="data row22 col4" >0.267241</td>
      <td id="T_698e5_row22_col5" class="data row22 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_698e5_row22_col6" class="data row22 col6" >4.812961</td>
    </tr>
    <tr>
      <td id="T_698e5_row23_col0" class="data row23 col0" >MultinomialNB</td>
      <td id="T_698e5_row23_col1" class="data row23 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row23_col2" class="data row23 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row23_col3" class="data row23 col3" >0.965847</td>
      <td id="T_698e5_row23_col4" class="data row23 col4" >0.246832</td>
      <td id="T_698e5_row23_col5" class="data row23 col5" >{"model__alpha": 1, "vectorizer__max_features": 500}</td>
      <td id="T_698e5_row23_col6" class="data row23 col6" >4.627067</td>
    </tr>
    <tr>
      <td id="T_698e5_row24_col0" class="data row24 col0" >ComplementNB</td>
      <td id="T_698e5_row24_col1" class="data row24 col1" >CountVectorizer</td>
      <td id="T_698e5_row24_col2" class="data row24 col2" >Original</td>
      <td id="T_698e5_row24_col3" class="data row24 col3" >0.944329</td>
      <td id="T_698e5_row24_col4" class="data row24 col4" >0.297268</td>
      <td id="T_698e5_row24_col5" class="data row24 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row24_col6" class="data row24 col6" >5.396361</td>
    </tr>
    <tr>
      <td id="T_698e5_row25_col0" class="data row25 col0" >ComplementNB</td>
      <td id="T_698e5_row25_col1" class="data row25 col1" >CountVectorizer</td>
      <td id="T_698e5_row25_col2" class="data row25 col2" >Stemmed</td>
      <td id="T_698e5_row25_col3" class="data row25 col3" >0.943821</td>
      <td id="T_698e5_row25_col4" class="data row25 col4" >0.240541</td>
      <td id="T_698e5_row25_col5" class="data row25 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row25_col6" class="data row25 col6" >4.727558</td>
    </tr>
    <tr>
      <td id="T_698e5_row26_col0" class="data row26 col0" >ComplementNB</td>
      <td id="T_698e5_row26_col1" class="data row26 col1" >CountVectorizer</td>
      <td id="T_698e5_row26_col2" class="data row26 col2" >Lemmatized</td>
      <td id="T_698e5_row26_col3" class="data row26 col3" >0.945060</td>
      <td id="T_698e5_row26_col4" class="data row26 col4" >0.255028</td>
      <td id="T_698e5_row26_col5" class="data row26 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row26_col6" class="data row26 col6" >4.606935</td>
    </tr>
    <tr>
      <td id="T_698e5_row27_col0" class="data row27 col0" >ComplementNB</td>
      <td id="T_698e5_row27_col1" class="data row27 col1" >CountVectorizer</td>
      <td id="T_698e5_row27_col2" class="data row27 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row27_col3" class="data row27 col3" >0.943771</td>
      <td id="T_698e5_row27_col4" class="data row27 col4" >0.237545</td>
      <td id="T_698e5_row27_col5" class="data row27 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row27_col6" class="data row27 col6" >4.499348</td>
    </tr>
    <tr>
      <td id="T_698e5_row28_col0" class="data row28 col0" >ComplementNB</td>
      <td id="T_698e5_row28_col1" class="data row28 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row28_col2" class="data row28 col2" >Original</td>
      <td id="T_698e5_row28_col3" class="data row28 col3" >0.971004</td>
      <td id="T_698e5_row28_col4" class="data row28 col4" >0.290711</td>
      <td id="T_698e5_row28_col5" class="data row28 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row28_col6" class="data row28 col6" >5.487691</td>
    </tr>
    <tr>
      <td id="T_698e5_row29_col0" class="data row29 col0" >ComplementNB</td>
      <td id="T_698e5_row29_col1" class="data row29 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row29_col2" class="data row29 col2" >Stemmed</td>
      <td id="T_698e5_row29_col3" class="data row29 col3" >0.969519</td>
      <td id="T_698e5_row29_col4" class="data row29 col4" >0.240811</td>
      <td id="T_698e5_row29_col5" class="data row29 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row29_col6" class="data row29 col6" >4.567892</td>
    </tr>
    <tr>
      <td id="T_698e5_row30_col0" class="data row30 col0" >ComplementNB</td>
      <td id="T_698e5_row30_col1" class="data row30 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row30_col2" class="data row30 col2" >Lemmatized</td>
      <td id="T_698e5_row30_col3" class="data row30 col3" >0.969854</td>
      <td id="T_698e5_row30_col4" class="data row30 col4" >0.252923</td>
      <td id="T_698e5_row30_col5" class="data row30 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row30_col6" class="data row30 col6" >4.638796</td>
    </tr>
    <tr>
      <td id="T_698e5_row31_col0" class="data row31 col0" >ComplementNB</td>
      <td id="T_698e5_row31_col1" class="data row31 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row31_col2" class="data row31 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row31_col3" class="data row31 col3" >0.969578</td>
      <td id="T_698e5_row31_col4" class="data row31 col4" >0.239258</td>
      <td id="T_698e5_row31_col5" class="data row31 col5" >{"model__alpha": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row31_col6" class="data row31 col6" >4.524278</td>
    </tr>
    <tr>
      <td id="T_698e5_row32_col0" class="data row32 col0" >LogisticRegression</td>
      <td id="T_698e5_row32_col1" class="data row32 col1" >CountVectorizer</td>
      <td id="T_698e5_row32_col2" class="data row32 col2" >Original</td>
      <td id="T_698e5_row32_col3" class="data row32 col3" >0.958692</td>
      <td id="T_698e5_row32_col4" class="data row32 col4" >90.377502</td>
      <td id="T_698e5_row32_col5" class="data row32 col5" >{"model__C": 100, "model__l1_ratio": 1.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row32_col6" class="data row32 col6" >6068.104392</td>
    </tr>
    <tr>
      <td id="T_698e5_row33_col0" class="data row33 col0" >LogisticRegression</td>
      <td id="T_698e5_row33_col1" class="data row33 col1" >CountVectorizer</td>
      <td id="T_698e5_row33_col2" class="data row33 col2" >Stemmed</td>
      <td id="T_698e5_row33_col3" class="data row33 col3" >0.960369</td>
      <td id="T_698e5_row33_col4" class="data row33 col4" >57.124586</td>
      <td id="T_698e5_row33_col5" class="data row33 col5" >{"model__C": 100, "model__l1_ratio": 0.5, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row33_col6" class="data row33 col6" >3712.599044</td>
    </tr>
    <tr>
      <td id="T_698e5_row34_col0" class="data row34 col0" >LogisticRegression</td>
      <td id="T_698e5_row34_col1" class="data row34 col1" >CountVectorizer</td>
      <td id="T_698e5_row34_col2" class="data row34 col2" >Lemmatized</td>
      <td id="T_698e5_row34_col3" class="data row34 col3" >0.960092</td>
      <td id="T_698e5_row34_col4" class="data row34 col4" >131.696927</td>
      <td id="T_698e5_row34_col5" class="data row34 col5" >{"model__C": 100, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row34_col6" class="data row34 col6" >8059.465757</td>
    </tr>
    <tr>
      <td id="T_698e5_row35_col0" class="data row35 col0" >LogisticRegression</td>
      <td id="T_698e5_row35_col1" class="data row35 col1" >CountVectorizer</td>
      <td id="T_698e5_row35_col2" class="data row35 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row35_col3" class="data row35 col3" >0.960330</td>
      <td id="T_698e5_row35_col4" class="data row35 col4" >102.627301</td>
      <td id="T_698e5_row35_col5" class="data row35 col5" >{"model__C": 10, "model__l1_ratio": 0.0, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row35_col6" class="data row35 col6" >5399.201219</td>
    </tr>
    <tr>
      <td id="T_698e5_row36_col0" class="data row36 col0" >LogisticRegression</td>
      <td id="T_698e5_row36_col1" class="data row36 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row36_col2" class="data row36 col2" >Original</td>
      <td id="T_698e5_row36_col3" class="data row36 col3" >0.986617</td>
      <td id="T_698e5_row36_col4" class="data row36 col4" >43.292536</td>
      <td id="T_698e5_row36_col5" class="data row36 col5" >{"model__C": 1, "model__l1_ratio": 0.5, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row36_col6" class="data row36 col6" >2345.625238</td>
    </tr>
    <tr>
      <td id="T_698e5_row37_col0" class="data row37 col0" >LogisticRegression</td>
      <td id="T_698e5_row37_col1" class="data row37 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row37_col2" class="data row37 col2" >Stemmed</td>
      <td id="T_698e5_row37_col3" class="data row37 col3" >0.984014</td>
      <td id="T_698e5_row37_col4" class="data row37 col4" >41.102306</td>
      <td id="T_698e5_row37_col5" class="data row37 col5" >{"model__C": 1, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row37_col6" class="data row37 col6" >2198.860511</td>
    </tr>
    <tr>
      <td id="T_698e5_row38_col0" class="data row38 col0" >LogisticRegression</td>
      <td id="T_698e5_row38_col1" class="data row38 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row38_col2" class="data row38 col2" >Lemmatized</td>
      <td id="T_698e5_row38_col3" class="data row38 col3" >0.984919</td>
      <td id="T_698e5_row38_col4" class="data row38 col4" >40.239015</td>
      <td id="T_698e5_row38_col5" class="data row38 col5" >{"model__C": 1, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row38_col6" class="data row38 col6" >2193.557615</td>
    </tr>
    <tr>
      <td id="T_698e5_row39_col0" class="data row39 col0" >LogisticRegression</td>
      <td id="T_698e5_row39_col1" class="data row39 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row39_col2" class="data row39 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row39_col3" class="data row39 col3" >0.984005</td>
      <td id="T_698e5_row39_col4" class="data row39 col4" >41.160850</td>
      <td id="T_698e5_row39_col5" class="data row39 col5" >{"model__C": 1, "model__l1_ratio": 0.25, "model__penalty": "elasticnet", "model__solver": "saga", "vectorizer__max_features": null}</td>
      <td id="T_698e5_row39_col6" class="data row39 col6" >2205.567522</td>
    </tr>
    <tr>
      <td id="T_698e5_row40_col0" class="data row40 col0" >XGBClassifier</td>
      <td id="T_698e5_row40_col1" class="data row40 col1" >CountVectorizer</td>
      <td id="T_698e5_row40_col2" class="data row40 col2" >Original</td>
      <td id="T_698e5_row40_col3" class="data row40 col3" >0.985959</td>
      <td id="T_698e5_row40_col4" class="data row40 col4" >11.995659</td>
      <td id="T_698e5_row40_col5" class="data row40 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row40_col6" class="data row40 col6" >831.363589</td>
    </tr>
    <tr>
      <td id="T_698e5_row41_col0" class="data row41 col0" >XGBClassifier</td>
      <td id="T_698e5_row41_col1" class="data row41 col1" >CountVectorizer</td>
      <td id="T_698e5_row41_col2" class="data row41 col2" >Stemmed</td>
      <td id="T_698e5_row41_col3" class="data row41 col3" >0.985284</td>
      <td id="T_698e5_row41_col4" class="data row41 col4" >9.742910</td>
      <td id="T_698e5_row41_col5" class="data row41 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row41_col6" class="data row41 col6" >691.172482</td>
    </tr>
    <tr>
      <td id="T_698e5_row42_col0" class="data row42 col0" >XGBClassifier</td>
      <td id="T_698e5_row42_col1" class="data row42 col1" >CountVectorizer</td>
      <td id="T_698e5_row42_col2" class="data row42 col2" >Lemmatized</td>
      <td id="T_698e5_row42_col3" class="data row42 col3" >0.985588</td>
      <td id="T_698e5_row42_col4" class="data row42 col4" >13.378208</td>
      <td id="T_698e5_row42_col5" class="data row42 col5" >{"model__colsample_bytree": 0.7, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row42_col6" class="data row42 col6" >918.109954</td>
    </tr>
    <tr>
      <td id="T_698e5_row43_col0" class="data row43 col0" >XGBClassifier</td>
      <td id="T_698e5_row43_col1" class="data row43 col1" >CountVectorizer</td>
      <td id="T_698e5_row43_col2" class="data row43 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row43_col3" class="data row43 col3" >0.985297</td>
      <td id="T_698e5_row43_col4" class="data row43 col4" >15.218013</td>
      <td id="T_698e5_row43_col5" class="data row43 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row43_col6" class="data row43 col6" >1037.484151</td>
    </tr>
    <tr>
      <td id="T_698e5_row44_col0" class="data row44 col0" >XGBClassifier</td>
      <td id="T_698e5_row44_col1" class="data row44 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row44_col2" class="data row44 col2" >Original</td>
      <td id="T_698e5_row44_col3" class="data row44 col3" >0.986995</td>
      <td id="T_698e5_row44_col4" class="data row44 col4" >103.693078</td>
      <td id="T_698e5_row44_col5" class="data row44 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row44_col6" class="data row44 col6" >6421.626396</td>
    </tr>
    <tr>
      <td id="T_698e5_row45_col0" class="data row45 col0" >XGBClassifier</td>
      <td id="T_698e5_row45_col1" class="data row45 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row45_col2" class="data row45 col2" >Stemmed</td>
      <td id="T_698e5_row45_col3" class="data row45 col3" >0.985848</td>
      <td id="T_698e5_row45_col4" class="data row45 col4" >89.540373</td>
      <td id="T_698e5_row45_col5" class="data row45 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row45_col6" class="data row45 col6" >5565.953025</td>
    </tr>
    <tr>
      <td id="T_698e5_row46_col0" class="data row46 col0" >XGBClassifier</td>
      <td id="T_698e5_row46_col1" class="data row46 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row46_col2" class="data row46 col2" >Lemmatized</td>
      <td id="T_698e5_row46_col3" class="data row46 col3" >0.986182</td>
      <td id="T_698e5_row46_col4" class="data row46 col4" >92.151520</td>
      <td id="T_698e5_row46_col5" class="data row46 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 0.7, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row46_col6" class="data row46 col6" >5716.530820</td>
    </tr>
    <tr>
      <td id="T_698e5_row47_col0" class="data row47 col0" >XGBClassifier</td>
      <td id="T_698e5_row47_col1" class="data row47 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row47_col2" class="data row47 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row47_col3" class="data row47 col3" >0.985967</td>
      <td id="T_698e5_row47_col4" class="data row47 col4" >90.555582</td>
      <td id="T_698e5_row47_col5" class="data row47 col5" >{"model__colsample_bytree": 0.5, "model__max_depth": 10, "model__subsample": 1, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row47_col6" class="data row47 col6" >5631.057042</td>
    </tr>
    <tr>
      <td id="T_698e5_row48_col0" class="data row48 col0" >BaggingClassifier</td>
      <td id="T_698e5_row48_col1" class="data row48 col1" >CountVectorizer</td>
      <td id="T_698e5_row48_col2" class="data row48 col2" >Original</td>
      <td id="T_698e5_row48_col3" class="data row48 col3" >0.965163</td>
      <td id="T_698e5_row48_col4" class="data row48 col4" >174.811794</td>
      <td id="T_698e5_row48_col5" class="data row48 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row48_col6" class="data row48 col6" >1432.970788</td>
    </tr>
    <tr>
      <td id="T_698e5_row49_col0" class="data row49 col0" >BaggingClassifier</td>
      <td id="T_698e5_row49_col1" class="data row49 col1" >CountVectorizer</td>
      <td id="T_698e5_row49_col2" class="data row49 col2" >Stemmed</td>
      <td id="T_698e5_row49_col3" class="data row49 col3" >0.966437</td>
      <td id="T_698e5_row49_col4" class="data row49 col4" >148.700142</td>
      <td id="T_698e5_row49_col5" class="data row49 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row49_col6" class="data row49 col6" >1223.369071</td>
    </tr>
    <tr>
      <td id="T_698e5_row50_col0" class="data row50 col0" >BaggingClassifier</td>
      <td id="T_698e5_row50_col1" class="data row50 col1" >CountVectorizer</td>
      <td id="T_698e5_row50_col2" class="data row50 col2" >Lemmatized</td>
      <td id="T_698e5_row50_col3" class="data row50 col3" >0.964489</td>
      <td id="T_698e5_row50_col4" class="data row50 col4" >134.667691</td>
      <td id="T_698e5_row50_col5" class="data row50 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row50_col6" class="data row50 col6" >1120.673748</td>
    </tr>
    <tr>
      <td id="T_698e5_row51_col0" class="data row51 col0" >BaggingClassifier</td>
      <td id="T_698e5_row51_col1" class="data row51 col1" >CountVectorizer</td>
      <td id="T_698e5_row51_col2" class="data row51 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row51_col3" class="data row51 col3" >0.964342</td>
      <td id="T_698e5_row51_col4" class="data row51 col4" >133.251862</td>
      <td id="T_698e5_row51_col5" class="data row51 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row51_col6" class="data row51 col6" >1115.898864</td>
    </tr>
    <tr>
      <td id="T_698e5_row52_col0" class="data row52 col0" >BaggingClassifier</td>
      <td id="T_698e5_row52_col1" class="data row52 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row52_col2" class="data row52 col2" >Original</td>
      <td id="T_698e5_row52_col3" class="data row52 col3" >0.970092</td>
      <td id="T_698e5_row52_col4" class="data row52 col4" >258.067960</td>
      <td id="T_698e5_row52_col5" class="data row52 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row52_col6" class="data row52 col6" >2199.784869</td>
    </tr>
    <tr>
      <td id="T_698e5_row53_col0" class="data row53 col0" >BaggingClassifier</td>
      <td id="T_698e5_row53_col1" class="data row53 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row53_col2" class="data row53 col2" >Stemmed</td>
      <td id="T_698e5_row53_col3" class="data row53 col3" >0.970356</td>
      <td id="T_698e5_row53_col4" class="data row53 col4" >215.014867</td>
      <td id="T_698e5_row53_col5" class="data row53 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row53_col6" class="data row53 col6" >1835.195167</td>
    </tr>
    <tr>
      <td id="T_698e5_row54_col0" class="data row54 col0" >BaggingClassifier</td>
      <td id="T_698e5_row54_col1" class="data row54 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row54_col2" class="data row54 col2" >Lemmatized</td>
      <td id="T_698e5_row54_col3" class="data row54 col3" >0.971229</td>
      <td id="T_698e5_row54_col4" class="data row54 col4" >217.465611</td>
      <td id="T_698e5_row54_col5" class="data row54 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row54_col6" class="data row54 col6" >1849.731134</td>
    </tr>
    <tr>
      <td id="T_698e5_row55_col0" class="data row55 col0" >BaggingClassifier</td>
      <td id="T_698e5_row55_col1" class="data row55 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row55_col2" class="data row55 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row55_col3" class="data row55 col3" >0.971270</td>
      <td id="T_698e5_row55_col4" class="data row55 col4" >218.341361</td>
      <td id="T_698e5_row55_col5" class="data row55 col5" >{"model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row55_col6" class="data row55 col6" >1857.983144</td>
    </tr>
    <tr>
      <td id="T_698e5_row56_col0" class="data row56 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row56_col1" class="data row56 col1" >CountVectorizer</td>
      <td id="T_698e5_row56_col2" class="data row56 col2" >Original</td>
      <td id="T_698e5_row56_col3" class="data row56 col3" >0.975566</td>
      <td id="T_698e5_row56_col4" class="data row56 col4" >7.260155</td>
      <td id="T_698e5_row56_col5" class="data row56 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row56_col6" class="data row56 col6" >714.958341</td>
    </tr>
    <tr>
      <td id="T_698e5_row57_col0" class="data row57 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row57_col1" class="data row57 col1" >CountVectorizer</td>
      <td id="T_698e5_row57_col2" class="data row57 col2" >Stemmed</td>
      <td id="T_698e5_row57_col3" class="data row57 col3" >0.976810</td>
      <td id="T_698e5_row57_col4" class="data row57 col4" >7.271615</td>
      <td id="T_698e5_row57_col5" class="data row57 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row57_col6" class="data row57 col6" >705.086587</td>
    </tr>
    <tr>
      <td id="T_698e5_row58_col0" class="data row58 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row58_col1" class="data row58 col1" >CountVectorizer</td>
      <td id="T_698e5_row58_col2" class="data row58 col2" >Lemmatized</td>
      <td id="T_698e5_row58_col3" class="data row58 col3" >0.977316</td>
      <td id="T_698e5_row58_col4" class="data row58 col4" >6.936464</td>
      <td id="T_698e5_row58_col5" class="data row58 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row58_col6" class="data row58 col6" >681.951016</td>
    </tr>
    <tr>
      <td id="T_698e5_row59_col0" class="data row59 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row59_col1" class="data row59 col1" >CountVectorizer</td>
      <td id="T_698e5_row59_col2" class="data row59 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row59_col3" class="data row59 col3" >0.976916</td>
      <td id="T_698e5_row59_col4" class="data row59 col4" >7.077846</td>
      <td id="T_698e5_row59_col5" class="data row59 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row59_col6" class="data row59 col6" >691.136447</td>
    </tr>
    <tr>
      <td id="T_698e5_row60_col0" class="data row60 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row60_col1" class="data row60 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row60_col2" class="data row60 col2" >Original</td>
      <td id="T_698e5_row60_col3" class="data row60 col3" >0.977822</td>
      <td id="T_698e5_row60_col4" class="data row60 col4" >8.819328</td>
      <td id="T_698e5_row60_col5" class="data row60 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row60_col6" class="data row60 col6" >838.956470</td>
    </tr>
    <tr>
      <td id="T_698e5_row61_col0" class="data row61 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row61_col1" class="data row61 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row61_col2" class="data row61 col2" >Stemmed</td>
      <td id="T_698e5_row61_col3" class="data row61 col3" >0.979089</td>
      <td id="T_698e5_row61_col4" class="data row61 col4" >8.393873</td>
      <td id="T_698e5_row61_col5" class="data row61 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row61_col6" class="data row61 col6" >799.868159</td>
    </tr>
    <tr>
      <td id="T_698e5_row62_col0" class="data row62 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row62_col1" class="data row62 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row62_col2" class="data row62 col2" >Lemmatized</td>
      <td id="T_698e5_row62_col3" class="data row62 col3" >0.979128</td>
      <td id="T_698e5_row62_col4" class="data row62 col4" >7.952093</td>
      <td id="T_698e5_row62_col5" class="data row62 col5" >{"model__class_weight": "balanced", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row62_col6" class="data row62 col6" >762.615708</td>
    </tr>
    <tr>
      <td id="T_698e5_row63_col0" class="data row63 col0" >RandomForestClassifier</td>
      <td id="T_698e5_row63_col1" class="data row63 col1" >TfidfVectorizer</td>
      <td id="T_698e5_row63_col2" class="data row63 col2" >Stemmed and Lemmatized</td>
      <td id="T_698e5_row63_col3" class="data row63 col3" >0.978474</td>
      <td id="T_698e5_row63_col4" class="data row63 col4" >8.286464</td>
      <td id="T_698e5_row63_col5" class="data row63 col5" >{"model__class_weight": "balanced_subsample", "model__max_depth": 100, "model__n_estimators": 100, "vectorizer__max_features": null}</td>
      <td id="T_698e5_row63_col6" class="data row63 col6" >794.422122</td>
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
 XGBClassifier with vectorizer TfidfVectorizer with input pre-processing Original had ROC AUC score 98.69954725563986% and mean fit time of 103.6930783228742 seconds.</li><li>The worst model was:
 DecisionTreeClassifier with vectorizer TfidfVectorizer with input pre-processing Stemmed had ROC AUC score 87.73545921411838% and mean fit time of 1.921985923250516 seconds.</li></ul></td>
<td><ul>
<li>The fastest model was:
 ComplementNB with vectorizer CountVectorizer with input pre-processing Stemmed and Lemmatized had ROC AUC score 94.3771473528919% and mean fit time of 0.2375452121098836 seconds.</li><li>The slowest model was
 BaggingClassifier with vectorizer TfidfVectorizer with input pre-processing Original had ROC AUC score 97.00922613181199% and mean fit time of 258.0679601281881 seconds.</li></ul></td>
</tr>
</table>



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
### Next Steps
Deep learning may be able get us even better results. This would be worth researching.









# Runtime Metrics:
```
{
  "pre_processing": {
    "total_time": 45.985604666988365
  },
  "modeling": {
    "DecisionTreeClassifier": {
      "CountVectorizer": {
        "Original": {
          "total_time": 29.435409792000428
        },
        "Stemmed": {
          "total_time": 25.612033208017237
        },
        "Lemmatized": {
          "total_time": 25.303553000005195
        },
        "Stemmed and Lemmatized": {
          "total_time": 25.30541295898729
        },
        "total_time": 106.5526664579811
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 52.073662916984176
        },
        "Stemmed": {
          "total_time": 43.79587774997344
        },
        "Lemmatized": {
          "total_time": 43.73308395899949
        },
        "Stemmed and Lemmatized": {
          "total_time": 44.66518791698036
        },
        "total_time": 185.38677316601388
      }
    },
    "total_time": 78989.07759445798,
    "KNeighborsClassifier": {
      "CountVectorizer": {
        "Original": {
          "total_time": 108.42332404199988
        },
        "Stemmed": {
          "total_time": 108.71841058300924
        },
        "Lemmatized": {
          "total_time": 98.60888470901409
        },
        "Stemmed and Lemmatized": {
          "total_time": 106.86412162499619
        },
        "total_time": 423.740269041009
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 115.39604504100862
        },
        "Stemmed": {
          "total_time": 152.72822883399203
        },
        "Lemmatized": {
          "total_time": 99.70551920798607
        },
        "Stemmed and Lemmatized": {
          "total_time": 180.872806667001
        },
        "total_time": 549.8940850000072
      }
    },
    "MultinomialNB": {
      "CountVectorizer": {
        "Original": {
          "total_time": 5.602990082988981
        },
        "Stemmed": {
          "total_time": 4.624882959004026
        },
        "Lemmatized": {
          "total_time": 4.775328999996418
        },
        "Stemmed and Lemmatized": {
          "total_time": 4.769068292021984
        },
        "total_time": 21.11140133300796
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 5.823270374996355
        },
        "Stemmed": {
          "total_time": 4.76124137500301
        },
        "Lemmatized": {
          "total_time": 4.8129605000140145
        },
        "Stemmed and Lemmatized": {
          "total_time": 4.627067333000014
        },
        "total_time": 21.415820207999786
      }
    },
    "ComplementNB": {
      "CountVectorizer": {
        "Original": {
          "total_time": 5.396360624989029
        },
        "Stemmed": {
          "total_time": 4.7275583329901565
        },
        "Lemmatized": {
          "total_time": 4.606934583018301
        },
        "Stemmed and Lemmatized": {
          "total_time": 4.499347666016547
        },
        "total_time": 21.194090083998162
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 5.487691458984045
        },
        "Stemmed": {
          "total_time": 4.567891834012698
        },
        "Lemmatized": {
          "total_time": 4.638796374987578
        },
        "Stemmed and Lemmatized": {
          "total_time": 4.524278208002215
        },
        "total_time": 20.624261832999764
      }
    },
    "LogisticRegression": {
      "CountVectorizer": {
        "Original": {
          "total_time": 6068.104392333014
        },
        "Stemmed": {
          "total_time": 3712.5990438330045
        },
        "Lemmatized": {
          "total_time": 8059.46575695902
        },
        "Stemmed and Lemmatized": {
          "total_time": 5399.201219375012
        },
        "total_time": 23240.912997290987
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 2345.625237542001
        },
        "Stemmed": {
          "total_time": 2198.8605105410097
        },
        "Lemmatized": {
          "total_time": 2193.5576149590197
        },
        "Stemmed and Lemmatized": {
          "total_time": 2205.5675222499995
        },
        "total_time": 8945.284257082996
      }
    },
    "XGBClassifier": {
      "CountVectorizer": {
        "Original": {
          "total_time": 831.3635886249831
        },
        "Stemmed": {
          "total_time": 691.1724815829948
        },
        "Lemmatized": {
          "total_time": 918.1099537080154
        },
        "Stemmed and Lemmatized": {
          "total_time": 1037.4841506250086
        },
        "total_time": 3481.8228723750217
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 6421.6263962090015
        },
        "Stemmed": {
          "total_time": 5565.953025166993
        },
        "Lemmatized": {
          "total_time": 5716.530819666979
        },
        "Stemmed and Lemmatized": {
          "total_time": 5631.057041916996
        },
        "total_time": 23338.859785834007
      }
    },
    "BaggingClassifier": {
      "CountVectorizer": {
        "Original": {
          "total_time": 1432.970788000006
        },
        "Stemmed": {
          "total_time": 1223.3690711669915
        },
        "Lemmatized": {
          "total_time": 1120.67374758402
        },
        "Stemmed and Lemmatized": {
          "total_time": 1115.898864250019
        },
        "total_time": 4894.748263082991
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 2199.7848693749984
        },
        "Stemmed": {
          "total_time": 1835.1951666249952
        },
        "Lemmatized": {
          "total_time": 1849.7311343750043
        },
        "Stemmed and Lemmatized": {
          "total_time": 1857.9831437079993
        },
        "total_time": 7744.627148333006
      }
    },
    "RandomForestClassifier": {
      "CountVectorizer": {
        "Original": {
          "total_time": 714.9583407920145
        },
        "Stemmed": {
          "total_time": 705.0865867089597
        },
        "Lemmatized": {
          "total_time": 681.9510156660108
        },
        "Stemmed and Lemmatized": {
          "total_time": 691.1364472500281
        },
        "total_time": 2795.3861675829685
      },
      "TfidfVectorizer": {
        "Original": {
          "total_time": 838.9564697500318
        },
        "Stemmed": {
          "total_time": 799.8681586250314
        },
        "Lemmatized": {
          "total_time": 762.6157083750004
        },
        "Stemmed and Lemmatized": {
          "total_time": 794.4221222919878
        },
        "total_time": 3197.9739349170122
      }
    }
  },
  "start_time": "2024-12-10 09:02:01.028715",
  "end_time": "2024-12-11 07:16:43.420141",
  "total_execution_time": 80082.42432816699
}
```


