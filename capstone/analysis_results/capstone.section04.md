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
