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

### Features
- There is only one feature Original which is free form text
- There are no missing values
- All data appears to be in english
- All data appears to be lowercased
- The data set is somewhat large and the classes are imbalanced.
- The classification field target has 8 distinct values
<a href="./analysis_results/capstone.raw.targetField.distribution.png" target="_blank"><img src="./analysis_results/capstone.raw.targetField.distribution.png"/></a>

