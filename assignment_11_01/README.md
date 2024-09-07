# Report: What drives the price of a car?

**Code:** [Data Analysis Workbook](./used_car_price_analysis.out.ipynb)

**NOTE:** *The processing of the juypter notebook take a long time and often disconnects from the session. In order to run it without monitoring it all the time use the wokraround below from command line.*

```
jupyter nbconvert --to notebook --execute used_car_price_analysis.template.ipynb --output=used_car_price_analysis.out.ipynb --ExecutePreprocessor.timeout=-1
```
<sub>Source: [screen-and-jupyter-a-way-to-run-long-notebooks-headles](https://www.maksimeren.com/post/screen-and-jupyter-a-way-to-run-long-notebooks-headless/)</sub>

Jump to the good stuff: [Recommendations](#Recommendations)

## Business Understanding

We are provided with a dataset of used car prices and features about that particular vehicle. Our final goal will be to identify which 'features' AND what values of those features most contribute to the final price both positively and negatively.

Because the data has high dimensionality we will need to make use of transformers to get the data ready for use in regularization. Once data is cleaned and prepared we will then try out multiple linear regression models to find the best one. Once done we will use the coefficients to identify how features contribute to price.

Once we have found these imortant features we will write up actionable guidance for used car business

**Steps involved:**

  - Examine the raw data and identify characterisitics of the data e.g. missing values, unique counts, invalid data...
  - Preprocess the data to get it ready for modelling by:
    - Identifying which features can be ignored and drop those columns/features
    - Identify non-ignorable missing features and either:
      - Impute missing values per row
      - Drop those rows
  - Decide what data transforms/normalization are required for numeric and categorical fields based on above decisions
  - Use regularization techniques with multiple (L1, L2,...) linear regression models using and find one with the best peformance for predicting prices
  - Repeat steps above if necessary to arrive at final 'best' model which we will measure by using the one with the lowest Mean Square Error (MSE)
  - Analyse the most important features 'selected' by the model based on the coefficients determined by the previous steps
## Data Understanding

### Raw Data Statistics

<table><tr><th>info()</th><th>describe()</th></tr><tr><td><pre><class 'pandas.core.frame.DataFrame'>
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 18 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            426880 non-null  int64  
 1   region        426880 non-null  object 
 2   price         426880 non-null  int64  
 3   year          425675 non-null  float64
 4   manufacturer  409234 non-null  object 
 5   model         421603 non-null  object 
 6   condition     252776 non-null  object 
 7   cylinders     249202 non-null  object 
 8   fuel          423867 non-null  object 
 9   odometer      422480 non-null  float64
 10  title_status  418638 non-null  object 
 11  transmission  424324 non-null  object 
 12  VIN           265838 non-null  object 
 13  drive         296313 non-null  object 
 14  size          120519 non-null  object 
 15  type          334022 non-null  object 
 16  paint_color   296677 non-null  object 
 17  state         426880 non-null  object 
dtypes: float64(2), int64(2), object(14)
memory usage: 58.6+ MB
</pre></td><td><pre>                 id         price           year      odometer
count  4.268800e+05  4.268800e+05  425675.000000  4.224800e+05
mean   7.311487e+09  7.519903e+04    2011.235191  9.804333e+04
std    4.473170e+06  1.218228e+07       9.452120  2.138815e+05
min    7.207408e+09  0.000000e+00    1900.000000  0.000000e+00
25%    7.308143e+09  5.900000e+03    2008.000000  3.770400e+04
50%    7.312621e+09  1.395000e+04    2013.000000  8.554800e+04
75%    7.315254e+09  2.648575e+04    2017.000000  1.335425e+05
max    7.317101e+09  3.736929e+09    2022.000000  1.000000e+07</pre></td></tr><tr><td colspan="2">
<a href="./analysis_results/module_11_01.step01.data_understanding.data.distribution.png" target="_blank"><img src="./analysis_results/module_11_01.step01.data_understanding.data.distribution.png"/></a>
</td></tr></tr></table>

### Analysis

 There are a lot of categorical columns that will need to be encoded. There are also a lot of missing values for fields that will likely be important to the model. We'll have to impute where we can and drop where it won't affect the size of the dataset too much.

<table>
<tr>
<th></th><th>Field</th><th>Type</th><th>Missing Value #</th><th>Missing Value %</th><th>Unique Value #</th><th>Notes</th></tr>
<tr>
<td>1</td><td>id</td><td>int64</td><td>0</td><td>0.0</td><td>426880</td><td><ul><li>Not useful for predictions.</ul></td></tr>
<tr>
<td>2</td><td>VIN</td><td>object</td><td>161042</td><td>37.72535607196401</td><td>118247</td><td><ul><li>Not useful for predictions.</ul></td></tr>
<tr>
<td>3</td><td>price</td><td>int64</td><td>0</td><td>0.0</td><td>15655</td><td><ul><li>Target Field.</li>
<li>Need to deal with outliers.</ul></td></tr>
<tr>
<td>4</td><td>odometer</td><td>float64</td><td>4400</td><td>1.030734632683658</td><td>104871</td><td><ul><li>Has an effect on price typically negative as mileage goes up.</li>
<li>Need to deal with outliers.There are only a small percentage of values missing.</ul></td></tr>
<tr>
<td>5</td><td>manufacturer</td><td>object</td><td>17646</td><td>4.133714392803598</td><td>43</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here and no easy way to determine them.</li>
<li>There are only a small percentage of values missing.</ul></td></tr>
<tr>
<td>6</td><td>model</td><td>object</td><td>5277</td><td>1.236178785607196</td><td>29650</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here and no easy way to determine them.</li>
<li>There are only a small percentage of values missing.</li>
<li>Free text field and there could be spelling mistakes or variations in order of words that aren't easy to normalize.</ul></td></tr>
<tr>
<td>7</td><td>type</td><td>object</td><td>92858</td><td>21.75271739130435</td><td>14</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here.</li>
<li>Can use manufacturer, model and year to fill in missing values</ul></td></tr>
<tr>
<td>8</td><td>drive</td><td>object</td><td>130567</td><td>30.58634745127436</td><td>4</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here.</li>
<li>Can use manufacturer, model and year to fill in missing values</ul></td></tr>
<tr>
<td>9</td><td>transmission</td><td>object</td><td>2556</td><td>0.5987631184407796</td><td>4</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here.</li>
<li>There are only a small percentage of values missing.</ul></td></tr>
<tr>
<td>10</td><td>size</td><td>object</td><td>306361</td><td>71.7674756371814</td><td>5</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here.</li>
<li>Can use manufacturer, model and year to fill in missing values</ul></td></tr>
<tr>
<td>11</td><td>cylinders</td><td>object</td><td>177678</td><td>41.6224700149925</td><td>9</td><td><ul><li>Has an effect on price.</li>
<li>There are lots of empty values here.</li>
<li>Can use manufacturer, model and year to fill in missing values</ul></td></tr>
<tr>
<td>12</td><td>fuel</td><td>object</td><td>3013</td><td>0.7058189655172413</td><td>6</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here.</li>
<li>There are only a small percentage of values missing.</ul></td></tr>
<tr>
<td>13</td><td>paint_color</td><td>object</td><td>130203</td><td>30.5010775862069</td><td>13</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here and no easy way to determine them.</ul></td></tr>
<tr>
<td>14</td><td>condition</td><td>object</td><td>174104</td><td>40.78523238380809</td><td>7</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here and no easy way to determine them.</ul></td></tr>
<tr>
<td>15</td><td>title_status</td><td>object</td><td>8242</td><td>1.930753373313343</td><td>7</td><td><ul><li>Has an effect on price.</li>
<li>There are empty values here.</li>
<li>There are only a small percentage of values missing.</ul></td></tr>
<tr>
<td>16</td><td>year</td><td>float64</td><td>1205</td><td>0.2822807346326837</td><td>115</td><td><ul><li>Has an effect on price typically positive as value goes up since it's a newer car.</li>
<li>Need to deal with outliers.</ul></td></tr>
<tr>
<td>17</td><td>state</td><td>object</td><td>0</td><td>0.0</td><td>51</td><td><ul><li>Has an effect on price.</li>
<li>Not really something dealer can control but can extract some useful information from this for other business decision making.</ul></td></tr>
<tr>
<td>18</td><td>region</td><td>object</td><td>0</td><td>0.0</td><td>404</td><td><ul><li>Has an effect on price.</li>
<li>Not really something dealer can control but can extract some useful information from this for other business decision making.</ul></td></tr>
</table>

## Data Preparation

### Cleanup Approach

<table>
<tr>
<th>Step</th><th>Processing</th></tr>
<tr>
<td>Step1</td><td><ul><li>Drop the VIN feature column</li>
<li>Drop the id feature column</li>
<li>Drop the paint_color feature column</ul></td></tr>
<tr>
<td>Step2</td><td><ul><li>Convert state to lower case values</li>
<li>Convert region to lower case values</li>
<li>Convert condition to lower case values</li>
<li>Convert cylinders to lower case values</li>
<li>Convert fuel to lower case values</li>
<li>Convert title_status to lower case values</li>
<li>Convert transmission to lower case values</li>
<li>Convert drive to lower case values</li>
<li>Convert manufacturer to lower case values</li>
<li>Convert size to lower case values</li>
<li>Convert type to lower case values</li>
<li>Convert model to lower case values and store value in new field cleaned_model</ul></td></tr>
<tr>
<td>Step3</td><td><ul><li>Drop rows where price is empty</li>
<li>Drop rows where model is empty</li>
<li>Drop rows where manufacturer is empty</li>
<li>Drop rows where odometer is empty</li>
<li>Drop rows where year is empty</ul></td></tr>
<tr>
<td>Step4</td><td><ul><li>Drop rows not meeting criteria Q1[0.25] <= price <= Q3[0.75]</li>
<li>Drop rows not meeting criteria Q1[0.25] <= odometer <= Q3[0.75]</li>
<li>Drop rows not meeting criteria Q1[0.25] <= year <= Q3[0.75]</ul></td></tr>
<tr>
<td>Step5</td><td><ul><li>Drop rows meeting the critera "price > 0"</li>
<li>Drop rows meeting the critera "odometer > 0"</li>
<li>Drop rows meeting the critera "year > 1900"</ul></td></tr>
<tr>
<td>Step6</td><td><ul><li>Fill rows where condition is empty with "unknown"</ul></td></tr>
<tr>
<td>Step7</td><td><ul><li>Using the fields manufacturer, cleaned_model, year find teh mode() for those fields in the dataset and assign to Lookup&Fill Pass1</li>
<li>Using the fields manufacturer, cleaned_model find teh mode() for those fields in the dataset and assign to Lookup&Fill Pass2</li>
<li>Using the fields manufacturer, type find teh mode() for those fields in the dataset and assign to Lookup&Fill Pass3</ul></td></tr>
<tr>
<td>Step8</td><td><ul><li>Fill rows where cylinders is empty with "unknown"</li>
<li>Fill rows where title_status is empty with "unknown"</li>
<li>Fill rows where transmission is empty with "unknown"</li>
<li>Fill rows where drive is empty with "unknown"</li>
<li>Fill rows where fuel is empty with "unknown"</li>
<li>Fill rows where size is empty with "unknown"</li>
<li>Fill rows where type is empty with "unknown"</ul></td></tr>
</table>

### Data Shape vs Processing Steps

<table>
<tr><td><a href="./analysis_results/module_11_01.step02.data_preparation.row_count.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.row_count.png"/></a></td></tr>
<tr><td><a href="./analysis_results/module_11_01.step02.data_preparation.missing_count.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.missing_count.png"/></a></td></tr>
<tr><td><a href="./analysis_results/module_11_01.step02.data_preparation.missing_percentage.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.missing_percentage.png"/></a></td></tr>
<tr><td><a href="./analysis_results/module_11_01.step02.data_preparation.unique_values.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.unique_values.png"/></a></td></tr>
</table>

### Prepared Data Statistics

<table><tr><th>info()</th><th>describe()</th></tr><tr><td><pre><class 'pandas.core.frame.DataFrame'>
Index: 347026 entries, 27 to 426879
Data columns (total 16 columns):
 #   Column         Non-Null Count   Dtype  
---  ------         --------------   -----  
 0   region         347026 non-null  object 
 1   price          347026 non-null  int64  
 2   year           347026 non-null  float64
 3   manufacturer   347026 non-null  object 
 4   model          347026 non-null  object 
 5   condition      347026 non-null  object 
 6   cylinders      347026 non-null  object 
 7   fuel           347026 non-null  object 
 8   odometer       347026 non-null  float64
 9   title_status   347026 non-null  object 
 10  transmission   347026 non-null  object 
 11  drive          347026 non-null  object 
 12  size           347026 non-null  object 
 13  type           347026 non-null  object 
 14  state          347026 non-null  object 
 15  cleaned_model  347026 non-null  object 
dtypes: float64(2), int64(1), object(13)
memory usage: 45.0+ MB
</pre></td><td><pre>               price           year       odometer
count  347026.000000  347026.000000  347026.000000
mean    18155.816293    2012.653957   91937.830367
std     12705.937595       5.269285   60031.657403
min         1.000000    1997.000000       1.000000
25%      7500.000000    2009.000000   39433.000000
50%     15590.000000    2014.000000   88000.000000
75%     26995.000000    2017.000000  134655.000000
max     57460.000000    2022.000000  275225.000000</pre></td></tr><tr><td colspan="2">
<a href="./analysis_results/module_11_01.step02.data_preparation.data.distribution.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.data.distribution.png"/></a>
</td></tr></tr></table>

## Modeling

### Model Analysis

Using the following features 

- Categorical=region, manufacturer, condition, cylinders, fuel, title_status, transmission, drive, size, type, state, cleaned_model 

- Numerical=year, odometer 

we tried several regression models including **Ridge, Lasso, ElasticNet** 

<a href="./analysis_results/module_11_01.step03.modeling.performance.png" target="_blank"><img src="./analysis_results/module_11_01.step03.modeling.performance.png"/></a> 

We have determined that the **best model** is **Ridge** based on **Test MSE=40949644.38**. We chose Test MSE because while it is more sensitive to outliers we've removed the outliers using IQR filtering. Had we not done this we would have used R2

## Evaluation

### Feature Results

We will now show the importances of all the features across models

<table>
<tr><td><a href="./analysis_results/module_11_01.step04.evaluation.coefficient.png" target="_blank"><img src="./analysis_results/module_11_01.step04.evaluation.coefficient.png"/></a></td></tr>
</table>

### Feature Analysis

We can see from this that the models generally agree on what one would expect:
- Model also contributes positively. This makes sense as people prefer some cars over others, but it was hard to get exact models since the cardinality is so high. This can be inferred from combination of other features however
- As year goes up price goes up. This makes sense as you are getting a newer car
- As odometer goes up price goes down. This makes sense as you are getting a car with a lot more miles, wear and tear
- Interestingly fuel, cylinders, region and drive also feature prominently. Cylinders made sense because you pay for more horsepower. But region has heavy influence
- Looking at other features we can see prefernces for types (truck+pickups > sedan) and drive (4wd > fwd)
## Recommendations

### Specific Features People Value

<table>
<tr><td><a href="./analysis_results/module_11_01.step04.evaluation.pos.coeff.png" target="_blank"><img src="./analysis_results/module_11_01.step04.evaluation.pos.coeff.png"/></a></td></tr>
</table>

Do Prioritize:

- Manufacturers=Toyota, Honda, Lexus, Tesla for the common nbrands
- Type=Pickups, Convertibles, Coupes and Trucks
- Size=Full Size, Mid Size
- Drive=4WD
- Cylinders=8
- Transmission=Manual
- Fuel=Diesel
- Title=Clean
If you are thinking of moving inventory the cars earn more in:

- States=ak, mt, wa, co, ca...
Perhaps you can move cars within thes positive feature regsion to maximise sale price

### Specific Features People DO NOT Value

<table>
<tr><td><a href="./analysis_results/module_11_01.step04.evaluation.neg.coeff.png" target="_blank"><img src="./analysis_results/module_11_01.step04.evaluation.neg.coeff.png"/></a></td></tr>
</table>

Do NOT Prioritize:

- Manufacturers=Dodge, Kia, Nissan, Mitsubishi,  for the common nbrands
- Type=Sedan, Hatchback, SUV, Wagon
- Size=Compact, Sub-Compact
- Drive=FWD
- Cylinders=4 and lower
If you are thinking of moving inventory the cars earn less in:

- States=me, fl, ny, nh, il, ....
Perhaps you can move cars with positive features to better performing regsions

Using the charts above to make decisions about wheat types of features about the vehicly to prioritize in your inventory
