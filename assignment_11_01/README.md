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

Because the data has high dimensionality and many categorical fields we will need to make use of transformers to get the data represented as numerical features, ready for use in regularization. Once data is cleaned and prepared we will then try out multiple linear regression models to find the best one. Once done we will use the coefficients to identify how features contribute to price.

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
<td>Step5</td><td><ul><li>Drop rows not meeting the critera "price > 0"</li>
<li>Drop rows not meeting the critera "odometer > 0"</li>
<li>Drop rows not meeting the critera "year > 1900"</ul></td></tr>
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

### Field Correlations

To get a better idea of how the variables/features (dependent and indepenednt) correlate to each other we will generate a heatmap.

<table>
<tr>
<th>Visual</th>
<th>Notes</th>
</tr>
<tr>
<th valign="top"><a href="./analysis_results/module_11_01.step02.data_preparation.raw.data.heatmap.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.raw.data.heatmap.png"/></a></th>
</tr>
<tr>
<th valign="top"><a href="./analysis_results/module_11_01.step02.data_preparation.preProcessed.data.heatmap.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.preProcessed.data.heatmap.png"/></a></th>
</tr>
<tr>
<th valign="top"><a href="./analysis_results/module_11_01.step02.data_preparation.final.data.heatmap.png" target="_blank"><img src="./analysis_results/module_11_01.step02.data_preparation.final.data.heatmap.png"/></a></th>
<th valign="top">Looking at just correlations above 0.3 we see:

Postive Correlations:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pair</th>
      <th>correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>(type_unknown, size_unknown)</td>
      <td>0.905622</td>
    </tr>
    <tr>
      <th>58</th>
      <td>(encoded_cleaned_model, price)</td>
      <td>0.729515</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(transmission_other, condition_good)</td>
      <td>0.608644</td>
    </tr>
    <tr>
      <th>0</th>
      <td>(year, price)</td>
      <td>0.602532</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(fuel_electric, manufacturer_tesla)</td>
      <td>0.586166</td>
    </tr>
    <tr>
      <th>66</th>
      <td>(encoded_cleaned_model, transmission_other)</td>
      <td>0.466898</td>
    </tr>
    <tr>
      <th>25</th>
      <td>(drive_fwd, cylinders_4 cylinders)</td>
      <td>0.434676</td>
    </tr>
    <tr>
      <th>45</th>
      <td>(type_sedan, drive_fwd)</td>
      <td>0.397905</td>
    </tr>
    <tr>
      <th>30</th>
      <td>(size_full-size, cylinders_8 cylinders)</td>
      <td>0.392260</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(fuel_electric, cylinders_other)</td>
      <td>0.389180</td>
    </tr>
    <tr>
      <th>26</th>
      <td>(drive_unknown, cylinders_unknown)</td>
      <td>0.388893</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(transmission_automatic, odometer)</td>
      <td>0.371882</td>
    </tr>
    <tr>
      <th>21</th>
      <td>(transmission_other, fuel_other)</td>
      <td>0.359839</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(transmission_other, price)</td>
      <td>0.358747</td>
    </tr>
    <tr>
      <th>59</th>
      <td>(encoded_cleaned_model, year)</td>
      <td>0.357198</td>
    </tr>
    <tr>
      <th>20</th>
      <td>(transmission_other, cylinders_unknown)</td>
      <td>0.352898</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(transmission_other, year)</td>
      <td>0.352314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(cylinders_12 cylinders, manufacturer_aston-martin)</td>
      <td>0.350461</td>
    </tr>
    <tr>
      <th>70</th>
      <td>(encoded_cleaned_model, type_pickup)</td>
      <td>0.340703</td>
    </tr>
    <tr>
      <th>62</th>
      <td>(encoded_cleaned_model, cylinders_8 cylinders)</td>
      <td>0.332079</td>
    </tr>
    <tr>
      <th>49</th>
      <td>(type_truck, cylinders_8 cylinders)</td>
      <td>0.331050</td>
    </tr>
    <tr>
      <th>61</th>
      <td>(encoded_cleaned_model, condition_good)</td>
      <td>0.329892</td>
    </tr>
    <tr>
      <th>34</th>
      <td>(size_unknown, drive_unknown)</td>
      <td>0.328905</td>
    </tr>
    <tr>
      <th>54</th>
      <td>(type_unknown, drive_unknown)</td>
      <td>0.328200</td>
    </tr>
    <tr>
      <th>44</th>
      <td>(type_sedan, cylinders_4 cylinders)</td>
      <td>0.307722</td>
    </tr>
    <tr>
      <th>41</th>
      <td>(type_pickup, cylinders_8 cylinders)</td>
      <td>0.306310</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(cylinders_5 cylinders, manufacturer_volvo)</td>
      <td>0.302344</td>
    </tr>
    <tr>
      <th>47</th>
      <td>(type_suv, drive_4wd)</td>
      <td>0.299876</td>
    </tr>
    <tr>
      <th>42</th>
      <td>(type_pickup, drive_4wd)</td>
      <td>0.299305</td>
    </tr>
    <tr>
      <th>28</th>
      <td>(size_compact, cylinders_4 cylinders)</td>
      <td>0.291356</td>
    </tr>
    <tr>
      <th>46</th>
      <td>(type_suv, manufacturer_jeep)</td>
      <td>0.288057</td>
    </tr>
    <tr>
      <th>31</th>
      <td>(size_full-size, drive_4wd)</td>
      <td>0.284774</td>
    </tr>
    <tr>
      <th>43</th>
      <td>(type_pickup, size_full-size)</td>
      <td>0.281581</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(transmission_automatic, condition_unknown)</td>
      <td>0.280951</td>
    </tr>
    <tr>
      <th>35</th>
      <td>(type_coupe, drive_rwd)</td>
      <td>0.274350</td>
    </tr>
    <tr>
      <th>52</th>
      <td>(type_truck, size_full-size)</td>
      <td>0.271703</td>
    </tr>
    <tr>
      <th>33</th>
      <td>(size_unknown, cylinders_unknown)</td>
      <td>0.271700</td>
    </tr>
    <tr>
      <th>57</th>
      <td>(encoded_region, price)</td>
      <td>0.270704</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(fuel_other, condition_good)</td>
      <td>0.270273</td>
    </tr>
    <tr>
      <th>50</th>
      <td>(type_truck, fuel_diesel)</td>
      <td>0.267547</td>
    </tr>
    <tr>
      <th>39</th>
      <td>(type_pickup, price)</td>
      <td>0.264162</td>
    </tr>
    <tr>
      <th>67</th>
      <td>(encoded_cleaned_model, drive_4wd)</td>
      <td>0.262205</td>
    </tr>
    <tr>
      <th>24</th>
      <td>(drive_4wd, cylinders_8 cylinders)</td>
      <td>0.261525</td>
    </tr>
    <tr>
      <th>53</th>
      <td>(type_unknown, cylinders_unknown)</td>
      <td>0.259580</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(cylinders_unknown, condition_good)</td>
      <td>0.257593</td>
    </tr>
    <tr>
      <th>69</th>
      <td>(encoded_cleaned_model, size_full-size)</td>
      <td>0.254064</td>
    </tr>
    <tr>
      <th>37</th>
      <td>(type_hatchback, size_compact)</td>
      <td>0.248381</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(cylinders_8 cylinders, manufacturer_chevrolet)</td>
      <td>0.247758</td>
    </tr>
    <tr>
      <th>56</th>
      <td>(type_wagon, manufacturer_subaru)</td>
      <td>0.242022</td>
    </tr>
    <tr>
      <th>48</th>
      <td>(type_truck, manufacturer_ram)</td>
      <td>0.238994</td>
    </tr>
    <tr>
      <th>63</th>
      <td>(encoded_cleaned_model, cylinders_unknown)</td>
      <td>0.236536</td>
    </tr>
    <tr>
      <th>71</th>
      <td>(encoded_cleaned_model, type_truck)</td>
      <td>0.235072</td>
    </tr>
    <tr>
      <th>38</th>
      <td>(type_other, transmission_other)</td>
      <td>0.233567</td>
    </tr>
    <tr>
      <th>64</th>
      <td>(encoded_cleaned_model, fuel_diesel)</td>
      <td>0.231526</td>
    </tr>
    <tr>
      <th>51</th>
      <td>(type_truck, drive_4wd)</td>
      <td>0.228739</td>
    </tr>
    <tr>
      <th>65</th>
      <td>(encoded_cleaned_model, fuel_other)</td>
      <td>0.228293</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(fuel_diesel, manufacturer_ram)</td>
      <td>0.226269</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(cylinders_unknown, year)</td>
      <td>0.223277</td>
    </tr>
    <tr>
      <th>22</th>
      <td>(drive_4wd, price)</td>
      <td>0.222485</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(cylinders_8 cylinders, price)</td>
      <td>0.221963</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(fuel_diesel, price)</td>
      <td>0.217940</td>
    </tr>
    <tr>
      <th>60</th>
      <td>(encoded_cleaned_model, manufacturer_ram)</td>
      <td>0.217676</td>
    </tr>
    <tr>
      <th>72</th>
      <td>(encoded_cleaned_model, encoded_region)</td>
      <td>0.217430</td>
    </tr>
    <tr>
      <th>40</th>
      <td>(type_pickup, manufacturer_ram)</td>
      <td>0.217240</td>
    </tr>
    <tr>
      <th>32</th>
      <td>(size_mid-size, cylinders_4 cylinders)</td>
      <td>0.212842</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(cylinders_unknown, price)</td>
      <td>0.212551</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(drive_4wd, manufacturer_jeep)</td>
      <td>0.211818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(cylinders_5 cylinders, manufacturer_volkswagen)</td>
      <td>0.205140</td>
    </tr>
    <tr>
      <th>68</th>
      <td>(encoded_cleaned_model, drive_unknown)</td>
      <td>0.203797</td>
    </tr>
    <tr>
      <th>27</th>
      <td>(drive_unknown, transmission_other)</td>
      <td>0.203666</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(cylinders_other, manufacturer_tesla)</td>
      <td>0.203160</td>
    </tr>
    <tr>
      <th>29</th>
      <td>(size_compact, drive_fwd)</td>
      <td>0.202905</td>
    </tr>
    <tr>
      <th>36</th>
      <td>(type_hatchback, drive_fwd)</td>
      <td>0.202477</td>
    </tr>
  </tbody>
</table>

Negative Correlations:<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pair</th>
      <th>correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31</th>
      <td>(transmission_other, transmission_automatic)</td>
      <td>-0.857166</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(fuel_other, fuel_gas)</td>
      <td>-0.679696</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(odometer, year)</td>
      <td>-0.676116</td>
    </tr>
    <tr>
      <th>44</th>
      <td>(size_mid-size, size_full-size)</td>
      <td>-0.671203</td>
    </tr>
    <tr>
      <th>35</th>
      <td>(drive_fwd, drive_4wd)</td>
      <td>-0.649229</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(title_status_rebuilt, title_status_clean)</td>
      <td>-0.594905</td>
    </tr>
    <tr>
      <th>20</th>
      <td>(title_status_unknown, title_status_clean)</td>
      <td>-0.592336</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(fuel_gas, fuel_diesel)</td>
      <td>-0.552362</td>
    </tr>
    <tr>
      <th>0</th>
      <td>(odometer, price)</td>
      <td>-0.547656</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(transmission_automatic, condition_good)</td>
      <td>-0.527686</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(condition_unknown, condition_good)</td>
      <td>-0.527258</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(cylinders_6 cylinders, cylinders_4 cylinders)</td>
      <td>-0.502511</td>
    </tr>
    <tr>
      <th>60</th>
      <td>(encoded_cleaned_model, drive_fwd)</td>
      <td>-0.472648</td>
    </tr>
    <tr>
      <th>57</th>
      <td>(encoded_cleaned_model, cylinders_4 cylinders)</td>
      <td>-0.455126</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(condition_unknown, condition_excellent)</td>
      <td>-0.444875</td>
    </tr>
    <tr>
      <th>27</th>
      <td>(transmission_other, odometer)</td>
      <td>-0.438511</td>
    </tr>
    <tr>
      <th>41</th>
      <td>(size_full-size, size_compact)</td>
      <td>-0.433855</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(title_status_salvage, title_status_clean)</td>
      <td>-0.423581</td>
    </tr>
    <tr>
      <th>39</th>
      <td>(size_full-size, cylinders_4 cylinders)</td>
      <td>-0.408740</td>
    </tr>
    <tr>
      <th>26</th>
      <td>(transmission_manual, transmission_automatic)</td>
      <td>-0.407041</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(cylinders_8 cylinders, cylinders_6 cylinders)</td>
      <td>-0.397851</td>
    </tr>
    <tr>
      <th>59</th>
      <td>(encoded_cleaned_model, transmission_automatic)</td>
      <td>-0.392093</td>
    </tr>
    <tr>
      <th>50</th>
      <td>(type_sedan, drive_4wd)</td>
      <td>-0.389559</td>
    </tr>
    <tr>
      <th>36</th>
      <td>(drive_rwd, drive_4wd)</td>
      <td>-0.376548</td>
    </tr>
    <tr>
      <th>34</th>
      <td>(drive_fwd, cylinders_8 cylinders)</td>
      <td>-0.376351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(condition_good, condition_excellent)</td>
      <td>-0.376321</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(cylinders_8 cylinders, cylinders_4 cylinders)</td>
      <td>-0.371407</td>
    </tr>
    <tr>
      <th>33</th>
      <td>(drive_fwd, price)</td>
      <td>-0.356622</td>
    </tr>
    <tr>
      <th>53</th>
      <td>(type_suv, type_sedan)</td>
      <td>-0.349783</td>
    </tr>
    <tr>
      <th>61</th>
      <td>(encoded_cleaned_model, type_sedan)</td>
      <td>-0.343515</td>
    </tr>
    <tr>
      <th>56</th>
      <td>(encoded_cleaned_model, odometer)</td>
      <td>-0.341687</td>
    </tr>
    <tr>
      <th>37</th>
      <td>(drive_rwd, drive_fwd)</td>
      <td>-0.333011</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(cylinders_4 cylinders, price)</td>
      <td>-0.322138</td>
    </tr>
    <tr>
      <th>29</th>
      <td>(transmission_other, condition_unknown)</td>
      <td>-0.317907</td>
    </tr>
    <tr>
      <th>58</th>
      <td>(encoded_cleaned_model, fuel_gas)</td>
      <td>-0.303658</td>
    </tr>
    <tr>
      <th>25</th>
      <td>(transmission_automatic, fuel_other)</td>
      <td>-0.300909</td>
    </tr>
    <tr>
      <th>24</th>
      <td>(transmission_automatic, cylinders_unknown)</td>
      <td>-0.298857</td>
    </tr>
    <tr>
      <th>21</th>
      <td>(transmission_automatic, price)</td>
      <td>-0.289433</td>
    </tr>
    <tr>
      <th>43</th>
      <td>(size_mid-size, size_compact)</td>
      <td>-0.283488</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(fuel_hybrid, fuel_gas)</td>
      <td>-0.282553</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(fuel_gas, price)</td>
      <td>-0.269519</td>
    </tr>
    <tr>
      <th>42</th>
      <td>(size_mid-size, cylinders_8 cylinders)</td>
      <td>-0.263274</td>
    </tr>
    <tr>
      <th>40</th>
      <td>(size_full-size, drive_fwd)</td>
      <td>-0.258092</td>
    </tr>
    <tr>
      <th>47</th>
      <td>(type_pickup, drive_fwd)</td>
      <td>-0.256023</td>
    </tr>
    <tr>
      <th>48</th>
      <td>(type_sedan, price)</td>
      <td>-0.255789</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(title_status_lien, title_status_clean)</td>
      <td>-0.253970</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(cylinders_unknown, odometer)</td>
      <td>-0.252266</td>
    </tr>
    <tr>
      <th>22</th>
      <td>(transmission_automatic, year)</td>
      <td>-0.248795</td>
    </tr>
    <tr>
      <th>49</th>
      <td>(type_sedan, cylinders_8 cylinders)</td>
      <td>-0.244575</td>
    </tr>
    <tr>
      <th>55</th>
      <td>(type_truck, drive_fwd)</td>
      <td>-0.240123</td>
    </tr>
    <tr>
      <th>46</th>
      <td>(type_pickup, cylinders_4 cylinders)</td>
      <td>-0.233278</td>
    </tr>
    <tr>
      <th>51</th>
      <td>(type_sedan, type_pickup)</td>
      <td>-0.230480</td>
    </tr>
    <tr>
      <th>32</th>
      <td>(drive_4wd, cylinders_4 cylinders)</td>
      <td>-0.227830</td>
    </tr>
    <tr>
      <th>30</th>
      <td>(transmission_other, fuel_gas)</td>
      <td>-0.227347</td>
    </tr>
    <tr>
      <th>28</th>
      <td>(transmission_other, condition_excellent)</td>
      <td>-0.227212</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(cylinders_unknown, cylinders_6 cylinders)</td>
      <td>-0.226550</td>
    </tr>
    <tr>
      <th>52</th>
      <td>(type_suv, type_pickup)</td>
      <td>-0.215330</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(cylinders_unknown, cylinders_4 cylinders)</td>
      <td>-0.211492</td>
    </tr>
    <tr>
      <th>38</th>
      <td>(size_compact, drive_4wd)</td>
      <td>-0.211023</td>
    </tr>
    <tr>
      <th>45</th>
      <td>(type_other, transmission_automatic)</td>
      <td>-0.208270</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(condition_good, odometer)</td>
      <td>-0.207734</td>
    </tr>
    <tr>
      <th>54</th>
      <td>(type_truck, cylinders_4 cylinders)</td>
      <td>-0.202070</td>
    </tr>
  </tbody>
</table>

</th>
</tr>
</table>

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

- Manufacturers=Toyota, Honda, Lexus, Tesla for the commonn brands
- Type=Pickups, Convertibles, Coupes and Trucks
- Size=Full Size, Mid Size
- Drive=4WD
- Cylinders=8
- Transmission=Manual
- Fuel=Diesel
- Title=Clean

This means move these to the front of your lot and they may benefit from a markup or marketing

If you are thinking of moving inventory the cars earn more in:

- States=ak, mt, wa, co, ca...
Perhaps you can move cars within these positive feature regions to maximize sale price

### Specific Features People DO NOT Value

<table>
<tr><td><a href="./analysis_results/module_11_01.step04.evaluation.neg.coeff.png" target="_blank"><img src="./analysis_results/module_11_01.step04.evaluation.neg.coeff.png"/></a></td></tr>
</table>

Do NOT Prioritize:

- Manufacturers=Dodge, Kia, Nissan, Mitsubishi,  for the commonn brands
- Type=Sedan, Hatchback, SUV, Wagon
- Size=Compact, Sub-Compact
- Drive=FWD
- Cylinders=4 and lower

This means move these to the back of your lot it may be worthwhile getting rid of these quickly from your lot

If you are thinking of moving inventory the cars earn less in:

- States=me, fl, ny, nh, il, ....
Perhaps you can move cars with negative features to better performing regions to benefit from the markup

Using the charts above to make decisions about what types of features about the vehicle to prioritize in your inventory

