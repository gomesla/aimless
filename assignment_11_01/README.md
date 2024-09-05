In this application, you will explore a dataset from kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

To frame the task, throughout these practical applications we will refer back to a standard process in industry for data projects called CRISP-DM. This process provides a framework for working through a data problem. Your first step in this application will be to read through a brief overview of CRISP-DM Links to an external site..

Data:
You will work with a used cars dataset Links to an external site.for this assignment.

Deliverables:
After understanding, preparing, and modeling your data, write up a basic report that details your primary findings. Your audience for this report is a group of used car dealers interested in fine-tuning their inventory.

# What drives the price of a car?

## Analysis
Source: [Data Analysis Workbook](./used_car_price_analysis.ipynb)
<table>
    <tr>
        <th align="left">Business Understanding</th>
    </tr>
    <tr>
        <td><p>We are provided with a dataset of used car prices and features about that particular vehicle. Our final goal will be to identify which 'features' AND what values of those features most contribute to the final price both positively and negatively.</p>
            <p>We will make use of transformers to get the data ready for use in regularization, we will then try out multiple linear regression models to find the best one. Once done we will use the coefficients to identoy how features contribute to price.</p>
            <p>Once we have found these imortant features we will write up actionable guidance for used car business based on the results of the analysis</p>
            <p><b>Steps involved:</b></p>
            <ul>
                <li>Examine the raw data and identify characterisitics of the data e.g. missing valuesm unique counts, invalid data</li>
                <li>Preprocess the data to get it ready for modelling by:
                    <ul>
                        <li>Identifying which features can be ignored</li>
                        <li>Identify missing features and either:</li>
                        <ul>
                            <li>Impute missing features</li>
                            <li>Drop Rows</li>
                            <li>Ignore feature for modelling</li>
                        </ul>
                    </ul>
                </li>
                <li>Decide what data transforms/normalization are required for numeric and categorical fields based on above decisions</li>
                <li>Use regularization techniques with multiple (L1, L2,...) linear regression models using and find one with the best peformance for predicting prices</li>
                <li>Repeat steps above if necessary to arrive at final 'best' model which we will measure by using the one with the lowest Mean Square Error (MSE)</li>
                <li>Analyse the most important features 'selected' by the model based on the coefficients determined by the previous steps</li>
            </ul>
    </tr>
</table>

## Data Understanding
<table>
    <tr>
        <th>Visual</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>
            <pre>
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
            </pre>
        </td>
        <td>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Notes</th>
                    <th>Decision</th>
                    <th>Transform</th>
                </tr>
                <tr>
                    <td>id</td>
                    <td>Not useful for predictions</td>
                    <td>Drop</td>
                </tr>
                <tr>
                    <td>region</td>
                    <td>Not really something dealer can control but can extract some useful information from this for other business decision making</td>
                    <td>Keep</td>
                    <td>One Hot</td>
                </tr>
                <tr>
                    <td>price</td>
                    <td>Target Field</td>
                    <td>Keep</td>
                    <td></td>
                </tr>
            </table>
        </td>
    </tr>
</table>