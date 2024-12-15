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
      <td>account locked re locked thank you for working best regards sent locked importance high dear can you please help colleague able log his computer after changing password thank you very much best regards</td>
      <td>Access</td>
    </tr>
    <tr>
      <td>confluence user confluence user hello name confluence thank</td>
      <td>Access</td>
    </tr>
    <tr>
      <td>windows upgrade upgrade dear after upgrading able connect via have attached logs app can you please help thanks senior software developer</td>
      <td>Administrative rights</td>
    </tr>
    <tr>
      <td>upgrade friday october pm upgrade hello followed instructions package install software center please advise thanks october upgrade hello everyone kind reminder please make update latest version assets highly critical perform upgrade by starting next upgrade pushed assets out date find detailed instructions how perform upgrade please note during update process lose approximately minutes questions encounter difficulties upgrading please hesitate thank kind regards ext october upgrade hello everyone please advised update version version assets highly critical perform upgrade during two weeks based availability order avoid virus attacks latest virus database available order update please follow steps more detailed guide found software center by searching software center search menu best endpoint agent install selected wait installation status becomes installed reboot computer icons newly installed endpoint tools note during update process lose approximately minutes questions encounter difficulties upgrading please hesitate thank kind regards ext hub</td>
      <td>Administrative rights</td>
    </tr>
    <tr>
      <td>oracle error when trying to log time on day th of error when trying log th hi guys getting error when trying log th you can create card only between start date end date dec can you please help out with thanks</td>
      <td>HR Support</td>
    </tr>
    <tr>
      <td>new pas close close dear ask please close opportunity active more date possible thank officer</td>
      <td>HR Support</td>
    </tr>
    <tr>
      <td>mail friday pm se la issue si la va similar cum se fond lead</td>
      <td>Hardware</td>
    </tr>
    <tr>
      <td>incident regarding installed programs and windows start incident hello please be advised today had incident with laptop none programs were installed anything start thank you applications engineer en</td>
      <td>Hardware</td>
    </tr>
    <tr>
      <td>project code delete thursday code delete hi please code thank</td>
      <td>Internal Project</td>
    </tr>
    <tr>
      <td>new project code pas tuesday pm setup has assigned hi please assign kind regards pm setup has assigned hello please advised record number has assigned please review details take appropriate action reference number details assign reference assignment summary null choose template applicable setup please provide short description setup details please attach completed setup form link forms policies forms policies procedures priority affected add requested location center location center please link kind regards ref msg</td>
      <td>Internal Project</td>
    </tr>
    <tr>
      <td>oracle project management change for thursday pm change hi given please thanks</td>
      <td>Miscellaneous</td>
    </tr>
    <tr>
      <td>stuck oracle stuck hi guys for some strange reason we cannot find process does show up processing queue although fully approved could you please kindly check what happened transaction advise hi could you please kindly log ticket thanks kind regards manager</td>
      <td>Miscellaneous</td>
    </tr>
    <tr>
      <td>thursday pm hey folks keyboard kb has please log installed please perform purchase order how presentation thanks administrator phone</td>
      <td>Purchase</td>
    </tr>
    <tr>
      <td>new purchase po wednesday march pm purchase po dear purchased keyboard please log installation please take consideration mandatory receipts section order receive item ordered how video kind regards administrator</td>
      <td>Purchase</td>
    </tr>
    <tr>
      <td>mailbox full sent saturday full hello full even if space does gets freed</td>
      <td>Storage</td>
    </tr>
    <tr>
      <td>mailbox com wednesday october pm mailbox hi please assist setting dedicated mailbox main managing employer initiatives besides mailbox thank help regards digital communications</td>
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

