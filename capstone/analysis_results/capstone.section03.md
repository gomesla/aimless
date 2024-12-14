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

