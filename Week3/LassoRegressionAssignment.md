#Predict a baseball player’s Salary#

In this assignment, we want to predict a baseball player's salary using linear regression, especially penalized linear regression such as Lasso regression, since there are 19 explanatory variables in the **Hitters** dataset which contains 322 observations. The **explanatory variables** in this dataset are listed as follows. 
- AtBat: Number of times at bat in 1986
- Hits : Number of hits in 1986
- HmRun: Number of home runs in 1986
- Runs : Number of runs in 1986
- RBI  : Number of runs batted in in 1986
- Walks: Number of walks in 1986
- Years: Number of years in the major leagues
- CAtBat: Number of times at bat during his career
- CHits: Number of hits during his career
- CHmRun: Number of home runs during his career
- CRuns: Number of runs during his career
- CRBI: Number of runs batted in during his career
- CWalks: Number of walks during his career
- League: A factor with levels A and N indicating player's league at the end of 1986
- Division: A factor with levels E and W indicating player's division at the end of 1986
- PutOuts: Number of put outs in 1986
- Assists: Number of assists in 1986
- Errors: Number of errors in 1986
- NewLeague: A factor with levels A and N indicating player's league at the beginning of 1987

The **response** is the player's salary.
- Salary: 1987 annual salary on opening day in thousands of dollars

To fit a linear regression, we first remove the 59 missing values in salary and transform the categorical variables into numeric variables.
```
TITLE 'Import Hitters.csv data';
FILENAME CSV "/home/debinqiu0/Practice/Hitters.csv" TERMSTR = CRLF;
PROC IMPORT DATAFILE = CSV OUT = Hitters DBMS = CSV REPLACE;
RUN;

TITLE 'Data cleaning and transformation';
DATA Hitters_New;
SET Hitters;
IF League = 'A' THEN League1 = 1;
IF League = 'N' THEN League1 = 2;
IF NewLeague = 'A' THEN NewLeague1 = 1;
IF NewLeague = 'N' THEN NewLeague1 = 2;
IF Division = 'E' THEN Division1 = 1;
IF Division = 'W' THEN Division1 = 2;
IF Salary = 'NA' THEN DELETE;
Salary1 = INPUT(Salary, comma6.);
DROP League NewLeague Division Salary;
RUN;
```
Secondly, we fit a classic linear regression to check whether we have to use a more complex fitting methodology (penalized linear regression) or not. The parameter estimates from classic linear regression below show that 13 explanatory variables are not relevant to the salary, so it is appropriate to perform the variable selection. 
```
TITLE 'Run a classic linear regression';
PROC REG DATA = Hitters_New;
MODEL Salary1 = AtBat Hits HmRun Runs RBI Walks Years 
               CAtBat CHits CHmRun CRuns CRBI CWalks 
               League1 Division1 PutOuts Assists Errors NewLeague1;
RUN;
```

Now we fit a Lasso regression on the Hitters data. We first split the entire dataset into training and testing data, respectively. 
```
TITLE 'Split into training and testing data';
PROC SURVEYSELECT DATA = Hitters_New OUT = traintest SEED = 123
SAMPRATE = 0.7 METHOD = SRS OUTALL;
RUN;
```
To fit the Lasso regression, we use the GLMSELECT procedure in SAS as follows. 
```
ODS GRAPHICS ON;
TITLE 'Run a Lasso regression';
PROC GLMSELECT DATA = traintest PLOTS = ALL SEED = 123;
PARTITION ROLE = SELECTED(train = '1' test = '0');
MODEL Salary1 = AtBat Hits HmRun Runs RBI Walks Years 
               CAtBat CHits CHmRun CRuns CRBI CWalks 
               League1 Division1 PutOuts Assists Errors 
               NewLeague1/SELECTION = LAR(CHOOSE = CV STOP = NONE)
               CVMETHOD = RANDOM(10);
RUN;
```
