# Identify risky bank loans using decision tree #

This post intends to finish the first assignment of course “Machine Learning for Data Analysis” on coursera.  We are assigned to perform a decision tree analysis to test nonlinear relationships among a series of explanatory variables and a binary, categorical response variable.

In this assignment, I used the finicial data which contains information on loans obtained from a credit agency in Germany. This data can be available from UCI Machine Learning Data Repository on https://archive.ics.uci.edu/ml. There are total 1000 observations in this data. The target is **default** which is a binary variable: 'yes' and 'no', meaning whether the loan goes into default. The explanatory variables consist of the following 16 components: &quot;checking_balance&quot;, &quot;months_loan_duration&quot;, &quot;credit_history&quot;,       &quot;purpose&quot;, &quot;amount&quot;, &quot;savings_balance&quot;, &quot;employment_duration&quot;, &quot;percent_of_income&quot;, &quot;years_at_residence&quot;, &quot;age&quot;, &quot;other_credit&quot;, &quot;housing&quot;, &quot;existing_loans_count&quot;, &quot;job&quot;, &quot;dependents&quot;, &quot;phone&quot;.

To identify the risky bank loans, we build a decision tree model using different programming languages such as SAS, Python and R.

##1. Fit the decision tree in SAS##
The decision tree is conducted by PROC HPSPLIT in SAS. To build the decision tree on training and testing data, we first randomly shuffle the original data and select the first 700 observations as training data and the rest as testing data. 
The SAS code is as follows.
```
TITLE 'Import credit.csv data';
FILENAME CSV "/home/debinqiu0/Practice/credit.csv" TERMSTR = CRLF;
PROC IMPORT DATAFILE = CSV OUT = credit DBMS = CSV REPLACE;
RUN;

PROC PRINT DATA = credit(OBS = 10); 
RUN;

TITLE 'Create training and testing data respectively by randomly shuffling strategy';
PROC SQL;
CREATE TABLE credit AS
SELECT * FROM credit
ORDER BY ranuni(0)
;
RUN;

TITLE 'Training data with 700 observations';
DATA credit_train;
SET credit;
IF _N_ <= 700 THEN OUTPUT;
RUN;

TITLE 'Testing data with 300 observations';
DATA credit_test;
SET credit;
IF _N_ > 700 THEN OUTPUT;
RUN;

ODS GRAPHICS ON;

PROC HPSPLIT DATA = credit_train SEED = 123;
TITLE 'Decision tree for credit training data';
CLASS checking_balance credit_history purpose savings_balance 
	  employment_duration other_credit housing job phone default;
MODEL default(event = 'yes') = checking_balance months_loan_duration  
		credit_history	purpose amount savings_balance employment_duration
	   	percent_of_income years_at_residence age other_credit 
	   	housing existing_loans_count job dependents phone default;
GROW ENTROPY;
PRUNE COSTCOMPLEXITY;
CODE FILE = '/home/debinqiu0/Practice/dt_credit.sas';
RUN;

TITLE 'Predictions on credit testing data';
DATA credit_pred(KEEP = Actual Predicted);
SET credit_test END = EOF;
%INCLUDE "/home/debinqiu0/Practice/dt_credit.sas";
Actual = default;
Predicted = (P_defaultyes >= 0.5);
run;

TITLE "Confusion Matrix Based on Cutoff Value of 0.5";
PROC FREQ DATA = credit_pred;
TABLES Actual*Predicted /norow nocol nopct;
RUN;
```
![cost_comp_sas](https://cloud.githubusercontent.com/assets/16762941/12804239/5286cfb8-cabf-11e5-9aee-8a490e5bbf1a.png)
