# Identify risky bank loans using decision tree #

This post intends to finish the first assignment of course “Machine Learning for Data Analysis” on coursera.  We are assigned to perform a decision tree analysis to test nonlinear relationships among a series of explanatory variables and a binary, categorical response variable.

In this assignment, I used the finicial data which contains information on loans obtained from a credit agency in Germany. This data can be available from UCI Machine Learning Data Repository on https://archive.ics.uci.edu/ml. There are total 1000 observations in this data. The target is **default** which is a binary variable: 'yes' and 'no', meaning whether the loan goes into default. The explanatory variables consist of the following 16 components: &quot;checking_balance&quot;, &quot;months_loan_duration&quot;, &quot;credit_history&quot;,       &quot;purpose&quot;, &quot;amount&quot;, &quot;savings_balance&quot;, &quot;employment_duration&quot;, &quot;percent_of_income&quot;, &quot;years_at_residence&quot;, &quot;age&quot;, &quot;other_credit&quot;, &quot;housing&quot;, &quot;existing_loans_count&quot;, &quot;job&quot;, &quot;dependents&quot;, &quot;phone&quot;.

To identify the risky bank loans, we build a decision tree model using different programming languages such as SAS, Python and R.

##1. Fit decision tree in SAS##
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

The trend of cost complexity shows that the smallest average ASE (0.176) obtains at cost complexity parameter = 0.0068. Let's look at the graph of fitted tree as follows. We can see that the most four important features are checking_balance, month_loan_duration, credit_history and savings_balance. 

![tree_sas](https://cloud.githubusercontent.com/assets/16762941/12804299/fdf6fecc-cabf-11e5-956f-23c8575640e3.png)

Finally, let's check the accuracy of the fitted decision tree on testing data. The confusion matrix gives us the accuracy of 59% which is somewhat low. However, the result can be improved by using random forest or gradient boosting that will be covered in the latter course.

![conf_mat_sas](https://cloud.githubusercontent.com/assets/16762941/12804393/da77c714-cac0-11e5-85e4-71659664e8a7.png)


## 2. Fit decision tree in Python ##
Python `sklearn` package provides numerous functions to perform machine learning methods, including decision tree. We now give the Python code to fit the decision tree for bank loan data. 

```Python
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
os.chdir('/Users/Deman/Desktop/Dropbox/ML')

credit = pd.read_csv("credit.csv")

credit = credit.dropna()
targets = LabelEncoder().fit_transform(credit['default'])

predictors = credit.ix[:,credit.columns != 'default']

# Recode categorical variables as numeric variables
for i in range(0,len(predictors.dtypes)):
    if predictors.dtypes[i] != 'int64':
        predictors[predictors.columns[i]] = LabelEncoder().fit_transform(predictors[predictors.columns[i]])

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

#Build model on training data
classifier = DecisionTreeClassifier().fit(pred_train,tar_train)
predictions = classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
```
Since Python does not provide pruing on the decision tree, the classification accuracy (69%) may be higher than that from SAS. Also, it results in a large tree shown in the following graph and overfitting.

```Python
>>> sklearn.metrics.confusion_matrix(tar_test,predictions)
Out[2]: 
array([[220,  66],
       [ 58,  56]])

>>> sklearn.metrics.accuracy_score(tar_test, predictions)
Out[3]: 0.68999999999999995
```

![tree_python](https://cloud.githubusercontent.com/assets/16762941/12804810/397d63c4-cac4-11e5-8a9e-259b8f45391a.png)

## 3. Fit decision tree in R ##
We finally build a decision tree in R using `rpart` package. In fact, there are several other packages such as `tree`, `C5.0` to fit such model. Here we only use `rpart` package for simplicity. The R code is as follows.
```
> credit <- read.csv("credit.csv")
> #Split into training and testing sets
> set.seed(123)
> train_sample <- sample(1000,700)
> credit_train <- credit[train_sample,]
> credit_test <- credit[-train_sample,]
> X_train <- credit_train[-c(which(colnames(credit) %in% 'default'))]
> X_test <- credit_test[-c(which(colnames(credit) %in% 'default'))]
> 
> # Build model on training data
> library(rpart)
> credit_model <- rpart(default~.,data = credit_train)
> 
> # Make predictions on testing data
> credit_pred <- predict(credit_model,X_test, type = 'class')
> # accuracy
> sum(diag(table(credit_test$default,credit_pred)))/300
[1] 0.74
> 
> # Displaying the decision tree
> library(rpart.plot)
> rpart.plot(credit_model,under = TRUE,faclen = 3,extra = 106)
```
The fitted model gives an accuracy of 74% on testing data with 300 observations, which is higher than those obtained from SAS and Python. Also, we also have the following graph of tree, which is simpler than that from Python but a bit more complex than that from SAS, but the frist four important features are the same as those fitted by SAS.

