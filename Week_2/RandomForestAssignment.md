#Identify risky bank loans using random forest #

In this assignment, we use the same dataset which contains information on loans obtained from a credit agency in Germany to perform the random forest in order to identify risky bank loans. There are total 1000 observations and 17 features in this dataset. The **target** is "default" which is a binary variable: 'yes' and 'no', meaning whether the loan went into default. 

The **explanatory variables** consist of the following 16 components:
- checking_balance (categorical)  : "< 0 DM",     "> 200 DM",   "1 - 200 DM", "unknown"
- months_loan_duration (interval) : 4 - 72
- credit_history (categorical)    : "critical",  "good",   "perfect",   "poor",      "very good"
- purpose  (categorical)          : "business",   "car",   "car0",      "education",  "furniture/appliances", "renovations" 
- amount   (interval)             : 250 - 18424
- savings_balance (categorical)   : "< 100 DM",     "> 1000 DM",     "100 - 500 DM",  "500 - 1000 DM", "unknown" 
- employment_duration (categorical): "< 1 year",    "> 7 years",   "1 - 4 years", "4 - 7 years", "unemployed" 
- percent_of_income (interval)    : 1 - 4
- years_at_residence (interval)   : 1 - 4
- age                (interval)   : 19 - 75
- other_credit     (categorical)  :  "bank",  "none",  "store"
- housing          (categorical)  : "other", "own",   "rent"
- existing_loans_count (interval) : 1 - 4
- job             (categorical)   : "management", "skilled",    "unemployed", "unskilled"
- dependents   (interval)         : 1 - 2
- phone        (categorical)      : "no",  "yes"

The dataset can be downloaded [here] (https://github.com/debinqiu/ML_Course/files/116846/credit.txt) and more information about this dataset is available on [UCI Machine Learning Data Repository] (https://archive.ics.uci.edu/ml).  

To identify the risky bank loans, we build a random forest model using different programming languages in SAS, Python, R and Spark.

## Run random forest in SAS ##
We use HPFOREST procedure in SAS to run the random forest which builds many decision trees ranther than a single decision tree in order to improve the accuracy of prediction. In the fitting procedure, we first randomly split the entire data into training set with 700 observations and testing set with 300 observations. We then run the random forest on the training data and make predictions on testing data using HP4SCORE procedure. 
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
PROC HPFOREST DATA = credit_train;
TITLE 'Random forest for credit training data';
TARGET default/LEVEL = BINARY;
INPUT checking_balance credit_history purpose savings_balance 
	  employment_duration other_credit housing job/ LEVEL= NOMINAL;
INPUT phone/LEVEL=BINARY;
INPUT months_loan_duration amount percent_of_income years_at_residence 
	  age existing_loans_count dependents/LEVEL = INTERVAL;
SAVE FILE = '/home/debinqiu0/Practice/rf_credit.sas';
RUN;

PROC HP4SCORE DATA = credit_test;
TITLE 'Predictions on credit testing data';
ID default;
SCORE FILE = '/home/debinqiu0/Practice/rf_credit.sas' OUT = rfscore;
RUN;

TITLE "Confusion matrix for testing data";
PROC FREQ DATA = rfscore;
TABLES default*I_default /norow nocol nopct;
RUN;
```
We can see several outputs such as Model Information, Fit Statistics, Loss Reduction Variable Importance from HPFOREST procedure, but we more care about the predicted accuracy on the testing data. The confusion matrix on testing data below shows the fitted random forest correctly classifies 74.3% of the default loans, which is much better than that from a single decision fitted in the first assignment. Thus, random forest can improve the accuracy of decision tree dramatically in some cases. 

## Run random forest tree in R##
We now give the R code to run the random forest, which is realized using `randomForest` package. We use the same strategy as explained above. The code is as follows. 
```
> credit <- read.table("credit.txt",header = TRUE, sep = "\t")
> #Split into training and testing sets
> set.seed(123)
> train_sample <- sample(1000,700)
> credit_train <- credit[train_sample,]
> credit_test <- credit[-train_sample,]
> X_train <- credit_train[-c(which(colnames(credit) %in% 'default'))]
> X_test <- credit_test[-c(which(colnames(credit) %in% 'default'))]
> 
> # Build model on training data
> library(randomForest)
> credit_rf <- randomForest(default~.,data = credit_train)
> 
> # Make predictions on testing data
> credit_rf_pred <- predict(credit_rf,X_test)
> # confusion matrix and accuracy
> (conf_matrix <- table(credit_test$default,credit_rf_pred))
     credit_rf_pred
       no yes
  no  197  12
  yes  58  33
> (sum(diag(conf_matrix))/sum(conf_matrix))
[1] 0.7666667
> # importance of explanatory variables
> importance(credit_rf)
                     MeanDecreaseGini
checking_balance            35.270599
months_loan_duration        29.007441
credit_history              19.788039
purpose                     18.535077
amount                      43.913243
savings_balance             16.717235
employment_duration         19.956369
percent_of_income           14.372510
years_at_residence          13.564876
age                         33.709076
other_credit                 8.541563
housing                      9.169049
existing_loans_count         7.006525
job                         10.429602
dependents                   4.279386
phone                        5.359734
> varImpPlot(credit_rf)
> 
> # Running a different number of trees and see the effect
> # of that on the accuracy of the prediction
> ntree <- seq(50,1000,by = 100)
> accuracy <- numeric(length(ntree))
> set.seed(123)
> for (i in 1:length(ntree)) {
+   credit_rf <- randomForest(default~.,data = credit_train,ntree = ntree[i])
+   credit_rf_pred <- predict(credit_rf,X_test)
+   conf_matrix <- table(credit_test$default,credit_rf_pred)
+   accuracy[i] <- sum(diag(conf_matrix))/sum(conf_matrix)
+ }
> accuracy
 [1] 0.7400000 0.7633333 0.7500000 0.7600000 0.7566667 0.7600000 0.7566667 0.7666667 0.7600000 0.7566667
> max(accuracy)
[1] 0.7666667
> ntree[which.max(accuracy)]
[1] 750
> plot(ntree, accuracy, type = 'l', main = 'acuracy vs. ntree')
```
In addition, we give two extra results. The first result is the importance of explanatory variables. The function `varImpPlot` gives us the plot of important explanatory variabls which is shown as follows. We can see that the first three most important explanatory variables are amount, checking_balance and age. 

The second result is the accuracy versus different number of trees. The accuracy trend shown in the following graph indicates the highest accuracy 77% obtained at ntree = 750. In fact, when the ntree = 100, we can achieve 76.3% accuracy which is very close to 77% but the computation is less intensive in this case. 
