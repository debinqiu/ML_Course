##Identify risky bank loans using random forest ##

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
