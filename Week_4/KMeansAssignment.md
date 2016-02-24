# Find teen market segments using k-means clustering#

In this assignment, we are assigned to run k-means clustering analysis. For this analysis, we use a dataset which represents a random sample of 30,000 U.S. high school students who had profiles on a well-known SNS in 2006. The data was sampled from senior, junior, sophomore and freshman classes with graduation years from 2006 through 2009. The oirginal dataset contains 40 features, but we will use only 39 features in the clustering analysis by dropping 'gradyear' variable. Also, we will transform categorical variable 'gender' into numberic values. Other features are all numeric. See for the details of the dataset and features. All above steps are completed in the folliwing Python code. 
```python
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn import preprocessing
>>> from sklearn.cluster import KMeans

# import csv data
>>> snsdata = pd.read_csv("snsdata.csv")
# drop the missing data
>>> snsdata_clean = snsdata.dropna()
>>> snsdata_clean.dtypes
Out[6]: 
gradyear          int64
gender           object
age             float64
friends           int64
basketball        int64
football          int64
soccer            int64
softball          int64
volleyball        int64
swimming          int64
cheerleading      int64
baseball          int64
tennis            int64
sports            int64
cute              int64
sex               int64
sexy              int64
hot               int64
kissed            int64
dance             int64
band              int64
marching          int64
music             int64
rock              int64
god               int64
church            int64
jesus             int64
bible             int64
hair              int64
dress             int64
blonde            int64
mall              int64
shopping          int64
clothes           int64
hollister         int64
abercrombie       int64
die               int64
death             int64
drunk             int64
drugs             int64
dtype: object
# recode the categorical variable 'gender' to numeric variable
>>> snsdata_clean['gender'] = preprocessing.LabelEncoder().fit_transform(snsdata_clean['gender'])
>>> del snsdata_clean['gradyear'] # drop useless variable 'gradyear'
```
