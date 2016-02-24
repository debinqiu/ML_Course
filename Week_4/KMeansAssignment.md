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

# recode the categorical variable 'gender' to numeric variable
>>> snsdata_clean['gender'] = preprocessing.LabelEncoder().fit_transform(snsdata_clean['gender'])
>>> del snsdata_clean['gradyear'] # drop useless variable 'gradyear'
```

Before runing a k-means clustering analysis, we first need to standardize each feature so that each of them has mean 0 and standard deviation 1 because euclidean distance is very sensitive to the scales of variables. This can be done as follows.
```python
# standardize each variable so that mean = 0 and std = 1
>>> for name in snsdata_clean.columns:
        snsdata_clean[name] = preprocessing.scale(snsdata_clean[name]).astype('float64')
```

Now we can perform k-means clustering analysis on those 39 standardized features. For simplicity, we only examine the number of clusters from 1 to 20, although they can be up to 39 clusters. In effect, it is rather safe to only check 1-10 clusters. The Python code to run k-means clustering is as follows.
```python
# perform k-means clustering for each k between 1 - 20   
>>> from scipy.spatial.distance import cdist

>>> clusters = range(1,20)
>>> meandist = []

>>> for k in clusters:
        model = KMeans(n_clusters = k)
        model.fit(snsdata_clean)
        clusassign = model.predict(snsdata_clean)
        meandist.append(sum(np.min(cdist(snsdata_clean,model.cluster_centers_,'euclidean'), axis = 1))/snsdata_clean.shape[0])
```
Since we have claculated the mean distance for each cluster, we can now plot the Elbow graph by checking the average distance versus the number of cluster which is shown as follows. The Elbow graph suggests us to choose the number of cluster as 5 because there is a small jump when the number of cluster is 6. 
