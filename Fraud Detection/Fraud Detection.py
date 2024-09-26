#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection

# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

# ##IMPORTING PACKAGES

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# Reading a Data

# In[2]:


df = pd.read_csv('creditcardFraud.csv',sep=',')
df.head()


# # Exploratory Data Analysis

# In[3]:


df.info()


# In[4]:


df.isnull().values.any()


# In[5]:


count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[6]:


## Get the Fraud and the normal dataset 

fraud = df[df['Class']==1]

normal = df[df['Class']==0]


# In[7]:


print(fraud.shape,normal.shape)


# In[8]:


## Analyze more amount of information from the transaction data
#How different are the amount of money used in different transaction classes?
fraud.Amount.describe()


# In[9]:


normal.Amount.describe()


# In[10]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[11]:


# How often Do fraudulent transactions occur during certain time frame ? 

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time,fraud.Amount)
ax1.set_title('fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[12]:


## sample of the data

sdf= df.sample(frac = 0.3,random_state=1)

sdf.shape


# In[13]:


df.shape


# In[14]:


#Determine the number of fraud and valid transactions in the dataset

Fraud = sdf[sdf['Class']==1]

Valid = sdf[sdf['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))


# In[15]:


print(outlier_fraction)

print("Fraud Cases : {}".format(len(Fraud)))

print("Valid Cases : {}".format(len(Valid)))


# In[16]:


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = sdf.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[17]:


#Create independent and Dependent Features
columns = sdf.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = sdf[columns]
Y = sdf[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# # Model Prediction

# In this project we are using two types of model for data prediction
# 
# Isolation Forest Algorithm :
#     
# The Isolation Forest algorithm identifies observations by randomly selecting a feature and then choosing a random split value between the feature's maximum and minimum values. The underlying reasoning is that isolating anomalous observations is simpler because they require fewer conditions to separate them from the rest of the data. In contrast, isolating normal observations takes more conditions. As a result, the anomaly score is determined by the number of conditions needed to isolate a particular observation.
# 
# The algorithm performs this by building isolation trees, which are random decision trees. The anomaly score is then calculated based on the path length required to isolate the observation within these trees.
# 
# 
# Local Outlier Factor(LOF) Algor
# ithm:
#     
# The LOF (Local Outlier Factor) algorithm is an unsupervised method for detecting outliers by measuring the local density deviation of a data point relative to its neighbors. It identifies data points as outliers if their density is significantly lower compared to that of their neighboring points.

# In[18]:


##Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}


# In[19]:


type(classifiers)


# In[20]:


n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
        scores_prediction = clf.decision_function(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # Reshape the prediction values: 1 for Fraud transactions, 0 for valid transactions
    y_pred = np.where(y_pred == 1, 0, 1)
    
    # Calculate the number of errors
    n_errors = (y_pred != Y).sum()

    # Run Classification Metrics
    print(f"{clf_name}: {n_errors}")
    print("Accuracy Score:")
    print(accuracy_score(Y, y_pred))
    print("Classification Report:")
    print(classification_report(Y, y_pred))


# # OBSERVATIONS

# Isolation Forest detected 199 errors, compared to 263 errors detected by the Local Outlier Factor (LOF) and 8,516 errors detected by the SVM model. Isolation Forest demonstrated an accuracy of 99.76%, outperforming LOF at 99.69% and SVM at 70.09%.
# 
# When evaluating error precision and recall across the three models, Isolation Forest showed significantly better performance than LOF, with a fraud detection rate of approximately 27%, compared to just 3% for LOF and 0% for SVM. Overall, Isolation Forest proved to be much more effective, achieving around 30% fraud detection accuracy.
# 
# This accuracy could be further improved by increasing the sample size or employing deep learning algorithms, though this would come with higher computational costs. Additionally, more complex anomaly detection models could be explored to enhance fraud detection accuracy.

# In[ ]:




