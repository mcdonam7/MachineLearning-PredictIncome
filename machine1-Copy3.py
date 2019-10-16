#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import csv
from sklearn.ensemble import RandomForestRegressor
#how to import category encoders????
#import category_encoders
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
#from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from subprocess import check_output
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder


import warnings
warnings.filterwarnings('ignore')


label = LabelEncoder()





#from sklearn.cross_validation import train_test_split 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


#dataset
train_set = pd.read_csv(r"C:\Users\Mary\Desktop\Marys College\Fourth year\Machine learning\Assignement 1\tcd ml 2019-20 income prediction training (with labels).csv") 
#dataset_test
test_set= pd.read_csv(r"C:\Users\Mary\Desktop\Marys College\Fourth year\Machine learning\Assignement 1\tcd ml 2019-20 income prediction test (without labels).csv") 
#dataset.drop(['Gender', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis = 1) 


# In[43]:


#train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
#dataset_copy = train_set.copy()
##train_set_full = train_set.copy()
#target = train_set['Income in EUR']
#train_id = train_set['Instance']
#test_id = test_set['Instance']
#train_set.drop(['Income in EUR', 'Instance'], axis=1, inplace=True)

#xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

######trying this#######

target = train_set['Income in EUR']
train_id = train_set['Instance']
test_id = test_set['Instance']

train_set = train_set.drop(['Hair Color'], axis=1)
test_set = test_set.drop(['Hair Color'], axis=1)

train_set = train_set.drop(['Wears Glasses'], axis=1)
test_set = test_set.drop(['Wears Glasses'], axis=1)


train_set.drop(['Income in EUR', 'Instance'], axis=1, inplace=True)


# In[44]:


#train_set_full = train_set.copy()

#train_set = train_set.drop(["Income in EUR"], axis=1)
#train_set = train_set.drop(['Gender', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color'], axis = 1) 
#train_labels = dataset_copy["Income in EUR"]

#lin_reg = LinearRegression()

#lin_reg.fit(train_set, train_labels)

scaler = StandardScaler()
train_set.Age = scaler.fit_transform(train_set.Age.values.reshape(-1,1))
test_set.Age = scaler.transform(test_set.Age.values.reshape(-1,1))

#train_set.Year of Record = scaler.fit_transform(train_set.['Year of Record'].values.reshape(-1,1))
#test_set.Year of Record = scaler.transform([test_set.['Year of Record'].values.reshape(-1,1))


# In[45]:


categoricals = train_set.select_dtypes(exclude=[np.number])
#print(categoricals.describe())

#dealing with missing data
train_set["Gender"] = train_set["Gender"].fillna("other")
train_set["University Degree"] = train_set["University Degree"].fillna("None")
#train_set["Hair Color"] = train_set["Hair Color"].fillna("Hair Color")
train_set["Age"] = train_set["Age"].fillna(train_set['Age'].mean())
train_set["Year of Record"] = train_set["Year of Record"].fillna(train_set['Year of Record'].median())
train_set["Profession"] = train_set["Profession"].fillna("None")

test_set["Gender"] = test_set["Gender"].fillna("other")
test_set["University Degree"] = test_set["University Degree"].fillna("None")
#test_set["Hair Color"] = test_set["Hair Color"].fillna("Hair Color")
test_set["Age"] = test_set["Age"].fillna(test_set['Age'].mean())
test_set["Year of Record"] = test_set["Year of Record"].fillna(test_set['Year of Record'].median())
test_set["Profession"] = test_set["Profession"].fillna("None")


# In[46]:


categoricals = train_set.select_dtypes(exclude=[np.number])
#print(categoricals.describe())

feature_list = list(train_set.columns)


# In[47]:


#MEE_encoder = MEstimateEncoder()
#train_mee = MEE_encoder.fit_transform(train_set[feature_list], target)
#test_mee = MEE_encoder.transform(test_set[feature_list])
#print(train_mee.head())
X_train, X_val, y_train, y_val = train_test_split(train_set, target, test_size=0.2, random_state=97)
lr = LinearRegression()
rf = RandomForestRegressor()


# In[48]:


TE_encoder = TargetEncoder()
train_te = TE_encoder.fit_transform(train_set[feature_list], target)
test_te = TE_encoder.transform(test_set[feature_list])
#print(train_te.head())
encoder_list = [ TargetEncoder(), MEstimateEncoder()]
X_train, X_val, y_train, y_val = train_test_split(train_set, target, test_size=0.2, random_state=97)
#X_train, X_val, y_train, y_val = dataset_test()
lr = LinearRegression()

for encoder in encoder_list:
#    print("Test {} : ".format(str(encoder).split('(')[0]), end=" ")

    train_enc = encoder.fit_transform(X_train[feature_list], y_train)
      # test_enc = encoder.transform(test[feature_list])
    val_enc = encoder.transform(X_val[feature_list])
    
    lr.fit(train_enc, y_train)
#    print(lr.score(train_enc, y_train))
    
    
TE_encoder = TargetEncoder()
train_te = TE_encoder.fit_transform(X_train[feature_list], y_train)
test_te = TE_encoder.transform(test_set[feature_list])
#test_te = TE_encoder.transform(dataset_test[feature_list])
#print(train_te.head())
#print(test_te.head())

#lr.fit(train_te, y_train)
rf.fit(train_te, y_train)

preds = rf.predict(test_te)
#preds = preds.clip(min=0)

print(preds)


# In[49]:


#df = pd.read_csv(r"C:\Users\Mary\Desktop\Marys College\Fourth/year\Machine/learning\Assignement/1\tcd ml 2019-20 income prediction submission file.csv") 

files = 'C:\\Users\\Mary\\Desktop\\Marys College\\Fourth year\\Machine learning\\Assignement 1\\tcd ml 2019-20 income prediction submission file.csv'
df = pd.read_csv(files)


#sub = pd.DataFrame()
df['Instance'] = df['Instance']
df['Income'] = preds
df.to_csv(r'C:\Users\Mary\Desktop\Marys College\Fourth year\Machine learning\Assignement 1\tcd ml 2019-20 income prediction submission file.csv',index=False)


# In[ ]:





# In[10]:


#histogram
#sns.distplot(dataset['Income in EUR']);


# In[12]:


#missing data
#total = dataset.isnull().sum().sort_values(ascending=False)
#percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data.head(20)


# In[50]:


#dataset.drop('Profession', axis=1)
#dataset = dataset.drop("Wears Glasses", axis=1)
#dataset['Income in EUR'].describe()


# In[51]:


#dataset.hist(bins=50, figsize=(20,15))
#plt.show()


# In[126]:


missing_val_count_by_column = (dataset.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[139]:


my_imputer = SimpleImputer()
#imputer only works with numerical values - i think
#data_with_imputed_values = my_imputer.fit_transform(dataset)


# In[108]:


knn = KNeighborsClassifier(n_neighbors=12)
#instantiating estimator
#n_neighbors=1 is 1 neighbor away
#n_neighbors is a hyperparameter/tuning parameter


# In[62]:


print(knn)


# In[64]:


dataset.dtypes


# In[65]:


dataset.isnull().sum()


# In[16]:


#sns.pairplot(train_set, x_vars='Age', y_vars='Income in EUR', size=7, aspect=0.7)


# In[17]:


train_set.describe()


# In[68]:


reg=linear_model.LinearRegression()


# In[ ]:





# In[69]:


plt.figure(figsize=(12,12))
sns.heatmap(dataset.corr(), annot=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




