#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h2o
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from h2o.estimators.gbm import H2OGradientBoostingEstimator

h2o.init()


# In[2]:


N = 1000
bloodTypes = ['A', 'O', 'AB', 'B']
np.random.seed(1)

id = range(0,N)
bloodTypesSample = [bloodTypes[z] for z in np.random.randint(0, 4, N)]
age = np.random.randint(18, 65, N)
healthyEating = np.round(np.random.normal(5, 2, N),0)
activeLifestyle = np.round(np.random.normal(5, 2, N),0)

for i in range(0,N):
    if healthyEating[i] > 9.:
        healthyEating[i] = 9
    if healthyEating[i] < 0.:
        healthyEating[i] = 0

for i in range(0,N):
    if age[i] < 30:
        activeLifestyle[i]+=1
    if activeLifestyle[i] > 9.:
        activeLifestyle[i] = 9
    if activeLifestyle[i] < 0.:
        activeLifestyle[i] = 0

income = [round(20000 + (age[i]*3)**2 + healthyEating[i] * 500 - activeLifestyle[i] * 300 + np.random.randint(0, 5000), -2)
             for i in range(0,N)]


# In[3]:


people = pd.DataFrame({
    'Id':id,
    'Blood Type':bloodTypesSample, 
    'Age':age, 
    'Healthy Eating':healthyEating, 
    'Active Lifestyle':activeLifestyle, 
    'Income':income
})

people = h2o.H2OFrame(
    people
    )


# In[4]:


people.describe()


# In[5]:


## Training to first model

train, valid, test = people.split_frame(
    ratios = [0.8, 0.1],
    destination_frames=['people_train','people_valid','people_test'],
    seed = 123
)

y = 'Income'
ignoreFields = [y, 'Id']
x = [i for i in train.names if i not in ignoreFields]

ml = H2OGradientBoostingEstimator(model_id = 'defaults')
ml.train(x, y, train, validation_frame = valid)


print('Train sample MAE:', ml.mae(train=True))
print('Validation sample MAE:', ml.mae(valid=True))

perf = ml.model_performance(test)

print('Perfomance MAE:', perf.mae()) 


# In[6]:


## Training the second model (cross-validation)

train_cv, test_cv = people.split_frame(
    ratios = [0.85],
    destination_frames=['people_train_cv','people_test_cv'],
    seed = 123
)


ml_cv = H2OGradientBoostingEstimator(model_id = 'def9folds', nfolds = 9)
ml_cv.train(x, y, train)

print('Train sample MAE:', ml_cv.mae(train=True))
print('Validation sample MAE:', ml_cv.mae(xval=True))

perf_cv = ml_cv.model_performance(test)
print('Perfomance MAE:', perf_cv.mae()) 

