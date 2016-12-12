
# coding: utf-8

# In[1]:

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, auc
from sklearn import grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


# In[2]:

data = pd.read_csv('NFLPlaybyPlay2015.csv')
data.columns


# In[3]:

data['PlayType_lag'] = data['PlayType']
data.PlayType_lag = data.PlayType_lag.shift(+1)
data['ydsnet_lag'] = data['ydsnet']
data.ydsnet_lag = data.ydsnet_lag.shift(+1)
data['ScoreDiff_lag'] = data['ScoreDiff']
data.ScoreDiff_lag = data.ScoreDiff_lag.shift(+1)


# In[ ]:

def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result

mask = data.groupby(['GameID'])['GameID'].transform(mask_first).astype(bool)
data1 = data.loc[mask]
data1.drop(data1.index[len(data1)-1])


# In[ ]:




# In[ ]:

#only want to use downs 1-3 

downs = [1,2,3]
 #find the downs in the down column, don't use fourth down because can be 
    #unpredictable and don't know if players will run it or kick
data1 = data1[data1['down'].isin(downs)]

#using only certain plays to predict what the offense will do.
#plays like no play, knee, illegal formation do not help to predict this
used_plays = ['Run', 'Pass']
#find only these plays in the play type column
data1 = data1[data1['PlayType'].isin(used_plays)]

#want to create a binary classification of what the play type is
#0 for run and 1 for pass/sack

data1['play'] = data1.PlayType.apply(lambda x: 1 if x == "Pass" else 0)

#these features are the features I will use to predict whether there will
#be a sack, run, or pass
data1 = data1[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'PosTeamScore', 'ScoreDiff','DefTeamScore','Drive','play','PlayType_lag','ydsnet_lag']]



# In[ ]:

data_play = pd.get_dummies(data1['PlayType_lag'])


# In[ ]:

data1 = pd.concat([data1,data_play],axis =1)
data1


# In[ ]:

data1.dtypes


# In[ ]:




# In[ ]:

random_state = 18
train, test = train_test_split(data1, test_size = 0.33)


# In[ ]:

train_X = train[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'PosTeamScore', 'ScoreDiff','DefTeamScore','Drive','ydsnet_lag','End of Game','Extra Point', 'Field Goal', 'Kickoff', 'No Play','Onside Kick','Pass','Punt','QB Kneel','Quarter End','Run','Sack','Spike','Timeout','Two Minute Warning']]
train_y = train[['play']]
test_X = test[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'PosTeamScore', 'ScoreDiff','DefTeamScore','Drive','ydsnet_lag','End of Game','Extra Point', 'Field Goal', 'Kickoff', 'No Play','Onside Kick','Pass','Punt','QB Kneel','Quarter End','Run','Sack','Spike','Timeout','Two Minute Warning']]
test_y = test[['play']]


# In[ ]:

train_X


# In[ ]:

clf = RandomForestRegressor(n_jobs = -1, oob_score = True, n_estimators = 100, min_samples_leaf = 12, max_features =.8)


# In[ ]:

clf.fit(train_X,train_y)


# In[ ]:

preds = clf.predict(test_X)
peep = clf.oob_prediction_
peep


# In[ ]:

clf.score(test_X,test_y)


# In[ ]:

print("roc score area", roc_auc_score(train_y,peep))


# In[ ]:

#most important features for predicting whether a player will pass/Sack or 
#run on the next play
x = range(train_X.shape[1])
sns.barplot(x = clf.feature_importances_, y = train_X.columns,palette="Blues_d")
sns.despine(left=True, bottom=True)


# In[ ]:




# In[ ]:




# In[ ]:

#clear output again to run this model 


# In[4]:

def mask_first(x):
    result = np.ones_like(x)
    result[0] = 0
    return result

mask = data.groupby(['GameID'])['GameID'].transform(mask_first).astype(bool)
data2 = data.loc[mask]
data2.drop(data2.index[len(data2)-1])


# In[ ]:





# In[5]:

downs = [1,2,3]
 #find the downs in the down column, don't use fourth down because can be 
    #unpredictable and don't know if players will run it or kick
data2 = data2[data2['down'].isin(downs)]

type_pass = ['Short','Deep']
 #only get the data rows that have passes that are either deep or short
data2 = data2[data2['PassLength'].isin(type_pass)]

#make throw equal 1 if the pass is short and 0 if it is deep
data2['throw'] = data2.PassLength.apply(lambda x: 1 if x == "Short" else 0)

data2 = data2[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'ydsnet_lag','ScoreDiff_lag','Drive','throw']]


# In[6]:

data2 = data2[np.isfinite(data2['ScoreDiff_lag'])]


# In[ ]:




# In[7]:

train_2, test_2 = train_test_split(data2, test_size = 0.33)


# In[8]:

train_Xa = train_2[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'ydsnet_lag', 'ScoreDiff_lag','Drive']]
train_ya = train_2[['throw']]
test_Xa = test_2[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'ydsnet_lag', 'ScoreDiff_lag','Drive']]
test_ya = test_2[['throw']]


# In[ ]:




# In[9]:

clf = RandomForestRegressor(n_jobs = -1, oob_score = True, n_estimators = 100, min_samples_leaf = 12, max_features =.8)


# In[10]:

clf.fit(train_Xa,train_ya)


# In[11]:

pred = clf.predict(test_Xa)
probas = clf.oob_prediction_
pred


# In[12]:

clf.score(test_Xa,test_ya)


# In[13]:

print("roc score area", roc_auc_score(train_ya,probas))


# In[14]:

xa = range(train_Xa.shape[1])
sns.barplot(x = clf.feature_importances_, y = train_Xa.columns,palette="Blues_d")
sns.despine(left=True, bottom=True)


# In[ ]:




# In[ ]:

#train_Xa = train_2[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'ydsnet', 'PosTeamScore', 'ScoreDiff','DefTeamScore','Drive']]
#train_ya = train_2[['throw']]
#test_Xa = test_2[['qtr', 'down', 'yrdline100','ydstogo', 'TimeSecs', 'ydsnet', 'PosTeamScore', 'ScoreDiff','DefTeamScore','Drive']]
#test_ya = test_2[['throw']]

data.dropna()
X = data[['qtr', 'down', 'yrdline100','ydstogo', 'ydsnet', 'PosTeamScore', 'ScoreDiff','DefTeamScore','Drive']]
y = data[['throw']]


    
train_Xa = X[0:12634].values
test_Xa = X[X > 12634].values
train_ya = y[0:12634].values
test_ya = y[y > 12634].values


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
 
# shuffle and split training and test sets
clf = GaussianNB()
#clf = RandomForestRegressor(n_jobs = -1, oob_score = True, n_estimators = 100, min_samples_leaf = 12, max_features =.8)
clf.fit(train_Xa, train_ya)
 
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(test_ya, clf.predict_proba(test_Xa)[:,1])
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %0.2f' % roc_auc
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print clf.predict_proba(test_Xa)


# In[ ]:

probas = clf.oob_prediction_(test_Xa) #[:,1]  
fpr, tpr, thresholds = metrics.roc_curve(test_ya, probas)
roc_auc = metrics.auc(fpr, tpr)
    
print("Area under the ROC %s curve : %f" % ('RandomForest', roc_auc))
    
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()

