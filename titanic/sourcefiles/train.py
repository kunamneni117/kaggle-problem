#!/usr/bin/env python
# coding: utf-8

# # ITERATION 1

# In[4]:


import os
for directory,_,files in os.walk('.\dataset'):
    for file in files:
        print(os.path.join(directory,file))


# In[6]:


import pandas as pd
train_path = "./dataset/input/train.csv"
test_path  = "./dataset/input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# In[5]:


print(train.info())


# In[6]:


print('--------')
print('Percentage of NA per property sorted')
print('--------')
p = (train.isna().sum()/len(train)*100).sort_values(ascending=False)
print(p)
print('--------')
print('Unique values for duplications and other useful info')
print('--------')
u = train.nunique().sort_values()
print(u)


# In[7]:


train['Embarked'].value_counts()


# In[8]:


train['Ticket'].value_counts()


# In[12]:


from sklearn import preprocessing

def cleanData(data):    
    # Data missing and categorical to drop
    data.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)

    # Data missing Case2
    data['Age'] = data.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    # FARE Data missing in test
    data['Fare'] = data.groupby(['Pclass','Sex'])['Fare'].transform(lambda x: x.fillna(x.median()))

    # Data missing Case3
    data.dropna(axis=0, subset=['Embarked'], inplace=True)
    
    # Categorical Data
    le = preprocessing.LabelEncoder()
    
    # Sex
    data['Sex'].replace({'male':0, 'female':1}, inplace=True)
    
    # Embarked
    data['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace=True)
    
    return data


# In[13]:


clean_train = cleanData(train)
clean_test = cleanData(test)


# In[14]:


print(clean_train.info())
print(clean_test.info())


# In[17]:


from sklearn import model_selection
# Set X and y
y = train['Survived']
X = pd.get_dummies(train.drop('Survived', axis=1))

# Split model train test data
X_train, X_val, y_train, y_val = model_selection.train_test_split(X,y, test_size=0.2, random_state=42)


# In[19]:


from sklearn import metrics
def fitAndPredict(model):
    """The following code makes faster to evaluate a model 
    automating the fit and accuracy process"""
    
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    return metrics.accuracy_score(y_val, prediction)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

#Lets some models
model1 = LogisticRegression(solver='liblinear', random_state=42)
model2 = GradientBoostingClassifier()
model3 = RandomForestClassifier()
model4 = SGDClassifier()
model5 = SVC()

models = [model1, model2, model3, model4, model5]
i = 0
for model in models:
    i +=1
    print("Model ", i,":", model)
    print("ACC: ", fitAndPredict(model))


# In[20]:


model = GradientBoostingClassifier(min_samples_split=20, min_samples_leaf=60, max_depth=3, max_features=7)
fitAndPredict(model)


# In[23]:


predict = model.predict(pd.get_dummies(clean_test))

output = pd.DataFrame({'PassengerId': clean_test.PassengerId, 'Survived': predict})
output.to_csv('./output/submission.csv', index=False)
print("Submission saved")


# # ITERATION 2

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('.\dataset\input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[8]:


import category_encoders as encoders
from pandas.api.types import is_numeric_dtype

from imblearn.over_sampling import SMOTE
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[2]:


train_path = "./dataset/input/train.csv"
test_path  = "./dataset/input/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


# In[3]:


print(df_train.info())
print('Size of Train data set = {}'.format(df_train.shape))


# In[4]:


# Delete columns with unique identifiers
col_lst = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_train.drop(col_lst, axis = 1, inplace=True)
print(df_train.info())


# In[5]:


print(df_train.isnull().sum()/len(df_train)*100)


# In[6]:


fare_median = df_train[(df_train['Fare']>0) & (df_train['Fare'].isnull() == False)]['Fare'].median()
age_median = df_train[(df_train['Age']>0) & (df_train['Age'].isnull() == False)]['Age'].median()
print('Median = {}'.format(age_median))
print('No. of records with non-null Age = {}'.format(df_train[(df_train['Age']>0) & (df_train['Age'].isnull() == False)]['Age'].count()))
print('===={} of Median {} with {} Null Records===='.format('Age', age_median, df_train[(df_train['Age'].isnull() == True)]['Survived'].count()))
df_train['Age'].fillna(age_median, inplace=True)


# In[9]:


numerical = ['Age', 'SibSp', 'Parch', 'Fare']
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

for col in numerical:
    if is_numeric_dtype(df_train[col]) == True:
        df_train[df_train[col]>0][col].plot.hist(bins=50, grid=True, legend=None)
        plt.title(col)
        plt.show()


# In[10]:


CATBoostENCODE = encoders.CatBoostEncoder()
categorical = ['Pclass', 'Sex', 'Embarked']

# Cast teh Pclass from integer to string so that we can apply the categorical encoding later
df_train['Pclass'] = df_train['Pclass'].astype(str)

df_target = df_train['Survived'].astype(str)

# Use CatBoost to encode the categorical values
encoder_cat = CATBoostENCODE.fit_transform(df_train[categorical], df_target)
encoded_cat = pd.DataFrame(encoder_cat)
print(encoded_cat.head(10))


# In[11]:


df_model_data = df_train.copy()
df_model_data.drop(categorical, axis = 1, inplace=True)
df_model_data = pd.concat([df_model_data, encoded_cat], axis=1)
df_model_data.info()


# In[15]:


def get_oversample (features, label):

    smote = SMOTE()
    
    X_smote, Y_smote = smote.fit_resample(features, label)
    print("length of original data is ",len(features))
    print("Proportion of True data in original data is ",len(label[label['Survived']==1])/len(label))
    print("Proportion of False data in original data is ",len(label[label['Survived']==0])/len(label))

    print("length of oversampled data is ",len(X_smote))
    print("Proportion of True data in oversampled data is ",len(Y_smote[Y_smote['Survived']==1])/len(Y_smote))
    print("Proportion of False data in oversampled data is ",len(Y_smote[Y_smote['Survived']==0])/len(Y_smote))
   
    return X_smote, Y_smote, features, label


# In[16]:


Y = df_model_data.iloc[:,0:1]
X = df_model_data.iloc[:,1:]
X_smote, Y_smote, X_train,Y_train = get_oversample(X, Y)


# ## GRID SEARCH

# ### RANDOM FOREST

# In[17]:


# parameter list
p_cv = 5
p_score = 'accuracy'

# Maximum number of depth in each tree:
max_depth = [7,8,9,10,11,12]
# Minimum number of samples to consider at each leaf node:
min_samples_leaf = [10,15,20,40,60,80]## Decision Tree
# Minimum number of samples to consider to split a node:
min_samples_split = [10,15,20,40]
# No. of estimators
estimators = [50, 100, 150, 200,300,500]

clf = RandomForestClassifier()
forest_params_grid={'n_estimators':estimators,
           'max_depth':max_depth,
           'min_samples_split':min_samples_split,
           'min_samples_leaf':min_samples_leaf}

cv = model_selection.StratifiedKFold(n_splits=p_cv, random_state=56456, shuffle=True)
model = model_selection.GridSearchCV(estimator= clf, param_grid=forest_params_grid, cv=cv, scoring=p_score, n_jobs=-1, verbose=1)


# In[19]:


model.fit(X_smote, Y_smote.values.ravel())
print(model.best_estimator_)

