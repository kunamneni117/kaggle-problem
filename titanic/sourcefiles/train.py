#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
for directory,_,files in os.walk('.'):
    for file in files:
        print(os.path.join(directory,file))


# In[3]:


import pandas as pd
train_path = "./dataset/train.csv"
test_path  = "./dataset/test.csv"

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

