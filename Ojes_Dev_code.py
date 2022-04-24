#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib as plt

train_data = pd.read_csv("train_data.csv",)
train_data = train_data.drop('Unnamed: 0',axis =1)
train_labels = pd.read_csv("train_labels.csv",)
train_labels.values
train_labels=train_labels.drop('Unnamed: 0',axis =1)
test_data = pd.read_csv("test_data.csv")
test_data= test_data.drop('Unnamed: 0',axis =1)
pd.options.display.max_columns = None
train_data.shape


# In[47]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_label =le.fit_transform(train_labels)
pd.set_option("display.max_rows", None)
train_label


# In[40]:


#Visualising Data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
vis_data_train = pca.fit_transform(train_data)


# In[49]:


vis_data_train_df = pd.DataFrame(vis_data_train, columns=["PCA1","PCA2"])
vis_data_train_df["Activity"] = train_labels["Activity"]
vis_data_train_df


# In[56]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
sns.scatterplot(data= vis_data_train_df, x= 'PCA1', y='PCA2', hue='Activity',s=60, palette='icefire')
plt.show


# In[ ]:


pip install mrmr_selection


# In[ ]:


#feature selection
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=train_data,y=train_label, K=400, relevance='f')


# In[3]:


#using the best features
train_data.drop(train_data.columns.difference(['fBodyAccJerk-entropy()-X',
 'tGravityAcc-mean()-X',
 'tGravityAcc-min()-X',
 'fBodyAccJerk-entropy()-Y',
 'tGravityAcc-max()-X',
 'tGravityAcc-energy()-X',
 'fBodyAcc-entropy()-X',
 'tBodyAccJerkMag-entropy()',
 'fBodyBodyAccJerkMag-entropy()',
 'tBodyAccJerk-entropy()-X',
 'angle(X,gravityMean)',
 'tBodyAcc-max()-X',
 'tBodyAccJerk-entropy()-Z',
 'fBodyAccJerk-entropy()-Z',
 'tBodyAcc-std()-X',
 'tBodyAccJerk-entropy()-Y',
 'fBodyAcc-mad()-X',
 'fBodyAcc-sma()',
 'fBodyAcc-std()-X',
 'tBodyAcc-sma()',
 'tBodyAccMag-mean()',
 'tBodyAccMag-sma()',
 'tGravityAccMag-mean()',
 'tGravityAccMag-sma()',
 'fBodyAcc-entropy()-Y',
 'tBodyAcc-mad()-X',
 'fBodyAcc-mean()-X',
 'fBodyAccMag-entropy()',
 'fBodyAcc-entropy()-Z',
 'tBodyAccJerkMag-mean()',
 'tBodyAccJerkMag-sma()',
 'tBodyAccJerk-sma()',
 'tBodyGyroJerk-entropy()-Z',
 'tBodyAccMag-max()',
 'tGravityAccMag-max()',
 'fBodyAccJerk-sma()',
 'tBodyAccJerkMag-mad()',
 'fBodyAccMag-mad()',
 'fBodyAccMag-mean()',
 'fBodyAccMag-sma()',
 'tBodyAccJerkMag-iqr()',
 'tBodyAccJerk-std()-X',
 'tBodyAccJerkMag-std()',
 'tBodyAccJerk-mad()-X',
 'fBodyAcc-mad()-Y',
 'fBodyBodyGyroJerkMag-entropy()',
 'fBodyAccJerk-mean()-X',
 'fBodyAccJerk-mad()-X',
 'fBodyAcc-mean()-Y',
 'fBodyAccJerk-std()-X',
 'fBodyBodyAccJerkMag-mean()',
 'fBodyBodyAccJerkMag-sma()',
 'fBodyGyro-entropy()-Z',
 'fBodyBodyAccJerkMag-std()',
 'fBodyBodyAccJerkMag-mad()',
 'tBodyAccJerk-mad()-Y',
 'tBodyAccMag-std()',
 'tGravityAccMag-std()',
 'tBodyAcc-std()-Y',
 'fBodyAcc-max()-X',
 'tBodyAccJerk-std()-Y',
 'fBodyGyro-entropy()-X',
 'tBodyAcc-mad()-Y',
 'fBodyAccJerk-mean()-Y',
 'fBodyAccMag-iqr()',
 'tBodyGyroJerkMag-entropy()',
 'fBodyAccJerk-mad()-Y',
 'tBodyGyroMag-mean()',
 'tBodyGyroMag-sma()',
 'fBodyAcc-std()-Y',
 'tBodyGyro-sma()',
 'fBodyAccJerk-std()-Y',
 'tBodyGyroJerk-entropy()-X',
 'fBodyGyro-entropy()-Y',
 'tBodyAccMag-mad()',
 'tGravityAccMag-mad()',
 'tBodyAccJerk-iqr()-X',
 'tBodyAcc-min()-X',
 'tBodyAccJerk-iqr()-Y',
 'fBodyBodyAccJerkMag-iqr()',
 'fBodyAccMag-std()',
 'fBodyBodyGyroMag-entropy()',
 'tBodyAcc-iqr()-X',
 'fBodyGyro-sma()',
 'tBodyAccJerkMag-max()',
 'fBodyAccJerk-iqr()-X',
 'tBodyAccJerk-max()-X',
 'fBodyBodyAccJerkMag-max()',
 'tGravityAcc-energy()-Y',
 'fBodyAccJerk-max()-X',
 'fBodyAcc-iqr()-X',
 'fBodyAccJerk-iqr()-Y',
 'tBodyAccMag-entropy()',
 'tBodyGyroJerk-entropy()-Y',
 'tGravityAccMag-entropy()',
 'fBodyAcc-iqr()-Y',
 'fBodyGyro-mean()-Z',
 'fBodyGyro-mad()-Z',
 'tBodyAcc-iqr()-Y',
 'fBodyGyro-mean()-X',
 'tBodyGyroJerk-mad()-Z',
 'fBodyAccJerk-max()-Y',
 'tBodyGyroJerk-iqr()-Z',
 'tBodyAccJerk-min()-X',
 'fBodyGyro-mad()-X',
 'tBodyGyroJerk-mad()-X',
 'fBodyAcc-mad()-Z',
 'tBodyGyro-std()-Z',
 'tBodyGyroJerk-sma()',
 'tBodyAcc-min()-Y',
 'fBodyAcc-mean()-Z',
 'tBodyGyroJerk-std()-Z',
 'tGravityAcc-max()-Y',
 'tBodyGyroJerk-std()-X',
 'tBodyAcc-max()-Y',
 'tBodyGyroJerk-iqr()-X',
 'tBodyGyro-mad()-Z',
 'tGravityAcc-mean()-Y',
 'tBodyAcc-mad()-Z',
 'tBodyAcc-std()-Z',
 'tBodyAccJerk-min()-Y',
 'tBodyGyro-mad()-X',
 'tBodyAccMag-energy()',
 'tGravityAccMag-energy()',
 'tBodyGyro-std()-X',
 'tBodyGyroJerkMag-mean()',
 'tBodyGyroJerkMag-sma()',
 'tGravityAcc-min()-Y',
 'fBodyGyro-iqr()-Z',
 'fBodyAcc-energy()-X',
 'fBodyAcc-bandsEnergy()-1,24',
 'fBodyGyro-std()-Z',
 'tBodyAcc-energy()-X',
 'fBodyAcc-bandsEnergy()-1,16',
 'angle(Y,gravityMean)',
 'fBodyAccJerk-mean()-Z',
 'tBodyAccJerk-mad()-Z',
 'tBodyAccMag-iqr()',
 'tGravityAccMag-iqr()',
 'fBodyAcc-std()-Z',
 'tBodyAccJerk-std()-Z',
 'fBodyAcc-max()-Y',
 'tBodyGyro-iqr()-X',
 'fBodyAccMag-energy()',
 'fBodyAccJerk-mad()-Z',
 'tBodyAcc-iqr()-Z',
 'fBodyAcc-bandsEnergy()-1,8',
 'tBodyAccJerk-max()-Y',
 'fBodyGyro-std()-X',
 'tGravityAcc-correlation()-Y,Z',
 'fBodyAcc-iqr()-Z',
 'tGravityAcc-energy()-Z',
 'tBodyGyroMag-mad()',
 'tBodyAccJerk-iqr()-Z',
 'fBodyBodyGyroMag-mean()',
 'fBodyBodyGyroMag-sma()',
 'fBodyAccJerk-std()-Z',
 'fBodyAccJerk-iqr()-Z',
 'tBodyGyroMag-max()',
 'fBodyBodyGyroMag-mad()',
 'tBodyGyroMag-iqr()',
 'fBodyGyro-iqr()-X',
 'tBodyGyro-iqr()-Z',
 'tBodyAcc-entropy()-X',
 'tBodyGyroMag-std()',
 'tBodyGyroJerk-max()-Z',
 'fBodyAccMag-max()',
 'tBodyGyro-min()-Z',
 'tBodyAcc-entropy()-Y',
 'tBodyAcc-max()-Z',
 'tBodyGyroJerkMag-iqr()',
 'tBodyGyroJerk-max()-X',
 'tBodyGyro-mad()-Y',
 'tBodyGyro-max()-Z',
 'tBodyAccJerk-min()-Z',
 'fBodyBodyGyroMag-iqr()',
 'tBodyGyro-iqr()-Y',
 'tBodyGyro-std()-Y',
 'fBodyGyro-mean()-Y',
 'fBodyGyro-mad()-Y',
 'fBodyBodyGyroMag-std()',
 'tBodyGyroJerk-min()-X',
 'fBodyGyro-std()-Y',
 'fBodyGyro-max()-Z',
 'tBodyAccJerk-max()-Z',
 'tBodyGyroJerkMag-mad()',
 'fBodyAccJerk-max()-Z',
 'fBodyAccJerk-bandsEnergy()-1,8',
 'tBodyAccJerkMag-energy()',
 'fBodyAcc-max()-Z',
 'fBodyGyro-max()-X',
 'tBodyAcc-min()-Z',
 'fBodyAccJerk-bandsEnergy()-1,16',
 'fBodyBodyGyroJerkMag-mean()',
 'fBodyBodyGyroJerkMag-sma()',
 'tBodyGyro-min()-X',
 'fBodyAccJerk-bandsEnergy()-1,24',
 'fBodyBodyAccJerkMag-energy()',
 'tGravityAcc-arCoeff()-Y,3',
 'tBodyAccJerk-energy()-X',
 'tBodyGyroJerkMag-std()',
 'fBodyAccJerk-energy()-X',
 'tBodyAccMag-min()',
 'tGravityAccMag-min()',
 'tBodyGyroJerk-min()-Z',
 'tBodyGyro-max()-X',
 'fBodyAcc-energy()-Y',
 'fBodyBodyGyroJerkMag-iqr()',
 'fBodyAcc-bandsEnergy()-1,24.1',
 'tBodyAccJerkMag-min()',
 'tBodyGyroJerk-iqr()-Y',
 'tGravityAcc-arCoeff()-Z,3',
 'fBodyBodyGyroJerkMag-mad()',
 'fBodyAcc-bandsEnergy()-1,16.1',
 'fBodyBodyGyroJerkMag-std()',
 'fBodyAcc-bandsEnergy()-1,8.1',
 'tBodyGyroJerk-mad()-Y',
 'tBodyGyroJerkMag-max()',
 'fBodyGyro-max()-Y',
 'tBodyGyroJerkMag-min()',
 'fBodyBodyGyroMag-max()',
 'tBodyAcc-entropy()-Z',
 'tBodyGyro-min()-Y',
 'fBodyGyro-iqr()-Y',
 'fBodyBodyGyroJerkMag-max()',
 'fBodyAccJerk-meanFreq()-X',
 'tBodyAccJerk-energy()-Y',
 'fBodyAccJerk-energy()-Y',
 'tBodyGyroJerk-std()-Y',
 'fBodyAccJerk-bandsEnergy()-1,24.1',
 'tBodyAcc-arCoeff()-X,1',
 'tGravityAcc-max()-Z',
 'tBodyAccJerk-arCoeff()-X,1',
 'tGravityAcc-arCoeff()-Z,2',
 'tGravityAcc-mean()-Z',
 'tBodyGyro-max()-Y',
 'tBodyGyroMag-energy()',
 'angle(Z,gravityMean)',
 'tGravityAcc-min()-Z',
 'tGravityAcc-arCoeff()-X,1',
 'tBodyGyroMag-min()',
 'tBodyGyro-energy()-X',
 'fBodyAcc-bandsEnergy()-9,16',
 'fBodyAccJerk-bandsEnergy()-1,8.1',
 'tBodyAcc-energy()-Y',
 'tGravityAcc-arCoeff()-Z,4',
 'fBodyAccJerk-bandsEnergy()-1,16.1',
 'fBodyAccJerk-bandsEnergy()-9,16',
 'fBodyAccJerk-bandsEnergy()-17,32.1',
 'tBodyGyroJerk-max()-Y',
 'tGravityAcc-sma()',
 'fBodyAcc-bandsEnergy()-17,32.1',
 'tBodyGyroJerk-min()-Y',
 'fBodyAccJerk-bandsEnergy()-33,48',
 'tGravityAcc-arCoeff()-Y,4',
 'fBodyAccJerk-bandsEnergy()-17,24.1',
 'fBodyAcc-bandsEnergy()-33,48',
 'fBodyAcc-bandsEnergy()-17,24.1',
 'tBodyGyro-arCoeff()-Z,2',
 'tGravityAcc-arCoeff()-Y,2',
 'tGravityAcc-arCoeff()-Z,1',
 'fBodyAcc-bandsEnergy()-17,32',
 'fBodyAccJerk-bandsEnergy()-25,48',
 'fBodyAcc-meanFreq()-Z',
 'fBodyAcc-energy()-Z',
 'fBodyAcc-bandsEnergy()-25,48',
 'fBodyAcc-bandsEnergy()-9,16.1',
 'fBodyAcc-bandsEnergy()-1,24.2',
 'fBodyAccJerk-bandsEnergy()-17,32',
 'fBodyBodyAccJerkMag-min()',
 'fBodyAccJerk-bandsEnergy()-9,16.1',
 'fBodyAcc-bandsEnergy()-1,8.2',
 'fBodyAcc-bandsEnergy()-33,40',
 'fBodyAcc-bandsEnergy()-1,16.2',
 'fBodyAccJerk-bandsEnergy()-41,48',
 'fBodyGyro-energy()-X',
 'tBodyAcc-correlation()-Y,Z',
 'tBodyGyroJerk-arCoeff()-Z,1',
 'fBodyAccJerk-bandsEnergy()-33,40',
 'fBodyAcc-bandsEnergy()-17,24',
 'fBodyAcc-meanFreq()-X',
 'fBodyGyro-bandsEnergy()-1,24',
 'fBodyAcc-bandsEnergy()-41,48',
 'fBodyAccJerk-bandsEnergy()-17,24',
 'fBodyGyro-energy()-Z',
 'fBodyGyro-bandsEnergy()-1,16',
 'fBodyAccJerk-meanFreq()-Z',
 'fBodyAcc-skewness()-X',
 'tBodyGyroJerk-energy()-X',
 'fBodyGyro-bandsEnergy()-1,24.2',
 'tBodyGyroJerk-energy()-Z',
 'tGravityAcc-arCoeff()-X,2',
 'tBodyAcc-energy()-Z',
 'fBodyAccJerk-bandsEnergy()-25,48.1',
 'fBodyAcc-kurtosis()-X',
 'fBodyAccJerk-bandsEnergy()-1,24.2',
 'tBodyGyro-energy()-Z',
 'tGravityAcc-arCoeff()-Y,1',
 'tBodyAccJerk-energy()-Z',
 'fBodyAccJerk-energy()-Z',
 'fBodyGyro-bandsEnergy()-1,16.2',
 'tBodyGyro-arCoeff()-Z,1',
 'fBodyAccJerk-maxInds-X',
 'fBodyAcc-bandsEnergy()-25,48.1',
 'tBodyGyro-entropy()-Z',
 'tBodyAccJerk-arCoeff()-Z,1',
 'fBodyAccJerk-bandsEnergy()-33,48.1',
 'tBodyAcc-arCoeff()-Z,1',
 'fBodyGyro-bandsEnergy()-1,8',
 'fBodyAccJerk-bandsEnergy()-1,16.2',
 'fBodyGyro-bandsEnergy()-1,8.2',
 'fBodyGyro-meanFreq()-X',
 'fBodyAccJerk-bandsEnergy()-41,48.1',
 'fBodyAcc-bandsEnergy()-25,32',
 'fBodyAcc-bandsEnergy()-33,48.1',
 'fBodyBodyGyroJerkMag-min()',
 'fBodyBodyGyroMag-energy()',
 'fBodyAcc-bandsEnergy()-41,48.1',
 'fBodyAccJerk-bandsEnergy()-49,64.1',
 'tGravityAcc-arCoeff()-X,3',
 'fBodyAccJerk-bandsEnergy()-49,56.1',
 'fBodyAccJerk-bandsEnergy()-9,16.2',
 'tBodyGyroJerk-correlation()-Y,Z',
 'fBodyAccJerk-bandsEnergy()-25,32',
 'tGravityAcc-arCoeff()-X,4',
 'tBodyGyroMag-entropy()',
 'fBodyGyro-bandsEnergy()-9,16',
 'tBodyAccJerk-arCoeff()-Y,1',
 'fBodyAcc-bandsEnergy()-9,16.2',
 'fBodyAccJerk-meanFreq()-Y',
 'fBodyGyro-bandsEnergy()-17,32',
 'fBodyAcc-bandsEnergy()-25,32.1',
 'tBodyAcc-arCoeff()-Y,1',
 'fBodyAccJerk-bandsEnergy()-25,32.1',
 'tBodyAccJerkMag-arCoeff()1',
 'fBodyAccJerk-min()-X',
 'tBodyAcc-arCoeff()-X,2',
 'fBodyGyro-skewness()-Z',
 'fBodyGyro-bandsEnergy()-17,24',
 'fBodyAccJerk-bandsEnergy()-17,32.2',
 'fBodyAcc-maxInds-X',
 'fBodyAccJerk-bandsEnergy()-49,64',
 'tGravityAcc-correlation()-X,Y',
 'fBodyAccJerk-bandsEnergy()-49,56',
 'tBodyGyroJerkMag-energy()',
 'fBodyGyro-bandsEnergy()-1,8.1',
 'fBodyAcc-bandsEnergy()-17,32.2',
 'fBodyAccJerk-bandsEnergy()-1,8.2',
 'tBodyGyro-arCoeff()-X,1',
 'tBodyGyro-entropy()-Y',
 'fBodyAccJerk-bandsEnergy()-17,24.2',
 'tBodyGyroJerk-arCoeff()-Z,2',
 'fBodyAccJerk-min()-Y',
 'fBodyAcc-bandsEnergy()-33,40.1',
 'tBodyAccJerkMag-arCoeff()2',
 'fBodyAccJerk-bandsEnergy()-33,40.1',
 'fBodyAcc-bandsEnergy()-49,56',
 'tBodyAccMag-arCoeff()1',
 'fBodyGyro-bandsEnergy()-1,16.1',
 'tGravityAccMag-arCoeff()1',
 'fBodyAcc-bandsEnergy()-17,24.2',
 'tGravityAcc-entropy()-X',
 'fBodyAcc-min()-X',
 'fBodyGyro-meanFreq()-Z',
 'tBodyAcc-arCoeff()-Z,2',
 'tBodyGyro-energy()-Y',
 'fBodyGyro-bandsEnergy()-9,16.2',
 'fBodyGyro-bandsEnergy()-1,24.1',
 'tBodyAcc-correlation()-X,Y',
 'fBodyAccJerk-skewness()-X',
 'fBodyGyro-energy()-Y',
 'fBodyGyro-kurtosis()-Z',
 'tBodyGyroJerk-arCoeff()-X,1',
 'tBodyGyro-arCoeff()-Z,3',
 'fBodyGyro-bandsEnergy()-17,32.2',
 'tGravityAcc-entropy()-Y',
 'fBodyAcc-bandsEnergy()-49,56.1',
 'fBodyBodyGyroMag-maxInds',
 'fBodyAccJerk-bandsEnergy()-25,48.2',
 'tBodyAcc-arCoeff()-Y,2',
 'fBodyAcc-bandsEnergy()-25,48.2',
 'tBodyAccMag-arCoeff()2',
 'tGravityAccMag-arCoeff()2',
 'fBodyAccJerk-min()-Z',
 'fBodyBodyAccJerkMag-skewness()',
 'fBodyAccJerk-bandsEnergy()-49,64.2',
 'fBodyAcc-meanFreq()-Y',
 'fBodyAcc-bandsEnergy()-33,48.2',
 'fBodyAccJerk-bandsEnergy()-33,48.2',
 'fBodyAccJerk-bandsEnergy()-49,56.2',
 'fBodyAccJerk-bandsEnergy()-41,48.2',
 'fBodyGyro-bandsEnergy()-17,24.2',
 'tBodyGyro-entropy()-X',
 'fBodyAcc-bandsEnergy()-49,64',
 'fBodyAcc-bandsEnergy()-41,48.2',
 'fBodyAccJerk-skewness()-Y',
 'fBodyAccJerk-bandsEnergy()-25,32.2',
 'tBodyGyro-correlation()-Y,Z',
 'tBodyGyro-arCoeff()-X,2',
 'fBodyAcc-bandsEnergy()-25,32.2'
 ]), 1, inplace=True)


# In[4]:


#spliting the training data for training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.20, random_state=47,stratify=train_label)


# In[5]:


#using different models for classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report 

log_reg_params = [ {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg', 'max_iter':10000, 'n_jobs':-1}]
""""dec_tree_params = [{"criterion": "gini"}, {"criterion": "entropy"}]
rand_for_params = [{"criterion": "gini",'max_depth':200}, {"criterion": "entropy",'max_depth':200}]
kneighbors_params = [{"n_neighbors":3, "weights":'distance'}, {"n_neighbors":5,"weights":'distance'}, {"n_neighbors":25,"weights":'distance'}]
naive_bayes_params = [{}]
naive_multinomial = [{}]
svc_params = [{"C":1,"class_weight":'balanced',"kernel":'linear'}]"""
modelclasses = [
    ["log regression", LogisticRegression, log_reg_params],
    #["decision tree", DecisionTreeClassifier, dec_tree_params],
    #["random forest", RandomForestClassifier, rand_for_params],
    #["k neighbors", KNeighborsClassifier, kneighbors_params],
    #["naive bayes", GaussianNB, naive_bayes_params],
    #["naive_multinomiak",MultinomialNB, naive_bayes_params]
    #["support vector machines", SVC, svc_params]
]
insights = []
for modelname, Model, params_list in modelclasses:
    for params in params_list:
        model = Model(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        insights.append((modelname, model, params, score,report))
        
insights.sort(key=lambda x:x[-1], reverse=True)
for modelname, model, params, score, report in insights:
    print(modelname, params, score, report)


# In[9]:


test_data.drop(test_data.columns.difference(['fBodyAccJerk-entropy()-X',
 'tGravityAcc-mean()-X',
 'tGravityAcc-min()-X',
 'fBodyAccJerk-entropy()-Y',
 'tGravityAcc-max()-X',
 'tGravityAcc-energy()-X',
 'fBodyAcc-entropy()-X',
 'tBodyAccJerkMag-entropy()',
 'fBodyBodyAccJerkMag-entropy()',
 'tBodyAccJerk-entropy()-X',
 'angle(X,gravityMean)',
 'tBodyAcc-max()-X',
 'tBodyAccJerk-entropy()-Z',
 'fBodyAccJerk-entropy()-Z',
 'tBodyAcc-std()-X',
 'tBodyAccJerk-entropy()-Y',
 'fBodyAcc-mad()-X',
 'fBodyAcc-sma()',
 'fBodyAcc-std()-X',
 'tBodyAcc-sma()',
 'tBodyAccMag-mean()',
 'tBodyAccMag-sma()',
 'tGravityAccMag-mean()',
 'tGravityAccMag-sma()',
 'fBodyAcc-entropy()-Y',
 'tBodyAcc-mad()-X',
 'fBodyAcc-mean()-X',
 'fBodyAccMag-entropy()',
 'fBodyAcc-entropy()-Z',
 'tBodyAccJerkMag-mean()',
 'tBodyAccJerkMag-sma()',
 'tBodyAccJerk-sma()',
 'tBodyGyroJerk-entropy()-Z',
 'tBodyAccMag-max()',
 'tGravityAccMag-max()',
 'fBodyAccJerk-sma()',
 'tBodyAccJerkMag-mad()',
 'fBodyAccMag-mad()',
 'fBodyAccMag-mean()',
 'fBodyAccMag-sma()',
 'tBodyAccJerkMag-iqr()',
 'tBodyAccJerk-std()-X',
 'tBodyAccJerkMag-std()',
 'tBodyAccJerk-mad()-X',
 'fBodyAcc-mad()-Y',
 'fBodyBodyGyroJerkMag-entropy()',
 'fBodyAccJerk-mean()-X',
 'fBodyAccJerk-mad()-X',
 'fBodyAcc-mean()-Y',
 'fBodyAccJerk-std()-X',
 'fBodyBodyAccJerkMag-mean()',
 'fBodyBodyAccJerkMag-sma()',
 'fBodyGyro-entropy()-Z',
 'fBodyBodyAccJerkMag-std()',
 'fBodyBodyAccJerkMag-mad()',
 'tBodyAccJerk-mad()-Y',
 'tBodyAccMag-std()',
 'tGravityAccMag-std()',
 'tBodyAcc-std()-Y',
 'fBodyAcc-max()-X',
 'tBodyAccJerk-std()-Y',
 'fBodyGyro-entropy()-X',
 'tBodyAcc-mad()-Y',
 'fBodyAccJerk-mean()-Y',
 'fBodyAccMag-iqr()',
 'tBodyGyroJerkMag-entropy()',
 'fBodyAccJerk-mad()-Y',
 'tBodyGyroMag-mean()',
 'tBodyGyroMag-sma()',
 'fBodyAcc-std()-Y',
 'tBodyGyro-sma()',
 'fBodyAccJerk-std()-Y',
 'tBodyGyroJerk-entropy()-X',
 'fBodyGyro-entropy()-Y',
 'tBodyAccMag-mad()',
 'tGravityAccMag-mad()',
 'tBodyAccJerk-iqr()-X',
 'tBodyAcc-min()-X',
 'tBodyAccJerk-iqr()-Y',
 'fBodyBodyAccJerkMag-iqr()',
 'fBodyAccMag-std()',
 'fBodyBodyGyroMag-entropy()',
 'tBodyAcc-iqr()-X',
 'fBodyGyro-sma()',
 'tBodyAccJerkMag-max()',
 'fBodyAccJerk-iqr()-X',
 'tBodyAccJerk-max()-X',
 'fBodyBodyAccJerkMag-max()',
 'tGravityAcc-energy()-Y',
 'fBodyAccJerk-max()-X',
 'fBodyAcc-iqr()-X',
 'fBodyAccJerk-iqr()-Y',
 'tBodyAccMag-entropy()',
 'tBodyGyroJerk-entropy()-Y',
 'tGravityAccMag-entropy()',
 'fBodyAcc-iqr()-Y',
 'fBodyGyro-mean()-Z',
 'fBodyGyro-mad()-Z',
 'tBodyAcc-iqr()-Y',
 'fBodyGyro-mean()-X',
 'tBodyGyroJerk-mad()-Z',
 'fBodyAccJerk-max()-Y',
 'tBodyGyroJerk-iqr()-Z',
 'tBodyAccJerk-min()-X',
 'fBodyGyro-mad()-X',
 'tBodyGyroJerk-mad()-X',
 'fBodyAcc-mad()-Z',
 'tBodyGyro-std()-Z',
 'tBodyGyroJerk-sma()',
 'tBodyAcc-min()-Y',
 'fBodyAcc-mean()-Z',
 'tBodyGyroJerk-std()-Z',
 'tGravityAcc-max()-Y',
 'tBodyGyroJerk-std()-X',
 'tBodyAcc-max()-Y',
 'tBodyGyroJerk-iqr()-X',
 'tBodyGyro-mad()-Z',
 'tGravityAcc-mean()-Y',
 'tBodyAcc-mad()-Z',
 'tBodyAcc-std()-Z',
 'tBodyAccJerk-min()-Y',
 'tBodyGyro-mad()-X',
 'tBodyAccMag-energy()',
 'tGravityAccMag-energy()',
 'tBodyGyro-std()-X',
 'tBodyGyroJerkMag-mean()',
 'tBodyGyroJerkMag-sma()',
 'tGravityAcc-min()-Y',
 'fBodyGyro-iqr()-Z',
 'fBodyAcc-energy()-X',
 'fBodyAcc-bandsEnergy()-1,24',
 'fBodyGyro-std()-Z',
 'tBodyAcc-energy()-X',
 'fBodyAcc-bandsEnergy()-1,16',
 'angle(Y,gravityMean)',
 'fBodyAccJerk-mean()-Z',
 'tBodyAccJerk-mad()-Z',
 'tBodyAccMag-iqr()',
 'tGravityAccMag-iqr()',
 'fBodyAcc-std()-Z',
 'tBodyAccJerk-std()-Z',
 'fBodyAcc-max()-Y',
 'tBodyGyro-iqr()-X',
 'fBodyAccMag-energy()',
 'fBodyAccJerk-mad()-Z',
 'tBodyAcc-iqr()-Z',
 'fBodyAcc-bandsEnergy()-1,8',
 'tBodyAccJerk-max()-Y',
 'fBodyGyro-std()-X',
 'tGravityAcc-correlation()-Y,Z',
 'fBodyAcc-iqr()-Z',
 'tGravityAcc-energy()-Z',
 'tBodyGyroMag-mad()',
 'tBodyAccJerk-iqr()-Z',
 'fBodyBodyGyroMag-mean()',
 'fBodyBodyGyroMag-sma()',
 'fBodyAccJerk-std()-Z',
 'fBodyAccJerk-iqr()-Z',
 'tBodyGyroMag-max()',
 'fBodyBodyGyroMag-mad()',
 'tBodyGyroMag-iqr()',
 'fBodyGyro-iqr()-X',
 'tBodyGyro-iqr()-Z',
 'tBodyAcc-entropy()-X',
 'tBodyGyroMag-std()',
 'tBodyGyroJerk-max()-Z',
 'fBodyAccMag-max()',
 'tBodyGyro-min()-Z',
 'tBodyAcc-entropy()-Y',
 'tBodyAcc-max()-Z',
 'tBodyGyroJerkMag-iqr()',
 'tBodyGyroJerk-max()-X',
 'tBodyGyro-mad()-Y',
 'tBodyGyro-max()-Z',
 'tBodyAccJerk-min()-Z',
 'fBodyBodyGyroMag-iqr()',
 'tBodyGyro-iqr()-Y',
 'tBodyGyro-std()-Y',
 'fBodyGyro-mean()-Y',
 'fBodyGyro-mad()-Y',
 'fBodyBodyGyroMag-std()',
 'tBodyGyroJerk-min()-X',
 'fBodyGyro-std()-Y',
 'fBodyGyro-max()-Z',
 'tBodyAccJerk-max()-Z',
 'tBodyGyroJerkMag-mad()',
 'fBodyAccJerk-max()-Z',
 'fBodyAccJerk-bandsEnergy()-1,8',
 'tBodyAccJerkMag-energy()',
 'fBodyAcc-max()-Z',
 'fBodyGyro-max()-X',
 'tBodyAcc-min()-Z',
 'fBodyAccJerk-bandsEnergy()-1,16',
 'fBodyBodyGyroJerkMag-mean()',
 'fBodyBodyGyroJerkMag-sma()',
 'tBodyGyro-min()-X',
 'fBodyAccJerk-bandsEnergy()-1,24',
 'fBodyBodyAccJerkMag-energy()',
 'tGravityAcc-arCoeff()-Y,3',
 'tBodyAccJerk-energy()-X',
 'tBodyGyroJerkMag-std()',
 'fBodyAccJerk-energy()-X',
 'tBodyAccMag-min()',
 'tGravityAccMag-min()',
 'tBodyGyroJerk-min()-Z',
 'tBodyGyro-max()-X',
 'fBodyAcc-energy()-Y',
 'fBodyBodyGyroJerkMag-iqr()',
 'fBodyAcc-bandsEnergy()-1,24.1',
 'tBodyAccJerkMag-min()',
 'tBodyGyroJerk-iqr()-Y',
 'tGravityAcc-arCoeff()-Z,3',
 'fBodyBodyGyroJerkMag-mad()',
 'fBodyAcc-bandsEnergy()-1,16.1',
 'fBodyBodyGyroJerkMag-std()',
 'fBodyAcc-bandsEnergy()-1,8.1',
 'tBodyGyroJerk-mad()-Y',
 'tBodyGyroJerkMag-max()',
 'fBodyGyro-max()-Y',
 'tBodyGyroJerkMag-min()',
 'fBodyBodyGyroMag-max()',
 'tBodyAcc-entropy()-Z',
 'tBodyGyro-min()-Y',
 'fBodyGyro-iqr()-Y',
 'fBodyBodyGyroJerkMag-max()',
 'fBodyAccJerk-meanFreq()-X',
 'tBodyAccJerk-energy()-Y',
 'fBodyAccJerk-energy()-Y',
 'tBodyGyroJerk-std()-Y',
 'fBodyAccJerk-bandsEnergy()-1,24.1',
 'tBodyAcc-arCoeff()-X,1',
 'tGravityAcc-max()-Z',
 'tBodyAccJerk-arCoeff()-X,1',
 'tGravityAcc-arCoeff()-Z,2',
 'tGravityAcc-mean()-Z',
 'tBodyGyro-max()-Y',
 'tBodyGyroMag-energy()',
 'angle(Z,gravityMean)',
 'tGravityAcc-min()-Z',
 'tGravityAcc-arCoeff()-X,1',
 'tBodyGyroMag-min()',
 'tBodyGyro-energy()-X',
 'fBodyAcc-bandsEnergy()-9,16',
 'fBodyAccJerk-bandsEnergy()-1,8.1',
 'tBodyAcc-energy()-Y',
 'tGravityAcc-arCoeff()-Z,4',
 'fBodyAccJerk-bandsEnergy()-1,16.1',
 'fBodyAccJerk-bandsEnergy()-9,16',
 'fBodyAccJerk-bandsEnergy()-17,32.1',
 'tBodyGyroJerk-max()-Y',
 'tGravityAcc-sma()',
 'fBodyAcc-bandsEnergy()-17,32.1',
 'tBodyGyroJerk-min()-Y',
 'fBodyAccJerk-bandsEnergy()-33,48',
 'tGravityAcc-arCoeff()-Y,4',
 'fBodyAccJerk-bandsEnergy()-17,24.1',
 'fBodyAcc-bandsEnergy()-33,48',
 'fBodyAcc-bandsEnergy()-17,24.1',
 'tBodyGyro-arCoeff()-Z,2',
 'tGravityAcc-arCoeff()-Y,2',
 'tGravityAcc-arCoeff()-Z,1',
 'fBodyAcc-bandsEnergy()-17,32',
 'fBodyAccJerk-bandsEnergy()-25,48',
 'fBodyAcc-meanFreq()-Z',
 'fBodyAcc-energy()-Z',
 'fBodyAcc-bandsEnergy()-25,48',
 'fBodyAcc-bandsEnergy()-9,16.1',
 'fBodyAcc-bandsEnergy()-1,24.2',
 'fBodyAccJerk-bandsEnergy()-17,32',
 'fBodyBodyAccJerkMag-min()',
 'fBodyAccJerk-bandsEnergy()-9,16.1',
 'fBodyAcc-bandsEnergy()-1,8.2',
 'fBodyAcc-bandsEnergy()-33,40',
 'fBodyAcc-bandsEnergy()-1,16.2',
 'fBodyAccJerk-bandsEnergy()-41,48',
 'fBodyGyro-energy()-X',
 'tBodyAcc-correlation()-Y,Z',
 'tBodyGyroJerk-arCoeff()-Z,1',
 'fBodyAccJerk-bandsEnergy()-33,40',
 'fBodyAcc-bandsEnergy()-17,24',
 'fBodyAcc-meanFreq()-X',
 'fBodyGyro-bandsEnergy()-1,24',
 'fBodyAcc-bandsEnergy()-41,48',
 'fBodyAccJerk-bandsEnergy()-17,24',
 'fBodyGyro-energy()-Z',
 'fBodyGyro-bandsEnergy()-1,16',
 'fBodyAccJerk-meanFreq()-Z',
 'fBodyAcc-skewness()-X',
 'tBodyGyroJerk-energy()-X',
 'fBodyGyro-bandsEnergy()-1,24.2',
 'tBodyGyroJerk-energy()-Z',
 'tGravityAcc-arCoeff()-X,2',
 'tBodyAcc-energy()-Z',
 'fBodyAccJerk-bandsEnergy()-25,48.1',
 'fBodyAcc-kurtosis()-X',
 'fBodyAccJerk-bandsEnergy()-1,24.2',
 'tBodyGyro-energy()-Z',
 'tGravityAcc-arCoeff()-Y,1',
 'tBodyAccJerk-energy()-Z',
 'fBodyAccJerk-energy()-Z',
 'fBodyGyro-bandsEnergy()-1,16.2',
 'tBodyGyro-arCoeff()-Z,1',
 'fBodyAccJerk-maxInds-X',
 'fBodyAcc-bandsEnergy()-25,48.1',
 'tBodyGyro-entropy()-Z',
 'tBodyAccJerk-arCoeff()-Z,1',
 'fBodyAccJerk-bandsEnergy()-33,48.1',
 'tBodyAcc-arCoeff()-Z,1',
 'fBodyGyro-bandsEnergy()-1,8',
 'fBodyAccJerk-bandsEnergy()-1,16.2',
 'fBodyGyro-bandsEnergy()-1,8.2',
 'fBodyGyro-meanFreq()-X',
 'fBodyAccJerk-bandsEnergy()-41,48.1',
 'fBodyAcc-bandsEnergy()-25,32',
 'fBodyAcc-bandsEnergy()-33,48.1',
 'fBodyBodyGyroJerkMag-min()',
 'fBodyBodyGyroMag-energy()',
 'fBodyAcc-bandsEnergy()-41,48.1',
 'fBodyAccJerk-bandsEnergy()-49,64.1',
 'tGravityAcc-arCoeff()-X,3',
 'fBodyAccJerk-bandsEnergy()-49,56.1',
 'fBodyAccJerk-bandsEnergy()-9,16.2',
 'tBodyGyroJerk-correlation()-Y,Z',
 'fBodyAccJerk-bandsEnergy()-25,32',
 'tGravityAcc-arCoeff()-X,4',
 'tBodyGyroMag-entropy()',
 'fBodyGyro-bandsEnergy()-9,16',
 'tBodyAccJerk-arCoeff()-Y,1',
 'fBodyAcc-bandsEnergy()-9,16.2',
 'fBodyAccJerk-meanFreq()-Y',
 'fBodyGyro-bandsEnergy()-17,32',
 'fBodyAcc-bandsEnergy()-25,32.1',
 'tBodyAcc-arCoeff()-Y,1',
 'fBodyAccJerk-bandsEnergy()-25,32.1',
 'tBodyAccJerkMag-arCoeff()1',
 'fBodyAccJerk-min()-X',
 'tBodyAcc-arCoeff()-X,2',
 'fBodyGyro-skewness()-Z',
 'fBodyGyro-bandsEnergy()-17,24',
 'fBodyAccJerk-bandsEnergy()-17,32.2',
 'fBodyAcc-maxInds-X',
 'fBodyAccJerk-bandsEnergy()-49,64',
 'tGravityAcc-correlation()-X,Y',
 'fBodyAccJerk-bandsEnergy()-49,56',
 'tBodyGyroJerkMag-energy()',
 'fBodyGyro-bandsEnergy()-1,8.1',
 'fBodyAcc-bandsEnergy()-17,32.2',
 'fBodyAccJerk-bandsEnergy()-1,8.2',
 'tBodyGyro-arCoeff()-X,1',
 'tBodyGyro-entropy()-Y',
 'fBodyAccJerk-bandsEnergy()-17,24.2',
 'tBodyGyroJerk-arCoeff()-Z,2',
 'fBodyAccJerk-min()-Y',
 'fBodyAcc-bandsEnergy()-33,40.1',
 'tBodyAccJerkMag-arCoeff()2',
 'fBodyAccJerk-bandsEnergy()-33,40.1',
 'fBodyAcc-bandsEnergy()-49,56',
 'tBodyAccMag-arCoeff()1',
 'fBodyGyro-bandsEnergy()-1,16.1',
 'tGravityAccMag-arCoeff()1',
 'fBodyAcc-bandsEnergy()-17,24.2',
 'tGravityAcc-entropy()-X',
 'fBodyAcc-min()-X',
 'fBodyGyro-meanFreq()-Z',
 'tBodyAcc-arCoeff()-Z,2',
 'tBodyGyro-energy()-Y',
 'fBodyGyro-bandsEnergy()-9,16.2',
 'fBodyGyro-bandsEnergy()-1,24.1',
 'tBodyAcc-correlation()-X,Y',
 'fBodyAccJerk-skewness()-X',
 'fBodyGyro-energy()-Y',
 'fBodyGyro-kurtosis()-Z',
 'tBodyGyroJerk-arCoeff()-X,1',
 'tBodyGyro-arCoeff()-Z,3',
 'fBodyGyro-bandsEnergy()-17,32.2',
 'tGravityAcc-entropy()-Y',
 'fBodyAcc-bandsEnergy()-49,56.1',
 'fBodyBodyGyroMag-maxInds',
 'fBodyAccJerk-bandsEnergy()-25,48.2',
 'tBodyAcc-arCoeff()-Y,2',
 'fBodyAcc-bandsEnergy()-25,48.2',
 'tBodyAccMag-arCoeff()2',
 'tGravityAccMag-arCoeff()2',
 'fBodyAccJerk-min()-Z',
 'fBodyBodyAccJerkMag-skewness()',
 'fBodyAccJerk-bandsEnergy()-49,64.2',
 'fBodyAcc-meanFreq()-Y',
 'fBodyAcc-bandsEnergy()-33,48.2',
 'fBodyAccJerk-bandsEnergy()-33,48.2',
 'fBodyAccJerk-bandsEnergy()-49,56.2',
 'fBodyAccJerk-bandsEnergy()-41,48.2',
 'fBodyGyro-bandsEnergy()-17,24.2',
 'tBodyGyro-entropy()-X',
 'fBodyAcc-bandsEnergy()-49,64',
 'fBodyAcc-bandsEnergy()-41,48.2',
 'fBodyAccJerk-skewness()-Y',
 'fBodyAccJerk-bandsEnergy()-25,32.2',
 'tBodyGyro-correlation()-Y,Z',
 'tBodyGyro-arCoeff()-X,2',
 'fBodyAcc-bandsEnergy()-25,32.2'
 ]), 1, inplace=True)


# In[20]:


test_label_values = model.predict(test_data)


# In[21]:


test_label_values


# In[22]:


test_labels = le.inverse_transform(test_label_values)


# In[23]:


test_labels.shape


# In[24]:


print(np.unique(test_labels))


# In[32]:


def count_label(labels):
    LAYING,SITTING,STANDING,WALKING,WALKING_DOWNSTAIRS,WALKING_UPSTAIRS=0,0,0,0,0,0
    for i in labels:
        if i==0:
            LAYING=LAYING+1
        if i==1:
            SITTING=SITTING+1
        if i==2:
            STANDING=STANDING+1
        if i==3:
            WALKING=WALKING+1
        if i==4:
            WALKING_DOWNSTAIRS=WALKING_DOWNSTAIRS+1
        if i==5:
            WALKING_UPSTAIRS=WALKING_UPSTAIRS+1
    return print("LAYING:%d,SITTING:%d,STANDING:%d,WALKING:%d,WALKING_DOWNSTAIRS:%d,WALKING_UPSTAIRS:%d"%(LAYING,SITTING,STANDING,WALKING,WALKING_DOWNSTAIRS,WALKING_UPSTAIRS))


# In[33]:


count_label(test_label_values)


# In[61]:


np.savetxt("Arjun_G_Test_labels.txt",test_labels, delimiter="\n", fmt="%s")


# In[ ]:




