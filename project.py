# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 03:13:26 2019

@author: chidi
"""
#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##### SYSTEM 1 #####

#loading the dataset
reason_ds = pd.read_csv('ReasonBOW.csv')

##bag of words model
def simple_split(data, y, length, split_mark=0.7):
    if split_mark  > 0. and split_mark < 1.0:
        n = int(split_mark*length)
    else:
        n = int(split_mark)
    x_train = data[:n].copy()
    x_test = data[n:].copy()
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    return x_train, x_test, y_train, y_test

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))


x_train, x_test, y_train, y_test = simple_split(reason_ds.Reason,reason_ds.Suicide,len(reason_ds))
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

features_names = tfidf.get_feature_names()
tfidf.vocabulary_

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), x_train, y_train, cv=5)
print('mean cross val accuracy: {:.2f}'.format(np.mean(scores)))

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
print('training set score: {:.3f}'.format(logreg.score(x_train, y_train)))
print('test set score: {:.3f}'.format(logreg.score(x_test, y_test)))

#testin our model
reason = 'my life is a mess, i can not keep up'
reason2 = 'in love with my life, depression cannot win me'
print(logreg.predict(tfidf.transform([reason]))[0])



#####  SYSTEM 2 #####


#loading the dataset
dataset_a = pd.read_csv('Suicide1.csv')
dataset_b = pd.read_csv('Customer Feedback.csv')

dataset_a['suicide'] = 1
dataset_b['suicide'] = 0

df = pd.concat([dataset_a,dataset_b], ignore_index=True)

for i in df.columns:
    print(i, df[i].isnull().sum())

print(df['Parental Status'].unique())
print(df['Parental Status'].value_counts())

df.head()
df.shape

###          DATA PREPROCESSING        ###
#checking for missing values
def null_value_check(df):
    for column_name in df.columns:
        print(column_name, df[column_name].isnull().sum())
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
   
null_value_check(df)

features = [x for x in df.columns if x not in ['Age', 'suicide'] ]

#taking care of missing values
def fill_na(df):
    for column_name in features:
        df[column_name] = df[column_name].fillna('unknown')
    df['Age'] =  df['Age'].fillna(  df['Age'].median() )
   
fill_na(df)

def cat_features(df):
    for column in features:
        print(df[column].unique())
        print(df[column].value_counts()) 

cat_features(df)

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df = pd.get_dummies(df, columns=[x for x in df.columns if x not in ['Age', 'Gender', 'suicide'] ], drop_first = True )

#splitting the dataset into train and test
from sklearn.model_selection import train_test_split
dtrain, dtest = train_test_split(df, test_size=0.25, random_state = 42)

#importing the classifiers
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

model1 = xgb.XGBClassifier()
model2 = RandomForestClassifier()
model2.fit(dtrain[predictors], dtrain['suicide'])


#for cross validation and AUC Score
from sklearn import model_selection, metrics   #Additional scklearn functions
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['suicide'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['suicide'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['suicide'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['suicide'], dtrain_predprob))

                    
predictors = [x for x in df.columns if x not in ['suicide'] ]

modelfit(model1, dtrain, predictors)
modelfit(model1, dtest, predictors)

pred_train = model2.predict(dtest[predictors])

y = dtest.iloc[:, 2:3].values

#checking f1score
from sklearn.metrics import f1_score
f1_score(pred_train, y)









#statistical analysis and visual analysis

df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include=['object'])

pd.crosstab(df['Gender'], df['Anxiety/ Depression'])
sns.countplot(x="Gender", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Anxiety/ Depression'], df['Pain/ Discomfort'])
sns.countplot(x="Anxiety/ Depression", hue='Pain/ Discomfort', data=df)

pd.crosstab(df['Occupation'], df['Anxiety/ Depression'])
sns.countplot(x="Occupation", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Age'], df['Anxiety/ Depression'])
sns.countplot(x="Age", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Receiving treatment for depression'], df['Anxiety/ Depression'])
sns.countplot(x="Receiving treatment for depression", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Ethnicity of victim'], df['Anxiety/ Depression'])
sns.countplot(x="Ethnicity of victim", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Marital Status'], df['Anxiety/ Depression'])
sns.countplot(x="Marital Status", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Parental Status'], df['Anxiety/ Depression'])
sns.countplot(x="Parental Status", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Ethnicity of victim'], df['Gender'])
sns.countplot(x="Ethnicity of victim", hue='Gender', data=df)

pd.crosstab(df['Tools/Means'], df['Gender'])
sns.countplot(x="Tools/Means", hue='Gender', data=df)

pd.crosstab(df['Year of Incident'], df['Anxiety/ Depression'])
sns.countplot(x="Year of Incident", hue='Anxiety/ Depression', data=df)

pd.crosstab(df['Level of Education'], df['Anxiety/ Depression'])
sns.countplot(x="Level of Education", hue='Anxiety/ Depression', data=df)









