# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:42:14 2023

@author: deepa
"""

##### 2023-05-21_1500 ##### 

## Titanic Data 

import pandas as pd
import seaborn as sns
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn. import 

raw_data = pd.read_csv("titanic_train.csv", index_col="PassengerId")

raw_data.head()

raw_data.shape

raw_data.isna().sum()

raw_data = raw_data.drop('Cabin',axis=1)

median_age = raw_data["Age"].median()

raw_data["Age"].fillna(median_age,inplace = True)

raw_data["Age"]

mode_emparked = raw_data["Embarked"].mode()[0]

mode_emparked

raw_data["Embarked"].fillna(mode_emparked,inplace = True)

raw_data.shape

raw_data.isna().sum()

raw_data.info()

raw_data.to_csv("clean_titanic_train.csv")

clean_data = pd.read_csv("clean_titanic_train.csv", index_col="PassengerId")

corr = clean_data.corr()
corr

sns.heatmap(corr, cmap = "coolwarm")

sns.boxplot(x=clean_data["Age"])

sns.countplot(x=clean_data["Survived"])

sns.countplot(data=clean_data, x="Survived", hue="Sex")

sns.countplot(data=clean_data, x="Survived", hue="Pclass")

sns.countplot(data=clean_data, x="Survived", hue="Embarked")

sns.histplot(data=clean_data, x="Age", hue="Survived")

sns.histplot(data=clean_data, x="Age", hue="Survived", multiple = "stack")

sns.histplot(data=clean_data, x="Pclass", hue="Survived", multiple = "stack")

sns.histplot(data=clean_data, x="Sex", hue="Survived", multiple = "stack")

sns.histplot(data=clean_data, x="Survived", hue="Sex", multiple = "stack")

sns.distplot(clean_data['Fare'])

clean_data["Survived"] = clean_data["Survived"].astype("int")

clean_data["Pclass"] = clean_data["Pclass"].astype("object")

clean_data.info()

gender_cols = pd.get_dummies(clean_data['Sex'],prefix='Sex')
clean_data = pd.concat([clean_data, gender_cols], axis = 1)
clean_data = clean_data.drop(['Sex'],axis=1)

embark_cols = pd.get_dummies(clean_data['Embarked'],prefix='Embarked')
clean_data = pd.concat([clean_data, embark_cols], axis = 1)
clean_data = clean_data.drop(['Embarked'],axis=1)

Pclass_cols = pd.get_dummies(clean_data['Pclass'],prefix='Pclass')
clean_data = pd.concat([clean_data, Pclass_cols], axis = 1)
clean_data = clean_data.drop(['Pclass'],axis=1)

clean_data.head()

# Embarked S, Q, C -> 1, 2, 3        ### One-Hot Encolding

bin = [0, 10, 20, 40, 60, 80]
catg_age = pd.cut(clean_data['Age'],bin)
clean_data = pd.concat([clean_data, catg_age], axis = 1)
clean_data = clean_data.drop(['Age'],axis=1)

age_cols = pd.get_dummies(clean_data['catg_age'],prefix='Age')
clean_data = pd.concat([clean_data, age_cols], axis = 1)
clean_data = clean_data.drop(['catg_age'],axis=1)

clean_data = clean_data.drop(['Name', 'Ticket'], axis = 1)

clean_data.to_csv("preprocessed_titanic_train.csv")

# preprocessed_data = clean_data.copy()

data = clean_data.copy()

features = data.drop(['Survived'], axis = 1)
labels = data['Survived']

# split the data into training data, validation data and test data
features_train, features_validation_test, labels_train, labels_validation_test = train_test_split(features, labels, test_size = 0.6, random_state = 0)
features_validation, features_test, labels_validation, labels_test = train_test_split(features_validation_test, labels_validation_test, test_size = 0.5, random_state = 0)

lr_model = LogisticRegression()
lr_model.fit(features_train,labels_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(features_train,labels_train)

nb_model = GaussianNB()
nb_model.fit(features_train,labels_train)

svm_model = SVC()
svm_model.fit(features_train,labels_train)

rf_model = RandomForestClassifier()
rf_model.fit(features_train,labels_train)

gb_model = GradientBoostingClassifier()
gb_model.fit(features_train,labels_train)

ab_model = AdaBoostClassifier()
ab_model.fit(features_train,labels_train)

# Testing accuracy of models
lr_model.score(features_validation, labels_validation)
dt_model.score(features_validation, labels_validation)
nb_model.score(features_validation, labels_validation)
svm_model.score(features_validation, labels_validation)
rf_model.score(features_validation, labels_validation)
gb_model.score(features_validation, labels_validation)
ab_model.score(features_validation, labels_validation)

# Testing F1 Score of models

from sklearn.metrics import f1_score

lr_predicted_labels = lr_model.predict(features, validation)
f1_score(labels_validation, lr_predicted_labels)

dt_predicted_labels = dt_model.predict(features, validation)
f1_score(labels_validation, dt_predicted_labels)

nb_predicted_labels = nb_model.predict(features, validation)
f1_score(labels_validation, nb_predicted_labels)

svm_predicted_labels = svm_model.predict(features, validation)
f1_score(labels_validation, svm_predicted_labels)

rf_predicted_labels = rf_model.predict(features, validation)
f1_score(labels_validation, rf_predicted_labels)

gb_predicted_labels = gb_model.predict(features, validation)
f1_score(labels_validation, gb_predicted_labels)

ab_predicted_labels = ab_model.predict(features, validation)
f1_score(labels_validation, ab_predicted_labels)


# Performance on Test Set

gb_model.score(features_test, labels_test)

gb_predicted_labels = gb_model.predict(features_test)
f1_score(labels_test, gb_predicted_labels)


##### 2023-05-28_1500 ##### 

