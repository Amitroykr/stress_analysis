# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:40:56 2021

@author: amitr
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#load data set
dataset = pd.read_csv(r'dreaddit_train.csv')

#plot pi-chart
subreaddit_list = ['almosthomeless','anxiety','assistance','domesticviolence','food_pantry','homeless','ptsd','relationships','stress','survivorsofabuse']
subreaddit_csv = dataset.iloc[:,0]
subreaddit_almosthomeless = 0
subreaddit_anxiety = 0
subreaddit_assistance = 0
subreaddit_domesticviolence = 0
subreaddit_food_pantry = 0
subreaddit_homeless = 0
subreaddit_ptsd = 0
subreaddit_relationships = 0
subreaddit_stress = 0
subreaddit_survivorsofabuse = 0

#lable encode
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
subreaddit_csv = labelencoder.fit_transform(subreaddit_csv)

subreaddit_csv_shape = np.shape(subreaddit_csv)
for i in subreaddit_csv:
    if i==0:
        subreaddit_almosthomeless = subreaddit_almosthomeless+1
    elif i==1:
        subreaddit_anxiety = subreaddit_anxiety+1
    elif i==2:
        subreaddit_assistance = subreaddit_assistance+1
    elif i==3:
        subreaddit_domesticviolence = subreaddit_domesticviolence +1
    elif i==4:
        subreaddit_food_pantry = subreaddit_food_pantry +1
    elif i==5:
        subreaddit_homeless = subreaddit_homeless + 1
    elif i==6:
        subreaddit_ptsd = subreaddit_ptsd + 1
    elif i==7:
        subreaddit_relationships = subreaddit_relationships + 1
    elif i==8:
        subreaddit_stress = subreaddit_stress + 1
    elif i==9:
        subreaddit_survivorsofabuse = subreaddit_survivorsofabuse +1
    else:
        pass

subreaddit_number = [subreaddit_almosthomeless,subreaddit_anxiety,subreaddit_assistance,subreaddit_domesticviolence,subreaddit_food_pantry,subreaddit_homeless,subreaddit_ptsd,subreaddit_relationships,subreaddit_stress,subreaddit_survivorsofabuse]


plt.pie(subreaddit_number, labels = subreaddit_list)
plt.show()

#delete some column
dataset.drop(['text', 'post_id' , 'sentence_range', 'id', 'social_timestamp'], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset['subreddit'] = labelencoder.fit_transform(dataset['subreddit'])


X = dataset.drop('label', axis=1)
Y = dataset['label']

#Training and testing data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#implement Linear regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

Y_predict_new = []
for i in Y_predict:
    if i>=0.6:
        Y_predict_new.append(1)
    else:
        Y_predict_new.append(0)

#find accurecy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict_new)
