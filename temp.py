# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy

dataset=pd.read_csv('train.csv')




X=dataset.iloc[:,0:4].values
Y=dataset.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
Y=enc.fit_transform(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,Y_test)

accuracy=np.trace(cm)/np.sum(cm)

 