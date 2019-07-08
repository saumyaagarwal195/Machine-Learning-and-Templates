import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
import time

def fox2(dataname):
    start_time=time.time()
    dataset = pd.read_csv(dataname)

    X = dataset.iloc[:, 1:179].values
    y = dataset.iloc[:, 179:].values

    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 1
        else:
            y[i] = 0

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)  

    y_pred=clf.predict(X_test)
    y_pred = (y_pred > 0.5)

    cm=accuracy_score(y_test, y_pred)
    print(cm)
    seconds=(time.time() - start_time)
    print("--- %s seconds ---" % seconds)

    return cm,  seconds

#cm,sec=fox()
#print(cm)
#print(sec)


#97.56
#3.679sec