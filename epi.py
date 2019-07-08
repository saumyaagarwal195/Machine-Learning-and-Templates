#Importing keras libraries and packages
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import time

def fox(dataname):
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

    classifier = Sequential()

    #Adding input layer and first hidden layer
    classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu', input_dim = 178))

    #Adding second hidden layer
    #classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))

    #Adding the output layer
    classifier.add(Dense(output_dim =1 , init = 'uniform', activation = 'sigmoid'))

    #Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Fitting the ANN to the training set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

    #gnb = GaussianNB()
    #y_pred = gnb.fit(X_train, y_train).predict(X_test)
    #print(y_pred)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(y_pred)
    y_pred = (y_pred > 0.5)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)

    cm=accuracy_score(y_test, y_pred)
    print(cm)
    seconds=(time.time() - start_time)
    print("--- %s seconds ---" % seconds)

    #from sklearn.externals import joblib
    # Output a pickle file for the model
    #joblib.dump(clf, 'saved_model.pkl') 

    return cm,  seconds