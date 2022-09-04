import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as mp
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import *

df = pd.read_csv("diabetes.csv")
df.head()

notzero = ["Glucose" , "SkinThickness" , "Insulin" ,"BloodPressure"]

for i in notzero:
    df[i] = df[i].replace(0 , np.NaN)
    m = int(df[i].mean(skipna=True))
    df[i].replace(np.NaN , m)
    
df.head()

x = df.iloc[: , 0:8]
y = df[["Outcome"]]

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 0 , test_size = 0.2 )

scx = StandardScaler()
x_train = scx.fit_transform(x_train)
x_test = scx.transform(x_test)

clas = KNeighborsClassifier(n_neighbors = 11 , p=2 , metric="euclidean")
clas.fit(x_train , y_train)

y_pred = clas.predict(x_test)
y_pred

cm = confusion_matrix(y_test , y_pred)
print(cm)
print(f1_score(y_test , y_pred))
print(accuracy_score(y_test , y_pred))

