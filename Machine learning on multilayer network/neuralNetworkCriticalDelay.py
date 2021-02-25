import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from statistics import mean

data = np.loadtxt("Naveen bhaiya paper\\dataMLCombined.txt")
features = np.delete(data , 3 , 1)
labels = data[:,3]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

numberOfModels = int(input("Enter number of models to run: "))

tauList = []
for i in range(numberOfModels):
    model = MLPClassifier(hidden_layer_sizes=(30,30) , activation='relu')
    model.fit(X_train , y_train)

    a = np.zeros((1,3))
    a[0][0] = 1.0     #InterLayer K
    a[0][1] = 0.3     #IntraLayer K
    a[0][2] = 0

    state = 0
    while(state == 0):
        if(model.predict(a) == 1):
            state = 1
        else:
            a[0][2] += 0.01
            print(a[0][2])

    tauList.append(a[0][2])
    #print(i , end=" ")

print("\n" + str(mean(tauList)))
