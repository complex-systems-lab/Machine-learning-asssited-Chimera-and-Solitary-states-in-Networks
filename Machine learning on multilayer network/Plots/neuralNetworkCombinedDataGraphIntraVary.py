import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from statistics import mean

data = np.loadtxt("dataMLCombined.txt")
features = np.delete(data , 3 , 1)
labels = data[:,3]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

numberOfModels = int(input("Enter number of models to run: "))
tauMatrix = np.zeros((numberOfModels,9))

for i in range(numberOfModels):
    model = MLPClassifier(hidden_layer_sizes=(30,30) , activation='relu')
    model.fit(X_train , y_train)

    a = np.zeros((1,3))
    a[0][0] = 3.82      #InterLayer K
    
    index = 0
    for j in np.arange(0.1 , 0.55 , 0.05): 
        a[0][1] = round(j , 2)      #IntraLayer K
        a[0][2] = 0

        state = 0
        while(state == 0):
            if(model.predict(a) == 1):
                state = 1
            else:
                a[0][2] += 0.01

            if(a[0][2] > 4.5):
                break
            else:
                pass

        tauMatrix[i][index] = a[0][2]
        index += 1

#print(tauMatrix)
tauCriticalValues = np.mean(tauMatrix , axis=0)

print(tauCriticalValues)

font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=19)



plt.plot(np.arange(0.1 , 0.55 , 0.05) , tauCriticalValues , marker='.')
plt.ylabel("Critical Delay" , fontdict=font)
plt.xlabel("IntraLayer Coupling Constant" , fontdict=font)
#plt.title("InterLayer Couping Constant = 2.81" , fontdict=font)

plt.show()


