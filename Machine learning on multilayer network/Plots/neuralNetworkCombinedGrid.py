import numpy as np 
import matplotlib.pyplot as plt    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from statistics import mean
import pylab as pl

data = np.loadtxt("dataMLCombined.txt")
features = np.delete(data , 3 , 1)
labels = data[:,3]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

numberOfModels = int(input("Enter number of models to run: "))
tauMatrix = np.zeros((numberOfModels , 19 , 19))

for i in range(numberOfModels):
    model = MLPClassifier(hidden_layer_sizes=(30,30) , activation='relu')
    model.fit(X_train , y_train)

    a = np.zeros((1,3))
    interIndex = 0
    for k in np.arange(2.8 , 3.7 , 0.05):
        a[0][0] = round(k , 2)      #InterLayer K

        intraIndex = 0
        for j in np.arange(0.1 , 0.575 , 0.025): 
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

            tauMatrix[i][interIndex][intraIndex] = a[0][2]
            intraIndex += 1

        interIndex += 1

#print(tauMatrix , "\n")

finalCriticalTauValues = np.mean(tauMatrix , axis=0)
print(finalCriticalTauValues)

#np.savetxt("tauGrid20.txt" , finalCriticalTauValues)


xmin = 0.1
xmax = 0.575
dx = 0.025
ymin = 2.8
ymax = 3.7
dy = 0.05

x2,y2 = np.meshgrid(np.arange(xmin,xmax+dx,dx)-dx/2.,np.arange(ymin,ymax+dy,dy)-dy/2.)
x,y = np.meshgrid(np.arange(xmin,xmax,dx),np.arange(ymin,ymax,dy))

font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=19)


pl.pcolormesh(x,y,finalCriticalTauValues)
pl.xlabel("Intralayer Coupling Constant" , fontdict=font)
pl.ylabel("Interlayer Coupling Constant" , fontdict=font)

bar = pl.colorbar()
bar.set_label(label="#Critical Delay" , weight='bold' , size=13)

pl.show()


