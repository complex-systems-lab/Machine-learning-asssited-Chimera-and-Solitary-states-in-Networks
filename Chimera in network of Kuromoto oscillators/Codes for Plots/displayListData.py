import numpy as np
import matplotlib.pyplot as plt 

data = np.loadtxt("CombinedDriftTemp.txt")
data[:,[0, 1]] = data[:,[1, 0]]
features = np.delete(data , 2 , 1)
labels = data[:,2]

x1 = np.arange(0 , 1 , 0.05)
x2 = np.arange(1 , 21 , 1)
xFinal = np.concatenate((x1 , x2) , axis=None)

yFinal = np.arange(0.5 , 2 , 0.075)

x , y = np.meshgrid(xFinal , yFinal)

temp = np.zeros((xFinal.size*yFinal.size))

a = 0
for i in yFinal:
    i = round(i , 3)
    for j in xFinal:
        j = round(j , 2)
        for k in range(temp.size):
            if(features[k][0] == j and features[k][1] == i):
                temp[a] = labels[k]
                #print(a , i , j)
                a += 1
                break
            else:
                pass

#print(temp)


z = np.reshape(temp , (20 , 40))
#print(z)

plt.contourf(x , y , z)
#plt.pcolormesh(z)

plt.colorbar()
plt.show()
