import networkx as nx 

import numpy as np 
import matplotlib.pyplot as plt 

arr = np.loadtxt("DriftMatrixEdit.txt")

delayList1 = []
delayList2 = []
coupList = []

for delay in np.arange(0 , 1 , 0.05):
    delayList1.append(delay)
for delay in np.arange(1 , 21 , 1):
    delayList2.append(delay)

delayList = np.concatenate((delayList1 , delayList2) , axis=None)
delayArr = np.asarray(delayList)

for coup in np.arange(0.5 , 2 , 0.075):
    coupList.append(coup)

coupArr = np.asarray(coupList)

x , y = np.meshgrid(delayArr , coupArr)

#plt.xticks([])
#plt.yticks([])
plt.xlabel("Delay Values")
plt.ylabel("Coupling Constant")

plt.contourf(x , y , arr)

#plt.pcolormesh(arr)

#plt.colorbar()
plt.show()