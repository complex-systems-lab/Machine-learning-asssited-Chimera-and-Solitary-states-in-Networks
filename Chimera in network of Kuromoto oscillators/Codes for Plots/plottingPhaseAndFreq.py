import matplotlib.pyplot as plt 
import numpy as np 

x = np.zeros(100)
for i in range(100):
    x[i] = i+1

y1 = np.loadtxt("ToPlot.txt")

y2 = np.zeros(100)
for i in range(100):
    y2[i] = 1.0

font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

#plt.xlim(0 , 0.05)

plt.xticks([0 , 25 , 45 , 65 , 85 , 100])
plt.yticks([1 , 1.35 , 1.7])
plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=25)

plt.xlabel("Node Index(i)" , fontdict=font)
plt.ylabel("$\\theta_i$ and $\dot\\theta_i$" , fontdict=font)

plt.plot(x , y1 , marker=".")
plt.plot(x , y2 , marker="x")

#ax = plt.subplot(111)
#ax.plot(x, y1, label='= Phase' , marker='.')
#ax.plot(x, y2, label='= Frequency' , marker='x')
#ax.legend()

plt.show()


