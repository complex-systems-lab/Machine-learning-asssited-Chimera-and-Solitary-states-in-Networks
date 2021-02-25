import matplotlib.pyplot as plt 
import numpy as np 

x = np.loadtxt("ToPlot.txt" , delimiter="\t\t\t\t")

print(x[:,0])

font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

plt.xlim(0 , 0.05)

plt.xticks([0 , 0.025 , 0.05])
plt.yticks([0 , 0.5 , 1])
plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=14)

plt.xlabel("Coupling Constant" , fontdict=font)
plt.ylabel("Order parameter" , fontdict=font)

plt.plot(x[:,0] , x[:,1] , marker="o")
plt.show()


