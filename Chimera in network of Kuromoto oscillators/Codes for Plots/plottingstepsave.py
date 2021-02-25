import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

data = np.loadtxt('Phase8data1.txt' , delimiter="\t\t")

phase = data[:,0]
freq = data[:,1]
time = data[:,2]

font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.xticks([0 , 100 , 200])
#plt.yticks([0 , 2.5 , 4.5])
plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=25)

plt.xlim(0 , 200)
#plt.ylim(-5 , 6)

plt.xlabel("Phase($\\theta_i$)" , fontdict=font)
plt.ylabel("Frequency($\dot\\theta_i$)" , fontdict=font)
plt.plot(phase , freq)

#sns.set()
#sns.scatterplot(phase , freq)
#plt.xlim(0 , 10)

plt.show()
