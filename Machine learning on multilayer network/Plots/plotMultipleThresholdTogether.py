import matplotlib.pyplot as plt 
import numpy as np 

list005 = [0.28133333 , 0.29033333 , 0.29833333 , 0.30833333 , 0.321 , 0.337 , 0.35366667, 0.375 , 0.40066667] #Enter your values for 0.005 here
#list005 = [round(i,2) for i in list1]

list01 = [0.278 , 0.284,0.29266667,0.30166667,0.31333333,0.32766667,0.347,0.369,0.39766667] #Enter your values for 0.01 here
#list01 = [round(i,2) for i in list2] 

list02 = [0.26866667,0.277,0.288,0.29966667,0.31433333,0.33133333,0.34966667,0.37433333,0.40366667] #Enter your values for 0.02 here
#list02 = [round(i,2) for i in list3] 

list05 = [0.30366667,0.31433333,0.32633333,0.341,0.355,0.37466667,0.39633333,0.42566667,0.477]

list001 = [0.275,0.28833333,0.30233333,0.317,0.33333333,0.34833333,0.36633333,0.38666667,0.41033333]

list1 = [0.34766667,0.357,0.36666667,0.38533333,0.406,0.43,0.45833333,0.48966667,0.52866667]

line005 = plt.plot(np.arange(0.1 , 0.55 , 0.05) , list005 , marker='.')
line01 = plt.plot(np.arange(0.1 , 0.55 , 0.05) , list01 , marker='.')
line02 = plt.plot(np.arange(0.1 , 0.55 , 0.05) , list02 , marker='.')
line05 = plt.plot(np.arange(0.1 , 0.55 , 0.05) , list05 , marker='.')
line001 = plt.plot(np.arange(0.1 , 0.55 , 0.05) , list001 , marker='.')
line1 = plt.plot(np.arange(0.1 , 0.55 , 0.05) , list1 , marker='.')


font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }

plt.tick_params(direction='inout', length=6, width=2, colors='black', labelsize=19)


plt.ylabel("Critical Delay" , fontdict=font)
plt.ylim(0 , 1)
plt.xlabel("IntraLayer Coupling Constant" , fontdict=font)
plt.title("InterLayer Couping Constant = 3.82" , fontdict=font)

plt.legend((line005[0] , line01[0] , line02[0] , line05[0] , line001[0] , line1[0]), ('0.005' , '0.01' , '0.02' , '0.05' , '0.001' , '0.1') , loc='upper left')

plt.show()