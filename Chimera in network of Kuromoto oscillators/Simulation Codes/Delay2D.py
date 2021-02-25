import networkx as nx 
import numpy as np 
import random
import math
import matplotlib.pyplot as plt
import time as ti

dim_lattice = int(input("Enter the size of the lattice you want: "))
graph = nx.grid_2d_graph(dim_lattice , dim_lattice , periodic=True)

Adj_matrix = nx.to_numpy_matrix(graph)      #square matrix
nodes = len(Adj_matrix)

omega = float(input("Enter the value of omega: "))
alpha = float(input("Enter the value of alpha: "))
coup_const = float(input("Enter the value of the coupling constant: "))
r = int(input("Enter the value of rounding digits for h: "))

t = np.linspace(0 , 80 , 3200)          #Time steps
time = np.size(t)
h = round(t[2] - t[1] , r)

arr_of_phases = np.zeros([time , nodes])

for i in range(nodes):
    random_number = random.uniform(0 , 2*np.pi)
    arr_of_phases[0][i] = random_number             #Initial random phases entered

i = 0

#arr_of_phases[0] = [1.9980929708849249,2.75500760674118,0.942697748674321,1.6203828420743123,5.969718763210831,4.313385483302511,1.215151602720469,4.27260749905603,4.274779170520433]
#above is for 3x3
#arr_of_phases[0] = [2.391855185238483,-1.0959416590523352,-2.1367817919276173,-0.7744640072186932,-1.653955464580573,-1.7307073556391241,2.6950356339914494,-0.839647027026245,-1.6639824346448775,-0.23414921309761816,-2.2073615091888072,2.7134377081941716,-2.3199547514608576,0.03671610871050435,-2.0770384049835915,0.5226967027193852,-3.013340505269143,0.2077569946782729,0.10973423234640522,0.15460640934328085,-1.9175188325626535,1.7036900867187486,1.3181266375441245,-0.4769427870291958,1.8124617530546896,2.142645066139732,0.15871548744488617,2.32214669415964,0.8327465596398009,-2.9381656926022113,0.0030091947023720422,1.641050508020144,-1.608963971466288,-0.09158217856586548,1.4165768077219676,-0.8667639103404858,2.6812012086392665,-1.1972710654111363,1.5852393877965607,-2.737573537853127,2.7371656542632197,-1.711738982029066,1.0165175870902088,-1.7278190330905427,-1.7577635509553562,1.969702508138094,-1.8330167233294834,1.9867278874924814,-2.8787438984733864,2.2927822058911023,-1.6402140929388458,1.3258593538820946,-1.5841012537044954,-1.1067841856376615,-2.718871052797591,3.11890650637452,-1.8376384515640543,1.2800435256517924,2.37752946465402,2.7982660047597987,0.9709956798511756,-2.5017044210749497,-1.96916220734009,0.44277869426119576,2.2672892888962544,2.192780402200862,-1.826787975478185,0.5670179663150154,0.6725580322233888,-0.9133393742143654,2.3524674448329055,-0.3277511367835104,1.864098623124657,0.9500535830084713,-0.3203926217555906,1.0112265073030278,-1.231830805472371,-0.3462764269752068,0.6564022591675553,-0.39533675342331875,1.2227533888126452,2.7984251411975496,3.1339540145044706,-0.0124062690536344,0.018670699706560434,-0.10175460789223134,-1.2318929983588562,1.3703785145917955,-0.8660195788050569,0.21540455305457806,2.635309529602213,-2.968832483175041,1.6526791169880397,1.418231906152461,2.8520872206630745,1.3069967407226937,2.075315258035139,1.4083502351333088,-3.070125500315241,-0.10682579874744436]
arr_of_phases[0] = [1.4002991446217603,4.33678409544464,3.6840252679576513,2.275414930621145,1.1670870608920825,4.520538738372099,4.832317516276672,2.4749099359891495,3.727005492687356,2.882576860089547,5.955021438345941,5.033856007115377,4.483830839166596,0.8412781810395883,4.370278518258701,4.038540805271476,2.8127574310763745,5.192015355531491,5.7393836879335005,5.923932316424973,3.3532658789745,0.3727400295216221,1.4956400547693722,2.1057187079770707,3.8251865978586403,2.8544291496613297,0.13028898892056512,5.674246533220196,1.197413547534861,3.9005649816100636,2.175056005556879,1.6641756537487176,4.155508706144167,4.3116264441924,0.32781618942726837,5.157316830490704,0.41673586815759733,1.1364889376386509,3.4489440383436443,4.774527215920203,1.2635484956060903,3.624079193840777,0.5306338977848842,0.4875352437754013,4.683785741349908,1.171730470119647,1.2920035086482182,0.7292471952893012,3.6692325440829165,6.260521635935542,0.6986957508651961,4.810929633815845,0.539108590049134,3.1311713479727463,0.2790650818162513,2.5039861151495186,1.7366767520193007,2.391258315338465,0.3740620213956376,2.4228060019685396,2.6783334310526743,2.497877323371877,3.4766836665962257,3.471179140245791,1.0230241629648718,1.9142501612219802,3.5360575859967116,3.329741278834965,4.057323325131764,6.035123487362174,0.6652945777936405,0.8319416660742538,1.6337158859721133,4.258868971905461,4.385976313408467,1.128393627249679,2.0509830850883604,2.016057039042839,6.041510540018568,3.5306276560987895,1.697697411544695,1.0300200962558288,1.6654354305655539,2.7391703829913085,2.844744227641928,1.9675513146028991,4.987598361409628,3.2919260203640355,4.086224986392764,3.6787411196401636,0.9193146993036685,0.7213252291887137,3.6122604994759326,6.230830125733253,4.459068235255846,6.273977833003344,5.93357624638183,3.1625220825835902,2.6486153022651138,5.640732951161984]
#above phases are of data 4

#arr_of_phases = np.round(arr_of_phases , decimals=4)

order_para = 0
real = 0
imag = 0
for i in range(nodes):
    real += np.cos(arr_of_phases[0][i])
    imag += np.sin(arr_of_phases[0][i])
summ = (real**2 + imag**2)/nodes**2
initial_order_para = math.sqrt(summ)
print("Initial Order parameter is: " , initial_order_para)

i = 0

b = 0
delay_node = int(input("Enter the node number to put delay: "))
delay = float(input("Enter the amount of delay to introduce: "))
algo_delay = int((delay)/h)                    #This 'h' might create problems

start = ti.clock()

def kuramoto(arr_of_phases , i):
    summation = 0
    for j in range(nodes):
        summation += Adj_matrix.item((i,j))*np.sin((arr_of_phases[j] - arr_of_phases[i]) + alpha)
    theta_dot = omega + (coup_const * summation)
    return theta_dot

for a in range(1 , time):
    #b += 1
    #print(b , end=" ")
    h = round(t[a] - t[a-1] , r)
    for i in range(nodes):
        if i == (delay_node-1):
                if a>= algo_delay:
                        k_1 = h*kuramoto(arr_of_phases[a-algo_delay-1] , i)
                        k_2 = h*kuramoto((arr_of_phases[a-algo_delay-1]+(k_1/2)) , i)
                        k_3 = h*kuramoto((arr_of_phases[a-algo_delay-1]+(k_2/2)) , i)
                        k_4 = h*kuramoto((arr_of_phases[a-algo_delay-1]+k_3) , i)

                        arr_of_phases[a][i] = arr_of_phases[a-1][i] + ((k_1+(2*k_2)+(2*k_3)+k_4)/6)
                else:
                        arr_of_phases[a][i] = arr_of_phases[a-1][i]
        else:
                k_1 = h*kuramoto(arr_of_phases[a-1] , i)
                k_2 = h*kuramoto((arr_of_phases[a-1]+(k_1/2)) , i)
                k_3 = h*kuramoto((arr_of_phases[a-1]+(k_2/2)) , i)
                k_4 = h*kuramoto((arr_of_phases[a-1]+k_3) , i)

                arr_of_phases[a][i] = arr_of_phases[a-1][i] + ((k_1+(2*k_2)+(2*k_3)+k_4)/6)

#arr_of_phases = np.round(arr_of_phases , decimals=4)

order_para = 0
real = 0
imag = 0
summ = 0
for i in range(nodes):
    real += np.cos(arr_of_phases[time-1][i])
    imag += np.sin(arr_of_phases[time-1][i])
summ = (real**2 + imag**2)/nodes**2
order_para = math.sqrt(summ)
print("Order parameter is: " , order_para)

i = 0

scaled_final_phases = np.fmod(arr_of_phases[time-1] , 2*np.pi)
scaled_final_phases = np.round(scaled_final_phases , decimals=4)
scaled_final_phases_matrix = np.reshape(scaled_final_phases , (dim_lattice,dim_lattice))

scaled_arr_phases = np.zeros((time , nodes))
for i in range(time):
        scaled_arr_phases[i] = np.fmod(arr_of_phases[i] , 2*np.pi)

i = 0
h = round(t[2] - t[1] , r)

#avg_freq = np.zeros((1 , nodes))
#for i in range(nodes):
#        freq = 0
#        d = 0
#        for j in range(time-5000 , time):                   #Select the "time steps" in long time average
#                freq += ((arr_of_phases[j+1][i]-arr_of_phases[j][i])/h)
#                d += 1
#        freq = freq/(d)
#        avg_freq[0][i] = round(freq , 4)

avg_freq = np.zeros((1 , nodes))
d = 0
for i in range(time-100 , time):               #Select the "time steps" in long time average
        avg_freq = np.add((np.subtract(arr_of_phases[i] , arr_of_phases[i-1]))/h , avg_freq)
        d += 1
avg_freq = np.round(avg_freq/(d) , decimals=4)

scaled_initial_phases = np.round(np.fmod(arr_of_phases[0] , 2*np.pi) , decimals=4)
scaled_two_phases = np.round(np.fmod(arr_of_phases[1] , 2*np.pi) , decimals=4)
scaled_two_last_phases = np.round(np.fmod(arr_of_phases[time-2] , 2*np.pi) , decimals=4)
initial_freq = np.round((np.subtract(arr_of_phases[1] , arr_of_phases[0]))/h , decimals=4)
final_freq = np.round((np.subtract(arr_of_phases[time-1] , arr_of_phases[time-2]))/h , decimals=4)

with open(file='delaydata1.txt' , mode='a') as o:
        o.write('\n'+'Lattice dimension='+str(dim_lattice)+' '+'Time='+str(t[time-1])+','+str(time)+'\n')
        o.write('omega='+str(omega)+' '+'alpha='+str(alpha)+' '+'Coup cons='+str(coup_const)+'\n')
        o.write('Delay node='+str(delay_node)+' Delay= '+str(delay)+'\n')
        o.write('Initial-'+'\n'+str(initial_order_para)+'\n'+'Phases-'+'\n')
for i in range(nodes):
        string = str(arr_of_phases[0][i])
        with open(file='delaydata1.txt' , mode='a') as o:
                o.write(string+' ')                           
with open(file='delaydata1.txt' , mode='a') as o:
        o.write('\n'+'Final-'+'\n'+str(order_para)+'\n'+'Phases-'+'\n')
for i in range(nodes):
        string = str(scaled_final_phases[i])
        with open(file='delaydata1.txt' , mode='a') as o:
                o.write(string+' ')
with open(file='delaydata1.txt' , mode='a') as o:
        o.write('\n'+'Avg final Freq-'+'\n')
for i in range(nodes):
        string = str(avg_freq[0][i])
        with open(file='delaydata1.txt' , mode='a') as o:
                o.write(string+' ')
with open(file='delaydata1.txt' , mode='a') as o:
        o.write('\n'+'Final Freq-'+'\n')
for i in range(nodes):
        string = str(final_freq[i])
        with open(file='delaydata1.txt' , mode='a') as o:
                o.write(string+' ')
with open(file='delaydata1.txt' , mode='a') as o:
        o.write('\n')

end = ti.clock()
with open(file='delaydata1.txt' , mode='a') as o:
        o.write('Code Duration= '+str(end-start)+'\n')

plt.pcolormesh(scaled_final_phases_matrix , vmin=0 , vmax=2*np.pi)
plt.colorbar()
plt.show()
