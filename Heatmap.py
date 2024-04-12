import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

plt.close('all')

'================ Initialising constants ====================================='

'Set to True if you only want to see the Heatmap. Set to false if you want to'
'include figures that were used in the analysis'
Only_Heatmap = True 

# Define constants and initial conditions
x0 = -1                 # initial displacement
v0 = 0.0                # initial velocity
m = 1.0                 # mass
k = 1.0                 # spring constant
T = 100                 # total simulation time
dt = 0.001              # time step -  has to be lower with random kick 
omega0 = np.sqrt(k/m)   # frequency
k_T = 3                 # boltzman constant
b = 0.2                 # dampening constant
T_max = 50              # maximum temperature
N = 10                  # number of particles
c_k = 0.5               # coupling constant
k_T_counter = 0         # initialising a counter

# Parameters for the potential
a = 1/2
p = -1
c = 1/2

# Coupling term parameters
jN = np.zeros((int(T/dt),N))

#x0_array = [-1, 1, -1]
x0_array = np.ones(N)

'================ Defining functions used in calculations ===================='

# Defining functions for acceleration and potential energy
def acceleration(x, v, x_left, x_right):
    return -4*a*x**3 - 2*p*x - v*b/m + np.random.normal(0.0, sd)/m - c_k*(x_left + x_right)

def potential_energy(x, a, p, c, x_left, x_right):
    return a*x**4 + p*x**2 + c + c_k*(x_left + x_right)*x

# Defining indexes for neighbours
def index_left(l):
    if(l-1 < 0):
        return N-1
    else: 
        return l-1
    
def index_right(l):
    if(l+1 > N-1):
        return 0
    else:
        return l+1
    
def increment(T_max):
    wave = T_max/(T/dt)*2
    flip = False
    if not hasattr(increment, "counter"):
        increment.counter = 0
        
    if(increment.counter < 50 and flip == False):
        increment.counter += wave
        
    elif(increment.counter > 50):
        flip == True
        
    if(flip == True):
        increment.counter -= wave
        
    return increment.counter, flip

'================ Generating the data, plotting, and simulating =============='

#Starting a loop to run the simulation for different k_T values if needed. 
for k_T in [0]:

    '----------- Generating data ----------------------------------------------' 
    
    D = b**2-4*m*k #Value to determine the type of dampening
    print('D is = ', D)
    global counter
    #Setting up the random distribution for the temperature kick force
    sd = np.sqrt(2*k_T/b);

    # Initialising arrays
    t_array = np.arange(0, T, dt)
    xN_array = np.zeros((int(T/dt), N))
    vN_array = np.zeros_like(xN_array)
    EN_array = np.zeros_like(xN_array)
    EN_tot_array = np.zeros_like(t_array)
     
    # Set initial conditions
    for i in range (0, N):
        xN_array[0,i] = x0_array[i]
        vN_array[0,i] = v0
        EN_array[0,i] = potential_energy(x0_array[i], a, p, c, x0_array[index_left(i)], x0_array[index_right(i)]) + 0.5*m*v0**2 
    
    # Makes the array for the tenperature that rises and falls
    k_T_array = np.zeros_like(t_array)
    for i in range(0, int(T/dt)):
        if(i < int(T/dt/2)):
            k_T_array[i] = T_max/(T/dt)*2*i
        
        elif(i >= int(T/dt/2)):
            k_T_array[i] = T_max - T_max/(T/dt)*2*(i-(T/dt/2))
    
    # Verlet method
    for i in range(1, len(t_array)):
        
        k_T = k_T_array[i]
        sd = np.sqrt(2*k_T/b);
        
        for l in range(0, N):
            
            # Deciding what particle to work on
            x_array = xN_array[:,l]
            v_array = vN_array[:,l]
            E_array = EN_array[:,l]
            j = jN[:,l]
            
            # Defining the old values
            x_old = x_array[i-1]
            v_old = v_array[i-1]
            j_old = j[i-1]
            
            # Defining indexes for neighbours
            i_left = index_left(l)
            i_right = index_right(l)
            
            # Assigning the old positions of the neighbours
            x_old_left = xN_array[:,i_left][i-1]
            x_old_right = xN_array[:,i_right][i-1]
            a_old = acceleration(x_old, v_old, x_old_left, x_old_right)
            
            # Defining new values
            x_new = x_old + v_old*dt + 0.5*a_old*dt**2
            a_new = acceleration(x_new, v_old, x_old_left, x_old_right)
            v_new = v_old + 0.5*(a_old + a_new)*dt
            
            # Assigning new values
            x_array[i] = x_new
            v_array[i] = v_new
            E_array[i] = potential_energy(x_new, a, p, c, x_old_left, x_old_right) + 0.5*m*v_new**2
            EN_tot_array[i] += E_array[i]
    
            
    '----------- Plotting the data -------------------------------------------' 

    # Plotting the first particle
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(t_array, xN_array[:,0], label='Position', color='navy')
    ax1.plot(t_array, vN_array[:,0], label='Velocity', color='red')
    ax1.plot(t_array, EN_array[:,0], label='Energy', color='green')
    plt.xlabel('Time')
    
    # Title depending on the case
    if (D>0):
            plt.title('Overdamped')
                
    elif(D==0):
            plt.title('Critically Damped')
                
    elif(D<0):
            plt.title('Underdamped')
                
    plt.legend()
    plt.show()
    if (Only_Heatmap):
        plt.close()
    
    # Plotting the total energy over time
    plt.figure()
    plt.plot(t_array[1 : :], EN_tot_array[1 : :])
    plt.axvline(25, color='black', linestyle='--')
    plt.axvline(50, color='black', linestyle='--')
    plt.axvline(75, color='black', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.show()
    if (Only_Heatmap):
        plt.close()
        
    # Plotting histogram around equilibrium when it's settled
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    index = int(x_array.size*8/10) #Selecting the last 20% of the dataset
    hist_x = np.copy(x_array[index:]) 
    ax2.hist(hist_x, bins=10, color='green', edgecolor='black', linewidth=1.0)
    ax2.axis([-1.8, 1.8, 0, 1000])
    
    plt.title(f'Histogram of Positions around Equilibrium {k_T}')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.show()
    if (Only_Heatmap):
        plt.close()

    '----------- Plotting of Heatmap -----------------------------------------' 
    # Preparing a new figure
    plt.figure() 
    
    # Find the indexes that I want to use for the Histogram
    k_T_indexes = []
    for i in range(0, int(T/dt)):
        if(k_T_array[i] % 2 == 0):
            k_T_indexes.append(i)
            if(i >= int(T/dt)/2):
                val = k_T_indexes.pop()
                k_T_indexes.append(val+1)
            
    # Select the positions of the particles at those indexes        
    x_select = np.zeros((len(k_T_indexes), N))

    for i in range(0, len(k_T_indexes)):
        x_select[i] = xN_array[k_T_indexes[i]]
       
    # Creating bins and bin name for graph
    bins = []
    b_name = []
    b_start = -2.125
    while(b_start <= 2.125):
        bins.append(b_start)
        b_name.append(b_start+0.125)
        b_start += 0.25
    b_name.pop()
    
    # Creating relevant lengths to shorten notation and initialises data for heat map
    bin_len = len(bins) -1
    hist_len = len(k_T_indexes)*bin_len
    hist_data = np.zeros((hist_len, 3))
    
    # Generates data for heat map
    for i in range(0, len(k_T_indexes)):
        step = i*bin_len
        counts, bin_edges = np.histogram(x_select[i,:], bins)
        for j in range(bin_len):
            hist_data[step+j, 0] = k_T_array[k_T_indexes[i]]
            hist_data[step+j, 1] = b_name[j]
            hist_data[step+j, 2] = counts[j]
    
    # To make the heatmap, the datatype has to be changed
    hm_data = {
        "T": hist_data[:,0].astype(int),
        "Position": hist_data[:,1],
        "Count": hist_data[:,2]
        }
    df = pd.DataFrame(hm_data)
    df2 = df.pivot_table(index='Position', columns='T', values='Count', aggfunc='first')
    df2 = df2.iloc[::-1]
    df2 = df2.reindex(k_T_array[k_T_indexes].astype(int), axis=1)
    sns.set()

    #Creating the heatmap
    ax = sns.heatmap(df2)
    plt.title("Position during temperature change")
    plt.xticks()
    plt.show()
    