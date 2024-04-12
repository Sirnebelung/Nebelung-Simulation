import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

plt.close('all')

'================ Initialising constants ====================================='

'Set to True if you only want to see the animation. Set to false if you want to'
'include figures that were used in the analysis'
Only_animation = True

# Define constants and initial conditions
x0 = -1             # initial displacement
v0 = 0.0              # initial velocity
m = 1.0               # mass
k = 1.0               # spring constant
T = 25                # total simulation time
dt = 0.001            # time step
omega0 = np.sqrt(k/m) # frequency
k_T = 3               # boltzman constant - can be overriden in line 86
b = 0.2               # dampening constant
T_max = 50            # maximum temperature
N = 7                 # number of particles
c_k = 0.5             # coupling constant

# Parameters for the potential
a = 1/2
p = -1
c = 1/2

# Coupling term parameters
jN = np.zeros((int(T/dt),N))

# Initialising the starting positions
x0_array = np.ones(N)*1.3
for i in range(0, N, 2):
    x0_array[i] = x0_array[i]*(-1.2)
k_T_counter = 0

'================ Defining functions used in calculations ===================='

# Defining functions for acceleration and potential energy
def acceleration(x, v, x_left, x_right):
    return -4*a*x**3 - 2*p*x - v*b/m*0 + np.random.normal(0.0, sd)/m*0 - c_k*(x_left + x_right)*0.5

def potential_energy(x, a, p, c, x_left, x_right):
    return a*x**4 + p*x**2 + c + c_k*(x_left + x_right)*x*0.5

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

# Defining a function to gradually increase the temperature
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
    
    # Verlet method - This is a numerical method to generate the next step in the particles motion
    for i in range(1, len(t_array)):
        
        k_T = k_T_array[i]*0 
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
    
    
    # Title depending on the type of dampening
    if (D>0):
            plt.title('Overdamped')
                
    elif(D==0):
            plt.title('Critically Damped')
                
    elif(D<0):
            plt.title('Underdamped')
                
    plt.legend()
    plt.show()
    if (Only_animation):
        plt.close()
        
    # Plotting the total energy over time    
    plt.figure()
    plt.plot(t_array[1 : :], EN_tot_array[1 : :])
    if (Only_animation):
        plt.close()
    
    # Plotting a histogram around the particle's equilibrium when it has settled at the new energy
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    index = int(x_array.size*8/10) #Selecting the last 20% of the dataset
    hist_x = np.copy(x_array[index:]) 
    ax2.hist(hist_x, bins=100, color='green', edgecolor='black', linewidth=1.0)
    ax2.axis([-1.8, 1.8, 0, 1000])
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.title(f'Histogram of Positions around Equilibrium {k_T}')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.show()
    if (Only_animation):
        plt.close()
    
    # Plotting the potential
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    x_p = np.linspace(-2, 2, 5000)
    ax3.plot(x_p, potential_energy(x_p, a, p, c, x0_array[0], x0_array[2]))
    plt.show()
    if (Only_animation):
        plt.close()
    
    # Animating average position
    fig4, ax4 = plt.subplots(nrows=1, ncols=N)
    y_aniN = np.zeros_like(xN_array)
    title0 = plt.text(0,4,'T = 0', alpha = 0, ha = 'center', va='center', verticalalignment = 'top')
    
    # Creating the initial figure and plots
    for i in range(0, N):
        plt.subplot(1,N,i+1)
        y_aniN[:,i] = potential_energy(xN_array[:,i], a, p , c, x0_array[index_left(i)], x0_array[index_right(i)])
        
    '----------- Manual plotting ----------------------------------------------' 
    'Everything below doesnt scale with N (number of particles) and has to be done manaully'  
    'While it can be automated, it was quicker to do it manually and use the time elsewhere'
    
    plt.subplot(1,7,1)
    ball0, = plt.plot(xN_array[0,0], y_aniN[0,0], 'o', markersize=15, alpha=0)
    line0, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[2], x0_array[1]), alpha=0, linewidth = 2.5)
    plt.axis([-2.2, 2.2, -1.1, 2])
    plt.ylabel('Potential Energy', fontsize = 25)
    
    plt.subplot(1,7,2)
    ball1, = plt.plot(xN_array[0,1], y_aniN[0,1], 'o', markersize=15, alpha=0)
    line1, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[0], x0_array[2]), alpha=0, linewidth = 2.5)
    plt.axis([-2.2, 2.2, -1.1, 2])
    
    plt.subplot(1,7,3)
    ball2, = plt.plot(xN_array[0,2], y_aniN[0,2], 'o', markersize=15, alpha=0)
    line2, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[1], x0_array[0]), alpha=0,linewidth = 2.5)
    plt.axis([-2.2, 2.2, -1.1, 2])
    
    plt.subplot(1,7,4)
    ball3, = plt.plot(xN_array[0,2], y_aniN[0,2], 'o', markersize=15, alpha=0)
    line3, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[2], x0_array[4]), alpha=0, linewidth = 2.5)
    plt.axis([-2.2, 2.2, -1.1, 2])
    plt.xlabel('Displacement', fontsize = 25)
    plt.title('Simulation of 7 particles \n ', fontsize = 25)
    
    plt.subplot(1,7,5)
    ball4, = plt.plot(xN_array[0,2], y_aniN[0,2], 'o', markersize=15, alpha=0)
    line4, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[3], x0_array[5]), alpha=0, linewidth = 2.5)
    plt.axis([-2.2, 2.2, -1.1, 2])
    
    plt.subplot(1,7,6)
    ball5, = plt.plot(xN_array[0,2], y_aniN[0,2], 'o', markersize=15, alpha=0)
    line5, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[4], x0_array[6]), alpha=0, linewidth = 2.5)
    plt.axis([-2.2, 2.2,-1.1, 2])
    
    plt.subplot(1,7,7)
    ball6, = plt.plot(xN_array[0,2], y_aniN[0,2], 'o', markersize=15, alpha=0)
    line6, = plt.plot(x_p, potential_energy(x_p, a, p, c, x0_array[5], x0_array[0]), alpha=0, linewidth = 2.5)
    plt.axis([-2.2, 2.2,-1.1, 2])
    
    
    # Creating the data for the potential energy, also sets the potential as its current neighbours rather
    # than its neighbours at the previous step
    Epot_array = np.zeros_like(xN_array)
    for i in range(0, len(t_array)):
        for l in range(0,N):
            l_left = index_left(l)
            l_right = index_right(l)
            Epot_array[i,l] = potential_energy(xN_array[i,l], a, p, c, xN_array[i, l_left], xN_array[i, l_right])

    '----------- Animating the data -------------------------------------------'     
    
    # Defining the function that returns the next frame
    def update(frame):
        
        # Making the elements visible
        ball0.set_alpha(1)
        ball1.set_alpha(1)
        ball2.set_alpha(1)
        ball3.set_alpha(1)
        ball4.set_alpha(1)
        ball5.set_alpha(1)
        ball6.set_alpha(1)
        
        line0.set_alpha(1)
        line1.set_alpha(1)
        line2.set_alpha(1)
        line3.set_alpha(1)
        line4.set_alpha(1)
        line5.set_alpha(1)
        line6.set_alpha(1)

        # Selecting the next frame to be plotted
        ball0.set_xdata([xN_array[:,0][frame]])
        line0.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,6][frame], xN_array[:,1][frame]))
        ball0.set_ydata([Epot_array[:,0][frame]])
        
        ball1.set_xdata([xN_array[:,1][frame]])
        line1.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,0][frame], xN_array[:,2][frame]))
        ball1.set_ydata([Epot_array[:,1][frame]])
        
        ball2.set_xdata([xN_array[:,2][frame]])
        line2.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,1][frame], xN_array[:,3][frame]))
        ball2.set_ydata([Epot_array[:,2][frame]])
        
        ball3.set_xdata([xN_array[:,3][frame]])
        line3.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,2][frame], xN_array[:,4][frame]))
        ball3.set_ydata([Epot_array[:,3][frame]])
        
        ball4.set_xdata([xN_array[:,4][frame]])
        line4.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,3][frame], xN_array[:,5][frame]))
        ball4.set_ydata([Epot_array[:,4][frame]])
        
        ball5.set_xdata([xN_array[:,5][frame]])
        line5.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,4][frame], xN_array[:,6][frame]))
        ball5.set_ydata([Epot_array[:,5][frame]])
        
        ball6.set_xdata([xN_array[:,6][frame]])
        line6.set_ydata(potential_energy(x_p, a, p, c, xN_array[:,5][frame], xN_array[:,0][frame]))
        ball6.set_ydata([Epot_array[:,6][frame]])
        
        title0.set_alpha(1)
        k_T_next = k_T_array[frame]
        k_T_next = "{:.1f}".format(k_T_next)
        title0.set_text(f'T = {k_T_next}')
        
        return line0, line1, line2, line3, line4, line5, line6, ball0, ball1, ball2, ball3, ball4, ball5, ball6, title0
    
    # Running the animation
    ani = FuncAnimation(fig4, update, frames=len(xN_array)-1, interval=1, blit=True)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
