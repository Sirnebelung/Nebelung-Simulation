import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

'================ Initialising constants ====================================='

# Define constants and initial conditions
x0 = 1.0               # initial displacement
v0 = 0.0               # initial velocity
m = 1.0                # mass
k = 1.0                # spring constant
T = 20.0               # total simulation time
dt = 0.001             # time step
omega0 = np.sqrt(k/m)  # frequency

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

'================ Defining functions used in calculations ===================='

# Defining functions for acceleration and potential energy
def acceleration(x, v):
    return -2*x*k/m - v*b/m

def potential_energy(x):
    return k*x**2

'================ Generating the data ========================================'
    
# Looping over different dampening constants to create the 3 types of dampening
for b in [0.2, 2, 2.5]: 
    
    D = b**2-4*m*k
    print('D is = ', D)

    # Initialising arrays
    t_array = np.arange(0, T, dt)
    x_array = np.arange(0, T, dt)
    v_array = np.arange(0, T, dt)
    E_array = np.arange(0, T, dt)

    # Set initial conditions
    x_array[0] = x0
    v_array[0] = v0
    E_array[0] = potential_energy(x0) + 0.5*m*v0**2
    
    # Verlet method
    for i in range(1, len(t_array)):
        # Defining the old values
        x_old = x_array[i-1]
        v_old = v_array[i-1]
        a_old = acceleration(x_old, v_old)
        
        # Defining new values
        x_new = x_old + v_old*dt + 0.5*a_old*dt**2
        a_new = acceleration(x_new, v_old)
        v_new = v_old + 0.5*(a_old + a_new)*dt
        
        # Assigning new values
        x_array[i] = x_new
        v_array[i] = v_new
        E_array[i] = potential_energy(x_new) + 0.5*m*v_new**2

    '================ Plotting the data ======================================'
    
    if(D < 0):
        ax1.plot(t_array, x_array, label='Underdamped', color='green')
    if(D == 0):
        ax1.plot(t_array, x_array, label='Critically Damped', color='red')
    if(D > 0):
        ax1.plot(t_array, x_array, label='Overdamped', color='navy')
    
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Verlet Method')
    plt.legend()
    plt.show()
                
'================ Analytical Solutions ======================================='

print("--------------") # Line for the console
# Making time steps the same size as the Verlet method
t_an = np.linspace(0, 20, 2000);
plt.figure()

'--------------- Underdamped -------------------------------------------------'

# Constants from theory
b_ud = 0.2; 
D_ud = b_ud**2-4*m*k
gamma_ud = 1/2 * np.sqrt((4*(m*k)**2 - b_ud**2))
A_ud = x0
B_ud = b_ud*x0/(2*gamma_ud) + v0/gamma_ud

# Equation for the underdamped case
x_ud = np.exp(-(b_ud/2)*t_an)*(A_ud*np.cos(gamma_ud*t_an) + B_ud*np.sin(gamma_ud*t_an))

# Plotting the results
plt.plot(t_an, x_ud, color='green', label='Underdamped')

print('D is = ' , D_ud)
    
'--------------- Critically Damped -------------------------------------------'

# Constants from theory
b_cd = 2.0*np.sqrt(k/m)
D_cd = b_cd**2-4*m*k
A_cd = x0
B_cd = v0 + m*k*x0

# Equation for the critically damped case
x_cd = (A_cd + B_cd*t_an)*np.exp(-m*k*t_an)

# Plotting the results
plt.plot(t_an, x_cd, color='red', label='Critically Damped')

print('D is = ' , D_cd)

'--------------- Overdamped --------------------------------------------------'

# Constants from theory
b_od = 2.5;
D_od = b_od**2-4*m*k
r_1 = 1/2*(-b_od + np.sqrt(b_od**2 - 4*(m*k)**2))
r_2 = 1/2*(-b_od - np.sqrt(b_od**2 - 4*(m*k)**2))
c_1 = (v0 + r_1*x0)/(2*np.sqrt(b_od**2 - k/m))
c_2 = x0 - c_1

# Equation for the overdamped case
x_od = c_1*np.exp(r_2*t_an) + c_2*np.exp(r_1*t_an);

# Plotting the results
plt.plot(t_an, x_od, color='navy', label='Overdamped')

print('D is = ' , D_od)

'--------------- Editing combined plot ---------------------------------------'

plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Analytical Solution')
plt.legend()
plt.show()
