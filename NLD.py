

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def f(y, t, params):
    theta, omega,t = y      # unpack current values of y
    m,g,k,f_ = params  # unpack parameters
    derivs = [omega,      # list of dy/dt=f functions
             -omega*g/m-k*theta/m+f_*np.sin(5*t)
             ,1]
    return derivs

#underdamped
#g=0.01
#m=1
#k=1
#f_=0

#overdamped
#g=1.9
#m=1
#k=1
#f_=0

#critically
g=2
m=1
k=1
f_=0

# Initial values
theta0 = 6.0     # initial angular displacement
omega0 = 0    # initial angular velocity
t0=0
# Bundle parameters for ODE solver
params = [m,g,k,f_]

# Bundle initial conditions for ODE solver
y0 = [theta0, omega0,t0]

# Make time array for solution
#tStop = 9000                #underdamped
tStop=90                     #overdamped
tInc = 0.05
t = np.arange(0., tStop, tInc)


# Call the ODE solver
psoln = odeint(f, y0, t, args=(params,))

# Plot results
fig = plt.figure(1, figsize=(8,8))


# Plot theta as a function of time
ax1 = fig.add_subplot(211)
ax1.plot(t, psoln[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('theta')

## Plot omega as a function of time
#ax2 = fig.add_subplot(412)
#ax2.plot(t, psoln[:,1])
#ax2.set_xlabel('time')
#ax2.set_ylabel('omega')

# Plot omega vs theta
ax3 = fig.add_subplot(212)
ax3.plot(psoln[:,0],psoln[:,1])
ax3.set_xlabel('theta')
ax3.set_ylabel('omega')


#plt.set_xlabel('omega')
#plt.set_ylabel('amplitude')
#yf = fft(psoln[:,0])
#ax4 = fig.add_subplot(414)
#xf = np.linspace(0.0, 1.0/(2.0*tInc), len(psoln[:,0])//2)
#ax4.plot(xf, 2.0/len(psoln[:,0]) * np.abs(yf[0:len(psoln[:,0])//2]))
#ax4.grid()


plt.show()
