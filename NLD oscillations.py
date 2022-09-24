# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 02:39:23 2019

@author: Yuktee
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fftpack import fft

def f(y, t, params):
    theta, omega,t = y      # unpack current values of y
    m,g,k,f_ = params  # unpack parameters
    derivs = [omega,      # list of dy/dt=f functions
             -omega*g/m-k*theta/m+f_*np.sin(FREQ*t)
             ,1]
    return derivs


g=0.01
m=1
k=1
f_=100

# Initial values
theta0 = 0.0     # initial angular displacement
omega0 = 4    # initial angular velocity
t0=0
# Bundle parameters for ODE solver
params = [m,g,k,f_]

# Bundle initial conditions for ODE solver
y0 = [theta0, omega0,t0]

# Make time array for solution
tStop = 900
tInc = 0.05
t = np.arange(0., tStop, tInc)

N=100
ampl=np.zeros(N)
i=0
for FREQ in np.linspace(0,7,num=N):
    # Call the ODE solver
    psoln = odeint(f, y0, t, args=(params,))
    
    # Plot results
    #fig = plt.figure(1, figsize=(8,8))
    
    ampl[i]=max(psoln[:,0][-100:])
    # Plot theta as a function of time
#    ax1 = fig.add_subplot(411)
#    ax1.plot(t, psoln[:,0])
#    ax1.set_xlabel('time')
#    ax1.set_ylabel('theta')
    
    ## Plot omega as a function of time
    #ax2 = fig.add_subplot(412)
    #ax2.plot(t, psoln[:,1])
    #ax2.set_xlabel('time')
    #ax2.set_ylabel('omega')
    
    # Plot omega vs theta
#    ax3 = fig.add_subplot(413)
#    twopi = 2.0*np.pi
#    ax3.plot(psoln[:,0],psoln[:,1])
#    ax3.set_xlabel('theta')
#    ax3.set_ylabel('omega')
    i+=1
    
plt.plot(np.linspace(0,10,num=N),ampl)
#plt.set_xlabel('omega')
#plt.set_ylabel('amplitude')
#yf = fft(psoln[:,0])
#ax4 = fig.add_subplot(414)
#xf = np.linspace(0.0, 1.0/(2.0*tInc), len(psoln[:,0])//2)
#ax4.plot(xf, 2.0/len(psoln[:,0]) * np.abs(yf[0:len(psoln[:,0])//2]))
#ax4.grid()


plt.show()
