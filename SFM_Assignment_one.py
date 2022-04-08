# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:43:53 2020

@author: Subbu
"""
import numpy as np
from numpy import sin, cos,sqrt,pi,arctan,tan,degrees
from numpy import radians
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#function for Newton Raphson Method
def raph_f(E,M,e):
	return E - e*sin(E) -M
#Derivative of the previous function
def raphdash_f(E,e):
	return 1- e*cos(E)
#Recursive solver to find Eccentric Anomaly
def solve_kepler(E0,M,e,error):
    E = E0
    delta = raph_f(E,M,e)
    c =abs(delta)
    if c-error > 0:
        E1 = E - (raph_f(E,M,e)/raphdash_f(E,e))
        return solve_kepler(E1,M,e,error)
    else:
        return E
#Function to find TA given EA and e
def new(E,e):
    return 2*arctan((sqrt((1+e)/(1-e)))*tan(E*0.5))
#Function to convert orbital elemants to state vector     
def orbit_2_state(xp,xpd,yp,ypd,zp,zpd,a,e,i,o,O,n):
    R11 = (cos(O)*cos(o)) - (sin(O)*sin(o)*cos(i))
    R12 = (cos(O)*sin(o)) + (sin(O)*cos(o)*cos(i))
    R13 = sin(i)*sin(O)
    R21 = (sin(O)*cos(o)) + (cos(O)*sin(o)*cos(i))
    R22 = (cos(O)*cos(o)*cos(i)) - (sin(O)*sin(o))	
    R23 = -sin(i)*cos(O)
    R31 = (sin(o)*sin(i))
    R32 = (cos(o)*sin(i))
    R33 = cos(i)
    x = R11*xp - R12*yp + R13*zp
    y = R21*xp + R22*yp + R23*zp
    z = R31*xp + R32*yp + R33*zp
    xd = R11*xpd - R12*ypd + R13*zpd
    yd = R21*xpd + R22*ypd + R23*zpd
    zd = R31*xpd + R32*ypd + R33*zpd
    return [x,y,z,xd,yd,zd]
#defining variables for the orbit	
a = 25000 #km
e = 0.1 
error = 10**(-6) #order of acceptable error in EA
mu = 398600 #km3/s
H = sqrt(mu*a*(1 - e**2))  
n = sqrt(mu/a**3)
T = sqrt((4*(pi**2)*(a**3))/mu)
t = np.linspace(0,T,int(T))
inc = radians(40) #Inclination
omega = radians(120) #Angle of Periapsis
Omega = radians(250) #Right Ascension of Ascending node
#Finding M values for the orbit
M = np.zeros(len(t)) 
M[0] = 0.0
for i in range(1,len(t)):
    M[i] = n*(t[i] - t[0])
#Using Recursive function finding EA values 
E = np.zeros(len(t))
E[0] = 0.0
for i in range(1,len(t)):
    E[i] = solve_kepler(M[i],M[i],e,error)   
#TA array
nur = np.zeros(len(t))
for i in range(0,len(t)):
    if new(E[i],e) > 0 :
        nur[i] = new(E[i],e)
    else :
        nur[i] = 2*3.14 + new(E[i],e)

#TA in degrees  
nud = np.zeros(len(t))
for i in range(0,len(t)):
    if new(E[i],e) > 0 :
        nud[i] = degrees(new(E[i],e))
    else :
        nud[i] = degrees(2*3.14 + new(E[i],e))

#Arrays of components of the state vectors at different instants during the orbit            
x=[]
y=[]
z=[]
xd=[]
yd=[]
zd=[]
for i in range(0,len(t)):
    r = (a*(1 - (e**2)))/(1 + e*cos(nur[i]))
    xp = r*cos(nur[i])
    xpd = (-mu/H)*sin(nur[i])
    yp =  r*sin(nur[i])
    ypd =  (mu/H)*(e + cos(nur[i]))
    zp = 0
    zpd = 0
    soln = orbit_2_state(xp,xpd,yp,ypd,zp,zpd,a,e,inc,omega,Omega,nur[i])  
    x.append(soln[0])    
    y.append(soln[1]) 
    z.append(soln[2])  
    xd.append(soln[3]) 
    yd.append(soln[4]) 
    zd.append(soln[5]) 

#Finding some other relevant quantities using the state vector
Md = []#Mean Anomaly in degrees
Ed =[]# Eccentric Anomaly in degrees
r = []#Radial distance
v = []#Velocity
gamma = []#Flight path angle
vr = []#radial velocity
vt = []#tangential velocity
Hexp = []# Value of angluar momentum from the state vectors
for i in range(0,len(t)):
    r.append(sqrt(x[i]**2 + y[i]**2 + z[i]**2))
    v.append(sqrt(xd[i]**2 + yd[i]**2 + zd[i]**2))
    gamma.append(arctan((e*sin(nur[i]))/(1 + e*cos(nur[i]))))
    vr.append(v[i]*sin(gamma[i]))
    vt.append(v[i]*cos(gamma[i]))
    Hexp.append(r[i]*vt[i])
    Ed.append(E[i]*180/pi)
    Md.append(M[i]*180/pi)

#Plotting relevant figures
'''fig = plt.figure(1, figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z)
ax.set_xlabel('X',size='16')
ax.set_ylabel('Y',size='16')
ax.set_zlabel('Z',size='16')'''

fig = plt.figure(1, figsize=(8,8))
ax3 = fig.add_subplot(311)
ax3.plot(nud[1:],vt[1:],'r',label ='Vt')
ax3.plot(nud[1:],vr[1:],'b',label = 'Vr')
ax3.set_xlabel('True Anomaly (deg)',size='16')
ax3.set_ylabel('Velocity (km/s)',size='16')
ax3.legend(loc='best')
#plt.ylim(-0.6,1.0)


