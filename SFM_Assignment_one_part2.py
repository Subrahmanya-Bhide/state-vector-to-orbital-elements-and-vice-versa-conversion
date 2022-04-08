# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:19:39 2020

@author: subbu
"""
import numpy as np
from numpy import sin, cos,sqrt,pi,arctan,tan,degrees,arccos,arctan2
from numpy import radians
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits import mplot3d
#Defining general functions for some vector operations
#dot product
def dot(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
#magnitude of the vector
def modulus(a):
    return sqrt(a[0]**2 + a[1]**2 + a[2]**2)
#Scalar multiplication of a vector 
def sm(a,b):
    return [a*b[0] , a*b[1] , a*b[2] ]
#negatie of a vector
def n(a):
    return [-a[0] , -a[1] , -a[2]]
#sum of 3 vectors
def Sum(a,b,c):
    return [(a[0] + b[0] +c[0]) , (a[1] + b[1] +c[1]) , (a[2] + b[2] +c[2]) ]    
#cross product
def cross(a,b):
    return [(a[1]*b[2] - a[2]*b[1]) , (a[2]*b[0] - a[0]*b[2]) , (b[1]*a[0] - a[1]*b[0])]

#Function to convert state vector to orbital elemants
def state_2_orbit(x,y,z,xd,yd,zd):
    rvec = [x,y,z]
    vvec = [xd,yd,zd]
    r = modulus(rvec)
    v = modulus(vvec)
    rc = sm((1/modulus(rvec)),rvec)
    a = (mu*r)/(2*mu - r*v**2)
    evec =  sm(1/mu , Sum(sm(v**2,rvec),n(sm(dot(rvec,vvec),vvec)), n(sm(mu/r,rvec)))) 
    e = modulus(evec)
    ec = sm((1/modulus(evec)),evec)
    wc = sm((1/modulus(cross(rvec,vvec))),cross(rvec,vvec))
    Nc = sm((1/modulus(cross(kc,wc))),cross(kc,wc))
    pc = cross(ec,rc)
    if arccos(dot(wc,kc)) > 0:
        inc = arccos(dot(wc,kc))
    else :
        inc = 2*pi + arccos(dot(wc,kc))
        
    if arctan2(dot(Nc,jc),dot(ic,Nc)) > 0:
        O = arctan2(dot(Nc,jc),dot(ic,Nc))
    else :
        O = 2*pi + arctan2(dot(Nc,jc),dot(ic,Nc))
    
    if arctan2(dot(wc,cross(Nc,ec)),dot(Nc,ec)) > 0:
        o = arctan2(dot(wc,cross(Nc,ec)),dot(Nc,ec))
    else :
        o = 2*pi + arctan2(dot(wc,cross(Nc,ec)),dot(Nc,ec))
        
        
    if arctan2(dot(pc,wc),dot(ec,rc)) > 0:
        nu = arctan2(dot(pc,wc),dot(ec,rc))
    else :
        nu = 2*pi + arctan2(dot(pc,wc),dot(ec,rc))
    
    return [a,e,inc,O,o,nu]

def raph_f(E,M,e):
	return E - e*sin(E) -M

def raphdash_f(E,e):
	return 1- e*cos(E)

def solve_kepler(E0,M,e,error):
    E = E0
    delta = raph_f(E,M,e)
    c =abs(delta)
    if c-error > 0:
        E1 = E - (raph_f(E,M,e)/raphdash_f(E,e))
        return solve_kepler(E1,M,e,error)
    else:
        return E
def new(E,e):
    return 2*arctan((sqrt(1+e)/sqrt(1-e))*tan(E*0.5))

def Ecc(n,e):
    return 2*arctan((sqrt(1-e)/sqrt(1+e))*tan(n*0.5))
  
def Mean(E,e):
    return E - e*sin(E)
    
def orbit_2_state(xp,xpd,yp,ypd,zp,zpd,a,e,i,o,O,n):
	R11 = cos(O)*cos(o) - sin(O)*sin(o)*cos(i)
	R12 = -cos(O)*sin(o) - sin(O)*cos(o)*cos(i)
	R13 = sin(i)*sin(O)
	R21 = sin(O)*cos(o) + cos(O)*sin(o)*cos(i)
	R22 = -sin(O)*sin(o) + cos(O)*cos(o)*cos(i)	
	R23 = -sin(i)*cos(O)
	R31 = sin(o)*sin(i)
	R32 = cos(o)*sin(i)
	R33 = cos(i)
	
	R = np.array([[R11,R12,R13] , [R21,R22,R23] ,[R31,R32,R33]])
	
	rp = np.array([xp,yp,zp])
	rpd = np.array([xpd,ypd,zpd])	

	r = np.matmul(R,rp)
	rd = np.matmul(R,rpd)

	return [r[0],r[1],r[2],rd[0],rd[1],rd[2]]

#defining given conditions and relevant constants  
mu = 398600 #km3/s
ic = [1,0,0]
jc = [0,1,0]
kc = [0,0,1]

x1 = -8503.8558701333313
y1 = 14729.110427313784 
z1 = 6190.3008264368991 
x1d = -4.3229148869407341       
y1d = -2.3293249804847838 
z1d = 5.24855402558600526*0.01

x2 = -13686.889393418738
y2 = -13344.772667428870
z2 = 10814.629905439588
x2d = 0.88259108105901152
y2d = 1.9876415852134037         
z2d = 3.4114313525042017

#Obtaining orbital elemants
orbit_element_1 = state_2_orbit(x1,y1,z1,x1d,y1d,z1d)
orbit_element_2 = state_2_orbit(x2,y2,z2,x2d,y2d,z2d)

a1 = orbit_element_1[0]
e1 = orbit_element_1[1]
i1 = orbit_element_1[2]
O1 = orbit_element_1[3]
o1 = orbit_element_1[4]
nu1 = orbit_element_1[5]
a2 = orbit_element_2[0]
e2 = orbit_element_2[1]
i2 = orbit_element_2[2]
O2 = orbit_element_2[3]
o2 = orbit_element_2[4]
nu2 = orbit_element_2[5]



E1 = Ecc(nu1,e1)
E2 = Ecc(nu2,e2)
M1 = Mean(E1,e1)
M2 = Mean(E2,e2)

n1 = sqrt(mu/a1**3)
n2 = sqrt(mu/a2**3)
#Obtaining MA,TA,EA after 1000s
M1d = n1*1000 + M1
M2d = n2*1000 + M2

E1d = solve_kepler(M1d,M1d,e1,0.000001)
E2d = solve_kepler(M2d,M2d,e2,0.000001)

nu1d = new(E1d,e1)
nu2d = new(E2d,e2)

H1 = sqrt(mu*a1*(1 - e1**2))
H2 = sqrt(mu*a2*(1 - e2**2))

r1 = a1*(1 - e1**2)/(1 + e1*cos(nu1d))
xp1 = r1*cos(nu1d)
xpd1 = -(mu/H1)*sin(nu1d)
yp1 = r1*sin(nu1d)
ypd1 =  (mu/H1)*(e1 + cos(nu1d))
zp1 = 0
zpd1 = 0

r2 = a2*(1 - e2**2)/(1 + e2*cos(nu2d))
xp2 = r2*cos(nu2d)
xpd2 = -(mu/H2)*sin(nu2d)
yp2 = r2*sin(nu2d)
ypd2 =  (mu/H2)*(e2 + cos(nu2d))
zp2 = 0
zpd2 = 0
#Finding state vectors after 1000s
state_vec_d_1 = orbit_2_state(xp1,xpd1,yp1,ypd1,zp1,zpd1,a1,e1,i1,o1,O1,nu1d)
state_vec_d_2 = orbit_2_state(xp2,xpd2,yp2,ypd2,zp2,zpd2,a2,e2,i2,o2,O2,nu2d)

a = a1 #km
e = e1
error = 10**(-6)
mu = 398600 #km3/s
H = sqrt(mu*a*(1 - e**2))
nd = sqrt(mu/a**3)
T = sqrt((4*(pi**2)*(a**3))/mu)
t = np.linspace(0,T,int(T))
inc = i1 #40
omega = o1 #120
Omega = O1 #250
M = np.zeros(len(t))
M[0] = 0.0
for i in range(1,len(t)):
    M[i] = nd*(t[i] - t[0])
E = np.zeros(len(t))
E[0] = 0.0
for i in range(1,len(t)):
    E[i] = solve_kepler(M[i],M[i],e,error)    
nur = np.zeros(len(t))
'''for i in range(0,len(t)):
    if new(E[i],e) > 0 :
        nur[i] = new(E[i],e)
    else :
        nur[i] = 2*3.14 + new(E[i],e)'''
for i in range(0,len(t)):
    nur[i] = new(E[i],e)
    
nud = np.zeros(len(t))
for i in range(0,len(t)):
    if new(E[i],e) > 0 :
        nud[i] = degrees(new(E[i],e))
    else :
        nud[i] = degrees(2*3.14 + new(E[i],e))
            
state_vec = []
x1=[]
y1=[]
z1=[]
xd1=[]
yd1=[]
zd1=[]
for i in range(0,len(t)):
    r = (a*(1 - (e**2)))/(1 + e*cos(nur[i]))
    xp = r*cos(nur[i])
    xpd = (-mu/H)*sin(nur[i])
    yp =  r*sin(nur[i])
    ypd =  (mu/H)*(e + cos(nur[i]))
    zp = 0
    zpd = 0
    soln = orbit_2_state(xp,xpd,yp,ypd,zp,zpd,a,e,inc,omega,Omega,nur[i])
    state_vec.append(soln)    
    x1.append(soln[0])    
    y1.append(soln[1]) 
    z1.append(soln[2])  
    xd1.append(soln[3]) 
    yd1.append(soln[4]) 
    zd1.append(soln[5])

a = a2 #km
e = e2
error = 10**(-6)
mu = 398600 #km3/s
H = sqrt(mu*a*(1 - e**2))
nd = sqrt(mu/a**3)
T = sqrt((4*(pi**2)*(a**3))/mu)
t = np.linspace(0,T,int(T))
inc = i2 #40
omega = o2 #120
Omega = O2 #250
M = np.zeros(len(t))
M[0] = 0.0
for i in range(1,len(t)):
    M[i] = nd*(t[i] - t[0])
E = np.zeros(len(t))
E[0] = 0.0
for i in range(1,len(t)):
    E[i] = solve_kepler(M[i],M[i],e,error)    
nur = np.zeros(len(t))
for i in range(0,len(t)):
    if new(E[i],e) > 0 :
        nur[i] = new(E[i],e)
    else :
        nur[i] = 2*3.14 + new(E[i],e)
 
nud = np.zeros(len(t))
for i in range(0,len(t)):
    if new(E[i],e) > 0 :
        nud[i] = degrees(new(E[i],e))
    else :
        nud[i] = degrees(2*3.14 + new(E[i],e))
            
state_vec = []
x2=[]
y2=[]
z2=[]
xd2=[]
yd2=[]
zd2=[]
for i in range(0,len(t)):
    r = (a*(1 - (e**2)))/(1 + e*cos(nur[i]))
    xp = r*cos(nur[i])
    xpd = (-mu/H)*sin(nur[i])
    yp =  r*sin(nur[i])
    ypd =  (mu/H)*(e + cos(nur[i]))
    zp = 0
    zpd = 0
    soln = orbit_2_state(xp,xpd,yp,ypd,zp,zpd,a,e,inc,omega,Omega,nur[i])
    state_vec.append(soln)    
    x2.append(soln[0])    
    y2.append(soln[1]) 
    z2.append(soln[2])  
    xd2.append(soln[3]) 
    yd2.append(soln[4]) 
    zd2.append(soln[5])
    
#plotting relevant plots
fig = plt.figure(1, figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot3D(x2, y2, z2, 'b' ,label = 'sat2')
ax.plot3D(x1, y1, z1,'r',label ='sat1')
ax.set_xlabel('X',size='16')
ax.set_ylabel('Y',size='16')
ax.set_zlabel('Z',size='16')
ax.legend(loc="best")




