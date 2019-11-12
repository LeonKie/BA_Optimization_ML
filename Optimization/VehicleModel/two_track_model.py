import matplotlib.pyplot as plt
import numpy as np
import time
import sys
#--------------------------------------------------------------------------
#Fnde different methode __init__.py ??
sys.path.insert(1, 'BA_Optimization_ML/Optimization/helper_functions')
#--------------------------------------------------------------------------

from getconfig import getConfig


def get_next_states_two_track_model(x,dx,t):
    
    erg= np.add(x,dx*t)
    #print("X: \n", x , "\ndx \n", dx,"\n ERG: \t ", erg,"\n")
    return erg



def get_dot_states_two_track_model(x,p,u):  


    g=9.81 
    mu=0.75 #should be dynamic in the future

    #--------------------------------------------------------------------------
    #p : parameter
    #x : states
    #--------------------------------------------------------------------------

    # Retrieve model parameters.
    m  = p[0];   # Vehicle mass.                    
    a  = p[1];   # Distance from front axle to COG. 
    b  = p[2];   # Distance from rear axle to COG.
    maxP = p[3]  # Maximum Power accelleration
    Cx = p[4];   # Longitudinal tire stiffness.     
    Cy = p[5];   # Lateral tire stiffness.      
    CA = p[6];   # Air resistance coefficient.     

    #--------------------------------------------------------------------------
    #x[0]: Longitudinal vehicle velocity. 
    #x[1]: Lateral vehicle velocity. 
    #x[2]: Yaw rate.
    #x[3]: Angle   
        #--------------------------------------------------------------------------
            #IMPORTANT!! Ange 0°/0 => car drives rigth !!
        #--------------------------------------------------------------------------
    #x[4]: X-Position
    #x[5]: Y-Position
    #-------------------------------------------------------------------------- 
    
    '''
    Calculate the update States
    '''

    
    #Controls
    #--------------------------------------------------------------------------
    # u[0] = s_FL(t)     Slip of Front Left tire [ratio].
    # u[1] = s_FR(t)     Slip of Front Right tire [ratio].
    # u[2] = s_RL(t)     Slip of Rear Left tire [ratio].
    # u[3] = s_RR(t)     Slip of Rear Right tire [ratio].
    # u[4] = delta(t)    Steering angle [rad].
    #--------------------------------------------------------------------------
    dx=np.zeros([6,1])

    dx[0] = x[1]*x[2]+1/m*(Cx*(u[0]+u[1])*np.cos(u[4])
            -2*Cy*(u[4]-(x[1]+a*x[2])/x[0])*np.sin(u[4]) 
            +Cx*(u[2]+u[3])-CA*np.power(x[0],2))
            
 

    dx[1] = -x[0]*x[2]+1/m*(Cx*(u[0]+u[1])*np.sin(u[4]) 
            +2*Cy*(u[4]-(x[1]+a*x[2])/x[0])*np.cos(u[4]) 
            +2*Cy*(b*x[2]-x[1])/x[0])

    dx[2] = 1/(np.power(((a+b)/2),2)*m)*(a*(Cx*(u[0]+u[1])*np.sin(u[4])
            +2*Cy*(u[4]-(x[1]+a*x[2])/x[0])*np.cos(u[4]))
            -2*b*Cy*(b*x[2]-x[1])/x[0])

    dx[3] = x[2]

    v_inertial=np.zeros([2,1])
    v_inertial[0]=x[0]
    v_inertial[1]=x[1]
    
    #Coordinate Transformation Matrix
    phi=x[3]
    S=np.squeeze(np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]))
    
    #print(S.shape,v_inertial.shape)
    v_global = np.matmul(S,v_inertial)
    
    #print(v_global)
    dx[4]=v_global[0]
    dx[5]=v_global[1]
    return dx


#--------------------------------------------------------------------------
    #How to use
#--------------------------------------------------------------------------
'''
c=getConfig()

x=[
    100,
    0,
    0,
    np.pi/2,
    0,
    0
]
x=np.reshape(x,[-1,1])

X=np.array(x)
X=np.reshape(X,[-1,1])

u=[
    0,
    0,
    0,
    0,
    0

]
for i in range(10):
    dx=get_dot_states_two_track_model(x,c[0],u)
    x=get_next_states_two_track_model(x,dx,1)
    X=np.hstack((X,x))
    

print(X)

plt.figure()
plt.plot(X[4,:],X[5,:])
'''

        