import numpy as np 
import matplotlib.pyplot as plt 
import json_ImExport
import pandas as pd
from matplotlib.widgets import Slider
import mplcursors

#df=pd.read_csv("/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/MaschineLearning/vel_planner/berlin_kappa_3.csv",delimiter=";")
#track=df.to_numpy()

#Import Data from JSON File
data=json_ImExport.loadData("Kappa_variation_shift10.json")

fig,ax= plt.subplots(2,1)



def distence(track):
    return np.sqrt(np.power(track[:,0],2)+np.power(track[:,1],2))

def Kappa2XY(kappa,distence):
    X=[0]
    Y=[0]
    delta_s=distence
    cumsumkappa=np.cumsum(kappa)
    for k in cumsumkappa:
        x = delta_s * np.cos(delta_s*k) + X[-1]
        y = delta_s * np.sin(delta_s*k) + Y[-1]
        X.append(x)
        Y.append(y)

    return X,Y

def updataLine(sel):
    global ax,iteration,X
    #print(sel)
    xy=sel.target[0]
    if xy in X:
        iteration=X.index(xy)

    sel.annotation.set_text("V: "+ str(velocity[iteration]))
    update()
    ax[0].plot([iteration,iteration],vmaxmin,"r")

k=1

kappa=data["list"][k]["Kappa"]
velocity=data["list"][k]["V_op"]
delta_s=data["list"][k]["delta_s"]

axfreq = plt.axes([0.1, 0.05, 0.80, 0.03])
strack = Slider(axfreq, 'TrackNr', 1, len(data["list"]), valinit=1, valstep=1)

p1,=ax[0].plot(range(len(velocity)),velocity)
ax[0].set_title("V_op")
ax[0].set_ylabel("Velocity")

X,Y=Kappa2XY(kappa,3) 
p2=ax[1].scatter(X,Y,s=1)
ax[1].set_aspect("equal")
ax[1].set_title("Track")
cursor = mplcursors.cursor(p2,hover=True)
cursor.connect("add",updataLine)

def updateSlider(val):
    global velocity,kappa,vmaxmin,ax,p1,cursor,p2,X,Y
    nr = int(strack.val)
    velocity=data["list"][nr]["V_op"]
    vmaxmin=[min(velocity),max(velocity)]
    ax[0].clear()
    p1,=ax[0].plot(range(len(velocity)),velocity)
    ax[0].set_title("V_op")
    ax[0].set_ylabel("Velocity")

    
    kappa=data["list"][nr]["Kappa"]
    X,Y=Kappa2XY(kappa,3) 
    ax[1].clear()
    p2=ax[1].scatter(X,Y,s=1)
    ax[1].set_aspect("equal")
    ax[1].set_title("Track")
    fig.canvas.draw_idle()
    cursor = mplcursors.cursor(p2,hover=True)
    cursor.connect("add",updataLine)


strack.on_changed(updateSlider)

def update():
    global ax

    ax[0].clear()
    ax[0].plot(range(len(velocity)),velocity)
    ax[0].set_title("V_op")
    ax[0].set_ylabel("Velocity")





iteration=1
velocity=data["list"][k]["V_op"]
vmaxmin=[min(velocity),max(velocity)]

#-------------------------------------
#- wy to i have to multipy with 3 ? --
#-------------------------------------

plt.show()