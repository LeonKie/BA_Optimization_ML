
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


plt.rcParams["mathtext.fontset"] = 'stix' # math fonts
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid

def plot_racetrack(racetrack: np.ndarray, ax=None, **kwargs):
    ax = ax or plt.gca()
 
    ax.plot(racetrack[0][0],racetrack[0][1],"g*",markersize=10)    
    ax.plot(racetrack[-1][0],racetrack[-1][1],"ro",markersize=10)
    ax.plot(racetrack[:,0],racetrack[:,1],"-",markersize=3)
    ax.axis('equal')
    plot=ax.legend(["Start","End","Curve"])
    

    return plot

def plot_racetrack_form_csv(name,ax=None, **kwargs):
    path='BA_Optimization_ML/Optimization/imput_tracks/'
    try:
        df=pd.read_csv(path+name)
    except:
        IOError("Wrong Filename")

    racetrack=df.to_numpy()
    return plot_racetrack(racetrack,ax, **kwargs)

def plot_velocity(racetrack: np.ndarray, ax=None, **kwargs):
    ax = ax or plt.gca()
 

    u_Vel=np.cos(racetrack[:,2]+math.pi/2)
    v_Vel=np.sin(racetrack[:,2]+math.pi/2)
    ax.plot(racetrack[0][0],racetrack[0][1],"g*",markersize=10)    
    ax.plot(racetrack[-1][0],racetrack[-1][1],"ro",markersize=10)
    steps=1
    ax.quiver(list(racetrack[:,0][::steps]),list(racetrack[:,1][::steps]),list(u_Vel[::steps]),list(v_Vel[::steps]),color='b',scale=100)
    plot=ax.axis('equal')
    

    return plot


def plot_curve(curve):
    steps = 100 #needs to change
    time=np.linspace(0,1,steps)
    plt.clf
    pxy=curve.createList()
    
    fig, axs = plt.subplots(1,2)
    axs[0].plot(curve.fx(time[0]),curve.fy(time[0]),"g*",markersize=10)
    axs[0].plot(curve.fx(time[-1]),curve.fy(time[-1]),"ro",markersize=10)
    axs[0].plot(curve.fx(time),curve.fy(time),marker="o",markersize=3)
    axs[0].set_title("Kurve without equal Spacing")
    axs[0].legend(["Start","End","Curve"])

    X=([x[0] for x in pxy])
    Y=([y[1] for y in pxy])

    axs[1].plot(X[0],Y[0],"g*",markersize=10)
    axs[1].plot(X[-1],Y[-1],"ro",markersize=10)
    axs[1].plot(X,Y,marker="o",markersize=3)
    axs[1].set_title("Kurve with equal Spacing")
    axs[1].legend(["Start","End","Curve"])
    plt.show()


def drawTrack(raceTrack):
    pass
