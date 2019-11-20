import numpy as np 
import matplotlib.pyplot as plt 
import json_ImExport
import pandas as pd

#df=pd.read_csv("/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/MaschineLearning/vel_planner/berlin_kappa_3.csv",delimiter=";")
#track=df.to_numpy()

#Import Data from JSON File
data=json_ImExport.loadData("Kappa_variation_shift10.json")



def distence(track):
    return np.sqrt(np.power(track[:,0],2)+np.power(track[:,1],2))

def plotKappa(kappa,distence,pltset="o-"):
    X=[0]
    Y=[0]
    delta_s=distence
    cumsumkappa=np.cumsum(kappa)
    for k in cumsumkappa:
        x = delta_s * np.cos(delta_s*k) + X[-1]
        y = delta_s * np.sin(delta_s*k) + Y[-1]
        X.append(x)
        Y.append(y)
    plt.plot(X,Y,pltset,markersize=1)
    plt.axis("equal")
    plt.title("Track")
    

k=20
kappa=data["list"][k]["Kappa"]
velocity=data["list"][k]["V_op"]
delta_s=data["list"][k]["delta_s"]

plt.figure()
plt.subplot(211)
plt.plot(range(len(velocity)),velocity,"-")
plt.title("V_op")
plt.ylabel("Velocity")

plt.subplot(212)

#-------------------------------------
#- wy to i have to multipy with 3 ? --
#-------------------------------------
plotKappa(kappa,3)



#vectors=np.diff(track[:,1:3],axis=0)
#distence_abs=np.mean(distence(vectors))
#plotKappa(track[:,3],distence_abs,"or")
plt.show()