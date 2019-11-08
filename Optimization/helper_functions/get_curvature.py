import numpy as np


def get_curvature(racetrack: np.ndarray):
    racetrack_delta=np.diff(racetrack,axis=0)
    #print("Delta: ",racetrack_delta)
    
    kappa=np.zeros([racetrack_delta.shape[0],1])

    for i,tup in enumerate(racetrack_delta):
        #print("\n",i," delta_XY: ", tup[0],tup[1],"\t delta_phi: ", tup[2])
        kappa[i]=tup[2]/(tup[0]**2+tup[1]**2)**(0.5)

    return kappa






