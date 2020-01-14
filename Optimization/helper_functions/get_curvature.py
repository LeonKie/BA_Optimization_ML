import numpy as np
import scipy.spatial.distance as dis

def get_curvature(racetrack: np.ndarray):
    #racetrack_delta=np.diff(racetrack,axis=0)
    #print("Delta: ",racetrack_delta)
    
    kappa=np.zeros([racetrack.shape[0],1])

    velocity_vectors=np.diff(racetrack,axis=0)
    phi=getangle(velocity_vectors)

    phi=np.reshape(phi,[-1,1])
    track=zip(list(racetrack[:,0]),list(racetrack[:,1]),list(phi))

    print("This is the ZIP Track:\n ", track)
    
    for i,tup in enumerate(track):
        print(tup)
        kappa[i]=tup[2]/(tup[0]**2+tup[1]**2)**(0.5)
    
    return kappa

def getangle(velocity_vectors):
    N=np.size(velocity_vectors,0)
    phi=np.zeros(N+1)
    for i,v in enumerate(velocity_vectors):
        phi[i]=(np.angle(v[0]+v[1]*1j)-np.pi/2)

    phi[N]=phi[N-1]
    return phi




