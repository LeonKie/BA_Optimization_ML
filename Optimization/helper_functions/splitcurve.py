import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import math 

def interpol_equal(race_track_tuple,N) -> "Race Track":
    '''
	Created by Leon Kiesgen
	Date: 29.10.19

	This function interpolates a already existing Curve, so that all the points in the Trajectory
	are equally spaced. This is necessary in the Optimization problem to eliminate one parameter
	either time or length of a Segment.


    The angel of a Trajectorie is the definde as angle between the positv y-Axis and the tangent verctor
    e.g. the vector 0+1j has the angle 0 deg
    and  the vector 1+0j hat the angle -90deg

	'''
    
    print("equallity befor: ", check_equal(race_track_tuple))

    # Create Velocitys
    velocity_vectors=np.diff(race_track_tuple,axis=0)

    #Calculatin the total lenght of the Track segment
    #Using the scipy libary
    distence_abs=np.diag(dis.squareform(dis.pdist(race_track_tuple)),1)
    distence=sum(distence_abs)


    #find distences where the points should be to get an equal distance
    
    seg_lenght=distence/N
    seg_lenght_array=[i*seg_lenght for i in range(1,N+1)] #how it sould be
    seg_length_array_real=np.cumsum(distence_abs)
    seg_length_array_real=np.insert(seg_length_array_real,0,0)

    #Interpolate between the points
		#decides in in between with old point the new points
    bins=np.digitize(seg_lenght_array,seg_length_array_real)
    bins=bins[:-1]
    race_track_out=np.zeros([N+1,2])
    race_track_out[0,:]=race_track_tuple[0]
    #print("\n ---------- \n", race_track_out)


    #For each equal distant point create a new point on the  interpolated path between the corresponding points
    
    for j,binn in enumerate(bins):

        # rel_distence is the Procentage value of the a Line between point A and B
        rel_distence=(seg_lenght_array[j]-seg_length_array_real[binn-1])/distence_abs[binn-1]
        newpoint=race_track_tuple[binn-1]+velocity_vectors[binn-1]*rel_distence
        print("J: " , j , "NewP: ", newpoint)
        race_track_out[j+1,:]=newpoint

    race_track_out[-1,:]=race_track_tuple[-1]

    print("\n ---------- \n", race_track_out)


    print("equallity after: ", check_equal(race_track_out))
    phi=getangle(race_track_out)
    #phi=np.array([np.append(phi,phi[-1])])
    phi=np.reshape(phi,[N+1,1])
    print("Phi: ",phi, "SIZE:" , (phi.shape))
    #print("Race:", race_track_out)
    #race_track_out=np.hstack(np.array(race_track_out),phi)
    #print(race_track_out)
    
    return np.concatenate((race_track_out,phi),axis=1)

def getangle(velocity_vectors):
    phi=np.zeros(np.size(velocity_vectors,0))
    for i,v in enumerate(velocity_vectors):
        phi[i]=(np.angle(v[0]+v[1]*1j)-math.pi/2)
    return phi

def check_equal(race_track_tuple) -> "Race Track":
    distence_abs=np.diag(dis.squareform(dis.pdist(race_track_tuple)),1)
    return np.std(distence_abs)


#velocity_vectors=np.diff([[1,2],[2,4],[3,3],[2,5]],axis=0)
#print(velocity_vectors)
#print (getangle(velocity_vectors))

out=interpol_equal([[1,2],[2,4],[3,3],[2,5]],100)

print(out)
#X=([x[0] for x in out])
#Y=([y[1] for y in out])

#plt.plot(X,Y)


if __name__ == "__main__":
	pass

