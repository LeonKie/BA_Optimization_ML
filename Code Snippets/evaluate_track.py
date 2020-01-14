import pickle
import matplotlib.pyplot as plt
import sys
from get_curvature import get_curvature
import numpy as np
import splitcurve
import trajectory_planning_helpers as tph

import os


#Importing Pickle File
f=open("/Users/leonkiesgen/Documents/Python/track_dataset.pkl", "rb")
graph_dic=pickle.load(f)
f.close()

dataset=[]

for num,graph in enumerate(graph_dic):
    
    #Converte path into kappa values
    x=graph[0]
    y=graph[1]
    x=np.array(x)
    x=np.reshape(x,[-1,1])
    y=np.array(y)
    y=np.reshape(y,[-1,1])
        #check if the segments have the same lenght



    track=np.hstack((x,y))
    #print(splitcurve.check_equal(track))

    #Define delta between each step
    delta=2

    track=splitcurve.interpol_equal(track,delta)

    print("Track",track.shape)
    total_points=track.shape[0]
    kappa=get_curvature(track)

    #extract section of 100 values from the kappa_list
        #but shift kappa values only by 50
    Nr_seg=int((total_points/100)*2)
    segments_kappa=[np.array(kappa[j*50:j*50+100]) for j in range(0,Nr_seg)]


    #Run the Opimizer on for every sevtion of the track
    #toppath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    #GGV_NORMAL_PATH=toppath + "/ltpl/testing_tools/ggv/ggv_db_test.txt"
    GGV_NORMAL_PATH="/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/Code Snippets/ggv_db_test.txt"

    ggv_normal = tph.import_ggv.import_ggv(ggv_import_path=GGV_NORMAL_PATH)

    vel_start_list=range(0,70,5)

    print(vel_start_list)


    
    for i,segment in enumerate(segments_kappa):
        #try a range of inital velocitys for every track_section
        vel_checker=1
        for vel_start in vel_start_list:
            
            kappa=segment.squeeze()
            dis=np.ones([len(kappa)-1,1])*2
            dis=dis.squeeze()
            #print(vel_start)
            vx = tph.calc_vel_profile.\
                    calc_vel_profile(ggv=ggv_normal,
                                    kappa=kappa,
                                    el_lengths=dis,
                                    v_start=vel_start,
                                    closed=False)
            
            if abs(vx[0]-vel_start)<0.1:
                #print(vx[0],vel_start)
                dataset.append((kappa,vx))
            else:
                if vel_checker>0:
                    #print(vx[0],vel_start)
                    dataset.append((kappa,vx))
                vel_checker-=1
                
        #print("Track Segment: ",i)

        #check if the tracks are drivable
    print("Track Number: ", num)
    #plt.plot(dataset[5][1])

    #export the dataset so that it can be used in the training dataset



#plt.plot(track[:,0],track[:,1])
#plt.show()