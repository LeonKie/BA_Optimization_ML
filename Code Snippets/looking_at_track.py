import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

filePath= "/Users/leonkiesgen/Documents/Python/mod_local_trajectory/inputs/traj_ltpl_cl/traj_ltpl_cl_monteblanco.csv"


f=open("/Users/leonkiesgen/Documents/Python/track_dataset_Modena.pkl", "rb")
graph_dic=pickle.load(f)
f.close()


#for lap in graph_dic:
#    plt.plot(lap[0],lap[1])
#plt.show()


df = pd.read_csv(filePath,sep= ";")

x=df[" x_ref_m"].to_numpy()
y=df[" y_ref_m"].to_numpy()
w_r=df[" width_right_m"].to_numpy()
w_l=df[" width_left_m"].to_numpy()
n_x=df[" x_normvec_m"].to_numpy()
n_y=df[" y_normvec_m"].to_numpy()

Norm_M = np.stack((n_y,n_x),axis=0)

M_turn_R= np.array([[0, 1],[-1, 0]])
M_turn_L= np.array([[0, -1],[1, 0]])


R_M=np.matmul(M_turn_R,Norm_M)
L_M=np.matmul(M_turn_L,Norm_M)


x_rLine = x + R_M[0,:] * w_r
y_rLine =y + R_M[1,:] * w_r

x_lLine = x + L_M[1,:]*w_l
y_lLine =y + L_M[0,:]*w_l

#plt.plot(x_rLine,y_rLine,"black")
#plt.plot(x_lLine,y_lLine,"black")
plt.plot(x,y)
plt.xlabel("east in $m$")
plt.ylabel("north in $m$")
plt.show()
#print(len(Dataset))


