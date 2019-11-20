import numpy as np 
import pandas as pd

df=pd.read_csv("/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/MaschineLearning/vel_planner/berlin_kappa.csv")
track=df.to_numpy()


length=len(track)

startindices=range(0,length-100,10)
print(startindices)
TrainingArray=np.zeros([100,1])

for i in startindices:
    t_hundred=np.reshape(np.array(track[i:i+100]),(-1,1))
    TrainingArray=np.hstack((TrainingArray,t_hundred))   

print(TrainingArray.shape)
np.savetxt("Track_in_baches.csv",TrainingArray,delimiter=",")




