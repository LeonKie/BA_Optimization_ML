import numpy as np
import torch
import sys
sys.path.insert(1, '/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/MaschineLearning/visualisation')
import LogDataImport
import random
sys.path.insert(1,"/Users/leonkiesgen/Documents/Python/src/tqdm")
from tqdm.notebook import tqdm
import time
import configparser
import os


def setupconfig():
    config=configparser.ConfigParser()
    config['Normalization'] = {
                            'Velocity': 70.0,
                            'kappa': 0.05
    }
    cwd=os.getcwd()
    print(cwd)

    with open("/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/MaschineLearning/"+ "dataProcessig.ini","w") as configfile:
        config.write(configfile)

def getConfig(path):
    c = configparser.ConfigParser()
    c.read(path)
    Parameter=[]
    for sec in c.sections():  
        p=[]
        for para in c.options(sec):
            try:
                p.append(float(c[sec][para]))
            except:
                p.append(c[sec][para])
        Parameter.append(p)
    return (Parameter)

def getlowestlen(inpdata):
    #Inpdata is a list of data from different tracks
    data=[]
    for d in inpdata:
        data.extend(d)
    

class Normalize():

    def __init__(self,scale):

        self.max=scale
        print(scale)

    def __call__(self,x):
        
        x=np.array(x)
        #print(type(x),type(self.max))

        return np.divide(x,self.max)

    def normal(self,x):
        x=np.array(x)
        return x*self.max



def prepareData(indata,feature_size: int=1,length=np.inf):
    prepareddata=[]

    #Initialize Nomalizer
    c = configparser.ConfigParser()
    c.read("/Users/leonkiesgen/Documents/Python/BA_Optimization_ML/MaschineLearning/dataProcessig.ini")

    #kappaN=Normalize(scale=float(c["Normalization"]["kappa"]))
    velN=Normalize(scale=float(c["Normalization"]["Velocity"]))
    kappaN=Normalize(scale=0.001)
    data=[]
    for d in indata:
        data.extend(d)

    for item in data:

        

        #Input Data --------------------

        kappatmp=kappaN(item[0][:])
        #print(kappatmp)
        #return
        sample_size=kappatmp.shape[0]

        kappa=np.zeros([sample_size,feature_size])
        
        kappa[:,0]=kappatmp
        for i in range(1,feature_size):
            nextfeature=np.roll(kappatmp,-i)
            nextfeature[-i:]=nextfeature[-i-1]
            kappa[:,i]=nextfeature

        
        
        inpdata=torch.Tensor(kappa) #Kappa
        
        #Label Data -------------------
        label=velN(item[1][:])

        #Convert to Tensors -----------
        label=torch.Tensor(label)

        tupledata=(inpdata,label)
        
        #print(inpdata.shape,label.shape)
        
        prepareddata.append((inpdata,label))
       
    #print(lowestLen)
    return prepareddata



'''
Logdata_MB=LogDataImport.get_data("/Users/leonkiesgen/Documents/Python/mod_local_trajectory/logs/ltpl/2019_12_01/15_00_59_data.csv")
data=prepareData([Logdata_MB])
print(data[1])
'''