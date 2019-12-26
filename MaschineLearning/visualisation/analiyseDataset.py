import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

import sys
import os
cwd = os.getcwd()
sys.path.insert(1, cwd+'/BA_Optimization_ML/MaschineLearning/visualisation')
sys.path.insert(1, cwd+ '/BA_Optimization_ML/MaschineLearning/deeplearningModel')
import LogDataImport
import random
import dataprocessing
from tqdm import tqdm
import time
from sklearn.cluster import AgglomerativeClustering



'''
with tqdm(total=4) as pbar:
    #Json
    #Jsondata=json_ImExport.loadData("/Users/leonkiesgen/Documents/Python/Kappa_variation_vmax100.json")
    #Log File (Monteblanco)
    pbar.update(1)
    

    Logdata_MB=LogDataImport.get_data("/Users/leonkiesgen/Documents/Python/mod_local_trajectory/logs/ltpl/2019_12_01/15_00_59_data.csv")

    #Log File (Berlin)
    pbar.update(1)
    ValLogdata_B=LogDataImport.get_data("/Users/leonkiesgen/Documents/Python/mod_local_trajectory/logs/ltpl/2019_12_05/12_19_36_data.csv")

    #Log File (Modena)
    pbar.update(1)
    ValLogdata_M=LogDataImport.get_data("/Users/leonkiesgen/Documents/Python/mod_local_trajectory/logs/ltpl/2019_12_05/15_41_27_data.csv")

    #Log File (zalazone)
    pbar.update(1)
    ValLogdata_Z=LogDataImport.get_data("/Users/leonkiesgen/Documents/Python/mod_local_trajectory/logs/ltpl/2019_12_05/15_51_25_data.csv")
'''

def orderDataset(datasetList,lenght):

    dataset=[]
    for data in datasetList:
        dataset.extend(data)

    data=[] 
    vlist=[]
    for (inp,label) in dataset:
        vini=label[0:lenght:]

        for i,v in enumerate(vini):
            out=[]
            
            #out.append(v)
            out.extend(inp[i*lenght:(i+1)*lenght])
            
            while len(out)<10:
                out.append(out[-1])
            #print(len(out))
            vlist.append(v)
            data.append(out)
    return (data,vlist)


def hierarchical(dataset):
    #dataset=orderDataset(dataset,10)
    linked = linkage(dataset, 'single')

    labelList = range(1, len(dataset)+1)

    dendrogram(linked,p=150,truncate_mode="lastp",
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
    

def cluster(dataset,k):
    dataset,vlist=orderDataset(dataset,10)
    dataset=np.array(dataset)
    vlist=np.array(vlist)

    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    cluster.fit_predict(dataset)

    label=cluster.labels_
    label=label.tolist()

    NrClass=[]
    for i in range(k):
        
        NrClass.append(label.count(i))


    return NrClass,cluster,dataset,vlist

#dataset=orderDataset([ValLogdata_B],10)
#dataset=np.array(dataset)
#print(cluster(dataset,10))