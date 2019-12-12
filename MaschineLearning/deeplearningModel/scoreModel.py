import numpy as np
import torch
import random
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch import nn

def __initGPU():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")  



def __Scoring(evalmodel,example,Score,vNorm):
    
    
    if Score=='MAE':
        criterion=nn.L1Loss()
    elif Score=="MSE":
        criterion=nn.MSELoss()
    elif Score=="R2":
        PRED=[]
        REAL=[]
        with torch.no_grad():
            for (inp,real) in tqdm(example,leave=True):


                pred=evalmodel(inp)

                #Normalisation is missing

                PRED.append(pred.tolist())
                REAL.append(real.tolist())
        return r2_score(PRED,REAL,multioutput='variance_weighted')
    else: 
        print("Loss Function does not exist")
        return
    
    cumloss=0
    
    with torch.no_grad():
        for (inp,real) in tqdm(example,leave=True):
        
            pred=evalmodel(inp)

            pred=pred*vNorm
            real=real*vNorm

            cumloss += criterion(pred,real)
        
    SCORE=cumloss/len(example)
    
    return SCORE.item()



def eval(evalmodel,example,vNorm):
    __initGPU()

    R2Score=__Scoring(evalmodel,example,"R2",vNorm)
    print(" R2: " ,R2Score)

    MAEScore=__Scoring(evalmodel,example,"MAE",vNorm)
    print(" MAE: " ,MAEScore)
    
    MSEScore=__Scoring(evalmodel,example,"MSE",vNorm)
    print(" MSE: " ,MSEScore)

    return {"R2": R2Score,"MAE": MAEScore,"MSE": MSEScore}
        
        
        
