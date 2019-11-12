
from BA_Optimization_ML.Optimization.helper_functions import splitcurve as sp
from BA_Optimization_ML.Optimization.helper_functions import Polynomial as pl
from BA_Optimization_ML.Optimization.helper_functions import Curve

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import math
import os
import pandas


#------------------------------
#Initial Values
polynoialDegree=10  #for the regression Problem so approximate a analytic curve
steps_for_equal_distent=100 
save_csv=0 #0 = False & 1 = True
#------------------------------






drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
listy=[]
listx=[]


# mouse callback function
def __draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    global listx, listy
    
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)
                listx.append(x)
                listy.append(-y)            
                
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

def __createImg():
    global listy,listx,img,pic
    listy=[]
    listx=[]

    img = np.zeros((512,512,3), np.uint8)
    pic=cv2.namedWindow('image')
    cv2.setMouseCallback('image',__draw_circle)
    start=True
    while(start or drawing):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break

        if drawing:
            start=False

def __prepCoor(pos):
    arrayPos=np.array(pos)
    arrayPos=arrayPos-arrayPos[0]
    return arrayPos.reshape(-1,1)

def __getcoef(lis,deg=polynoialDegree):
    sol=[]
    axes=["x","y"]
    for count,lis in enumerate(lis):
        X=np.linspace(0,1,len(lis)).reshape(-1,1)
        y=__prepCoor(lis)
        polmodel = make_pipeline(PolynomialFeatures(deg),LinearRegression(normalize=True))
        polmodel.fit(X, y)
        
        score=polmodel.score(X,y)
        print('Accuracy List',axes[count]," :",score)
        sol.append(np.squeeze(polmodel.steps[1][1].coef_))
    return sol


#-------------------------------------------------
#TO BE DONE !!!!
def get_as_curve():
    global listy,listx,img,pic
    __createImg()
    coef=__getcoef([listx,listy])
    print("\nCoefficient X:\n",coef[0].T,"\n--------\nCoefficient Y:\n",coef[1])
    #fx=pl.Polynomial(list(coef[0]))
    #fy=pl.Polynomial(list(coef[1]))
    #fx_dot=pl.Polynomial.dot(list(coef[0]))
    #fy_dot=pl.Polynomial.dot(list(coef[1]))


    #return curve(fx,fy,fx_dot,fy_dot)
#-------------------------------------------------


#TO BE DONE
#-------------------------------------------------
def add_angel(racetrack_without_phi):
    pass
#-------------------------------------------------

# MAIN FUNCTION OF THE FILE
# CREATE A TRACK FROM DRAWING +
# EXPORTING AS CSV 
#-------------------------------------------------
def get_as_csv(steps=steps_for_equal_distent,save=save_csv,name='racetrack_data'):
    global listy,listx,img,pic
    __createImg()
    coef=__getcoef([listx,listy])
    print("\nCoefficient X:\n",coef[0].T,"\n--------\nCoefficient Y:\n",coef[1])
    fx=pl.Polynomial(list(coef[0]))
    fy=pl.Polynomial(list(coef[1]))

    #10 only for testing purposes to get a good quality equal distant racetrack array
    #might take a little longer that needet

    t=np.linspace(0,1,steps*10)
    fx_digital=fx(t)
    fy_digital=fy(t)
    #----------------------------------
    #Space for phi the direction angle
    #----------------------------------
    racetrack=np.vstack((np.array(fx_digital),np.array(fy_digital)))
    racetrack=racetrack.T
    #fig,axs=plt.subplots(1,2)
    #axs[0].plot(racetrack[:,0],racetrack[:,1])
    racetrack_equal_dis=sp.interpol_equal(racetrack,steps)
    #axs[1].plot(racetrack_equal_dis[:,0],racetrack_equal_dis[:,1])
    
    if save:
        filename=name
        df = pd.DataFrame(data=racetrack_equal_dis,columns=['X', 'Y','phi'])
        df.to_csv('BA_Optimization_ML/Optimization/imput_tracks/'+filename+'.csv')
    return racetrack_equal_dis

#get_as_csv()
#cv2.destroyAllWindows