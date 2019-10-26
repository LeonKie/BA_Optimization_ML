#!/usr/bin/env python
# coding: utf-8

# In[69]:



import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

drawing = False # true if mouse is pressed
mode = False # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
listy=[]
listx=[]

# mouse callback function
def draw_circle(event,x,y,flags,param):
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
            


# In[70]:


def createImg():
    global listy,listx,img,pic
    listy=[]
    listx=[]

    img = np.zeros((512,512,3), np.uint8)
    pic=cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    tempdrawing=drawing
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


# In[71]:


#cv2.destroyAllWindows


# In[72]:


def prepCoor(pos):
    arrayPos=np.array(pos)
    arrayPos=arrayPos-arrayPos[0]
    return arrayPos.reshape(-1,1)


# # Regression Model

# In[73]:


def getcoef(lis,deg=5):
    sol=[]
    axes=["x","y"]
    for count,lis in enumerate(lis):
        X=np.linspace(0,1,len(lis)).reshape(-1,1)
        y=prepCoor(lis)
        polmodel = make_pipeline(PolynomialFeatures(deg),LinearRegression(normalize=True))
        polmodel.fit(X, y)
        
        score=polmodel.score(X,y)
        print('Accuracy List',axes[count]," :",score)
        sol.append(np.squeeze(polmodel.steps[1][1].coef_))
    return sol


# # Polymomoal Class
# 

# In[74]:


class Polynomial:
    
    def __init__(self, coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0 
        """
        self.coefficients =  coefficients # tuple is turned into a list
     
    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        return "Polynomial" + str(self.coefficients)
            
    def __call__(self, x):    
        res = 0
        for index, coeff in enumerate(self.coefficients):
            #print(index,coeff)
            res = res + coeff * x** index
        return res 


# # Creat & Plote Curve

# In[75]:



class curve: 
    def __init__ (self,fx,fy):
        self.fx
        self.fx=fx
        self.fy=fy
    def fx(t,self):
        return self.fx(t)
    def fy(t,self):
        return self.fy(t)
    '''
    def __repr__(self):
        steps=100;
        time=np.linspace(0,1,steps);
        plt.clf
        plt.plot(self.fx(time[0]),self.fy(time[0]),"g*",markersize=10)
        plt.plot(self.fx(time[-1]),self.fy(time[-1]),"ro",markersize=10)
        plt.plot(self.fx(time),self.fy(time),marker="o",markersize=3)
        plt.legend(["Start","End","Curve"])
        plt.title("Kurve")
        plt.show()
        return 
    '''
def set_Curve(coeff_X,coeff_Y):
    fx=Polynomial(coeff_X)
    fy=Polynomial(coeff_Y)
    return curve(fx,fy)


    
    
    
def plotcurve(curve):
    steps=100;
    time=np.linspace(0,1,steps);
    plt.clf
    plt.plot(curve.fx(time[0]),curve.fy(time[0]),"g*",markersize=10)
    plt.plot(curve.fx(time[-1]),curve.fy(time[-1]),"ro",markersize=10)
    plt.plot(curve.fx(time),curve.fy(time),marker="o",markersize=3)
    plt.legend(["Start","End","Curve"])
    plt.title("Kurve")
    plt.show()
    
    


# # Display Polynomial View



def drawTrack(showPlot=True,curve=None):
    if (curve is None):
        createImg()
        coef=getcoef([listx,listy],20)
        print("\nCoefficient X:\n",coef[0].T,"\n--------\nCoefficient Y:\n",coef[1])
        c=set_Curve(list(coef[0]),list(coef[1]))
        if showPlot:
            plotcurve(c)
    else:
        if showPlot:
            c=curve
            plotcurve(c)
    return c






