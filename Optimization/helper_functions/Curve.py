import numpy as np
import BA_Optimization_ML.Optimization


class curve:
    global step
    def __init__ (self,fx,fy,fx_dot=0,fy_dot=0):
        self.fx
        self.fx=fx
        self.fy=fy
        self.fx_dot=fx_dot
        self.fy_dot=fy_dot
        
        
    def fx(self,t):
        return self.fx(t)
    def fy(self,t):
        return self.fy(t)
    def fx_dot(self,t):
        return self.fx_dot(t)
    def fy_dot(self,t):
        return self.fy_dot(t)
    def pxy(self):
        return self.pxy
    
    
    def createList(self):
        global steps
        Fx= np.array([self.fx(i) for i in np.linspace(0,1,steps)])
        Fy= np.array([self.fy(i) for i in np.linspace(0,1,steps)])
        
        self.pxy=sp.split(steps,Fx,Fy)
        #pxy=pxy[:int(steps*0.8)]
        return self.pxy 
        
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