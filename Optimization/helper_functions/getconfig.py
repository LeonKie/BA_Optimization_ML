import configparser
import os


def getConfig():
    c = configparser.ConfigParser()
    c.read('BA_Optimization_ML/Optimization/setup.ini')
    Parameter=[]
    for sec in c.sections():  
        p=[]
        for i,para in enumerate(c.options(sec)):
            try:
                p.append(float(c[sec][para]))
            except:
                p.append(c[sec][para])
        Parameter.append(p)
    return (Parameter)


    