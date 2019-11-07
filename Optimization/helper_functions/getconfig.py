import configparser
import os


def getConfig():
    config = configparser.ConfigParser()
    config.read('BA_Optimization_ML/Optimization/setup.ini')
    return config



    