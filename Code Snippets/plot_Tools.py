import numpy as np
import matplotlib.pyplot as plt

def plotKappa(kappa,distence,color=None):
    
    X=[0]
    Y=[0]
    delta_s=distence
    #print("Mean:", distence_mean)
    cumsumkappa=np.cumsum(kappa)
    #print(cumsumkappa)
    for k in cumsumkappa:
        x = delta_s * np.cos(delta_s*k) + X[-1]
        y = delta_s * np.sin(delta_s*k) + Y[-1]
        X.append(x)
        Y.append(y)

    if color is not None:
        plt.plot(X,Y,"o-",markersize=1,color=color)
    else:
        plt.plot(X,Y,"o-",markersize=1)
    plt.axis("equal")