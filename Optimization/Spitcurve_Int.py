#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy.integrate import quad
import math


def slit(N,fx,fy,fx_dot,fy_dot):
    
    def integr(x):
        return math.sqrt(fx_dot(x)**2+fy_dot(x)**2)

   
    eval_n=np.linspace(0,1,N)
    print(eval_n)
    eval_n=np.delete(eval_n,0)
    print(eval_n)


    length_seg=	[quad(integr,0,b) for b in eval_n]
    print(length_seg)

    pxy=(fx[length_seg],fy[length_seg])
    return pxy


# In[2]:





# In[ ]:




