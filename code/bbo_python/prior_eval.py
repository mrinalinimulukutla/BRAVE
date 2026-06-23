# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:51:15 2024

@author: Danial
"""

import numpy as np
import pandas as pd
from priors import UTStoYS_prior, EUTS_prior, YS_prior

x=pd.DataFrame(pd.read_csv('x_test.csv', header=None)).to_numpy()

Y1 = UTStoYS_prior(x)
Y2 = EUTS_prior(x)
Y3 = YS_prior(x)


pd.DataFrame(Y3.reshape(-1,1)).to_csv("YS_prior.csv", header=None, index=None)
pd.DataFrame(Y2.reshape(-1,1)).to_csv("EUTS_prior.csv", header=None, index=None)
pd.DataFrame(Y1.reshape(-1,1)).to_csv("UTStoYS_prior.csv", header=None, index=None)

