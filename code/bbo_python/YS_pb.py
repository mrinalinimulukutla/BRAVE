# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:51:45 2024

@author: Danial
"""
import numpy as np
import pandas as pd
from HT_FCC_YS import cal_taus_HT



def YS_pb(data_points):
    
    elements= ['Al','V','Cr','Mn','Fe','Co','Ni','Cu']
    
    data_points = data_points*100

    T = 25+273
    taus=cal_taus_HT(list=elements,concs=data_points,T=T)
    V = np.array(taus)
    return V



