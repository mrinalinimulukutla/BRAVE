# -*- coding: utf-8 -*-
"""

@author: Danial Khatamsaz dkhatamsaz@gmail.com

HTMDEC second year design framework with 5 objectives

The order of elements
Al	V	Cr	Mn	Fe	Co	Ni	Cu
"""

import numpy as np
import pandas as pd
from pyDOE import *
from copy import deepcopy
from gpModel import gp_model
from sklearn.preprocessing import normalize
import timeit
from sklearn_extra.cluster import KMedoids
from scipy.stats import norm
from multiobjective import EHVI, Pareto_finder, HV_Calc
import timeit
import random
from multiprocessing import Pool
import multiprocessing
from joblib import Parallel, delayed
from reificationFusion import reification
from priors import YS_prior, EUTS_prior, UTStoYS_prior
from YS_pb import YS_pb



iteration=1
N_dim=8
N_obj=5
# N_test=10 # candidates in single fidelity or test in multi-fidelity
# N_training=16 # number of initial training data
# N_prior=2000
Batch_size=16 # use more than 1 for batch BO
N_GP=1000; # if batch BO
# goal = np.ones([1,N_obj]) ## maximizing all objectives
# ref=np.zeros([1,N_obj]) ## reference point
goal = np.array([[1,1,1,1,0]])
ref = np.array([[0,0,0,0,8]])

# normalizing constants to ensure all objectives are in the same order of magnitude
nz1=100
nz2=1
nz3=0.15
nz4=0.5
nz5=1

normal = [nz1,nz2,nz3,nz4,nz5]

def sigmoid(x):
    return 1 / (1 + np.exp(-2.2*(x-9)))


## load length-scales and design space
lhp=pd.DataFrame(pd.read_csv('lhp.csv', header=None)).to_numpy()

feasibles=pd.DataFrame(pd.read_csv('feasibles.csv', header=None)).to_numpy()
infeasibles=pd.DataFrame(pd.read_csv('infeasibles.csv', header=None)).to_numpy()
p_infeasibles=pd.DataFrame(pd.read_csv('probs.csv', header=None)).to_numpy()

all_probs = np.concatenate((np.ones([feasibles.shape[0]]),p_infeasibles.reshape(-1)))
all_space = np.concatenate((feasibles,infeasibles),axis=0)


## This is the file that contains 16*iter number of compositions that are tested
## This may be different than each source input as they may have less points available
tested_alloys=pd.DataFrame(pd.read_csv('tested_alloys.csv', header=None)).to_numpy()

## indices of samples lacking at least 1 objective value
## These should be excluded from ground truth data before Pareto front and hypervolume calculations
## If something is violating the constraint, do NOT add it here
o1_incomplete_alloys =[]
o1_is1_incomplete_alloys =[]
o2_incomplete_alloys =[]
o3_incomplete_alloys = []
o4_incomplete_alloys = []
o5_incomplete_alloys = []

## samples that violated the phase constraint
#BBB05 BBB06 BBB09 BBB11 BBB13
phase_violators = [23,24,27,29,31]
brittle = [0,3,8,15,28,34]

violators = phase_violators + brittle

# BBB: 02 03 04 08 14 15 are filtered out by TCHEA7 but not TCHEA6. 
#These are actually feasible samples not in the feasible space 

## Ensure already tested samples are excluded from the search space
zz=[]
for ii in range(tested_alloys.shape[0]):
    ind=np.where((tested_alloys[ii] == all_space).all(1))[0]
    if len(ind.tolist())>0:
        zz.append(ind.tolist()[0])

space = np.delete(all_space,zz,0)
probs = np.delete(all_probs,zz,0)


## Load data here ######## (inputs, outputs, noises)

# first objective data: Yield strength
o1_GT_x = np.delete(tested_alloys,violators+o1_incomplete_alloys,0)
o1_is1_x = np.delete(tested_alloys,violators+o1_is1_incomplete_alloys,0)

o1_tested_y=pd.DataFrame(pd.read_csv('o1_GT_y.csv', header=None)).to_numpy()/nz1
o1_is1_y=pd.DataFrame(pd.read_csv('o1_is1_y.csv', header=None)).to_numpy()/nz1
o1_GT_y=np.delete(o1_tested_y,violators+o1_incomplete_alloys,0)
o1_is1_y=np.delete(o1_is1_y,violators+o1_is1_incomplete_alloys,0)

o1_GT_sd=pd.DataFrame(pd.read_csv('o1_GT_sd.csv', header=None)).to_numpy()/nz1
o1_is1_sd=pd.DataFrame(pd.read_csv('o1_is1_sd.csv', header=None)).to_numpy()/nz1
o1_GT_sd=np.delete(o1_GT_sd,violators+o1_incomplete_alloys,0)
o1_is1_sd=np.delete(o1_is1_sd,violators+o1_is1_incomplete_alloys,0)



# second objective data: UTS/YS
o2_GT_x = np.delete(tested_alloys,violators+o2_incomplete_alloys,0)

o2_tested_y=pd.DataFrame(pd.read_csv('o2_GT_y.csv', header=None)).to_numpy()/nz2
o2_GT_y=np.delete(o2_tested_y,violators+o2_incomplete_alloys,0)

o2_GT_sd=pd.DataFrame(pd.read_csv('o2_GT_sd.csv', header=None)).to_numpy()/nz2
o2_GT_sd=np.delete(o2_GT_sd,violators+o2_incomplete_alloys,0)


# third objective data: Euts
o3_GT_x = np.delete(tested_alloys,violators+o3_incomplete_alloys,0)

o3_tested_y=pd.DataFrame(pd.read_csv('o3_GT_y.csv', header=None)).to_numpy()/nz3
o3_GT_y=np.delete(o3_tested_y,violators+o3_incomplete_alloys,0)

o3_GT_sd=pd.DataFrame(pd.read_csv('o3_GT_sd.csv', header=None)).to_numpy()/nz3
o3_GT_sd=np.delete(o3_GT_sd,violators+o3_incomplete_alloys,0)


# fourth objective data: hdyn/hqs
o4_GT_x = np.delete(tested_alloys,violators+o4_incomplete_alloys,0)

o4_tested_y=pd.DataFrame(pd.read_csv('o4_GT_y.csv', header=None)).to_numpy()/nz4
o4_GT_y=np.delete(o4_tested_y,violators+o4_incomplete_alloys,0)

o4_GT_sd=pd.DataFrame(pd.read_csv('o4_GT_sd.csv', header=None)).to_numpy()/nz4
o4_GT_sd=np.delete(o4_GT_sd,violators+o4_incomplete_alloys,0)


# fifth objective data: peneteration depth
o5_GT_x = np.delete(tested_alloys,violators+o5_incomplete_alloys,0)

o5_tested_y=pd.DataFrame(pd.read_csv('o5_GT_y.csv', header=None)).to_numpy()/nz5
o5_GT_y=np.delete(o5_tested_y,violators+o5_incomplete_alloys,0)


o5_GT_sd=pd.DataFrame(pd.read_csv('o5_GT_sd.csv', header=None)).to_numpy()/nz5
o5_GT_sd=np.delete(o5_GT_sd,violators+o5_incomplete_alloys,0)


##########################


#####load prior data here if any ######


#### if prior models are different:

## which models have prior?
prior_existence = [[True,True],[True],[True],[False],[False]]

## prior models if any (put GP.predict_mean as a function if prior is a GP)
priors_models = [[YS_pb,YS_prior],[UTStoYS_prior],[EUTS_prior],[[]],[[]]]

## query training data from priors and keep in this prior list
priors_values = [[[],[]],[[]],[[]],[[]],[[]]]


## if input to all models
inputs = [[o1_GT_x,o1_is1_x],[o2_GT_x],[o3_GT_x],[o4_GT_x],[o5_GT_x]]

for j in range(len(prior_existence)):
    for jj in range(len(prior_existence[j])):
        if prior_existence[j][jj]:
            priors_values[j][jj]=priors_models[j][jj](inputs[j][jj])/normal[j]
        else:
            priors_values[j][jj]=np.zeros([inputs[j][jj].shape[0]])
            

incomplete_alloys_list = o1_incomplete_alloys+o2_incomplete_alloys+o3_incomplete_alloys+o4_incomplete_alloys+o5_incomplete_alloys+violators
incomplete_alloys = list(set(incomplete_alloys_list))
y1_temp = np.delete(o1_tested_y,incomplete_alloys,0)
y2_temp = np.delete(o2_tested_y,incomplete_alloys,0)
y3_temp = np.delete(o3_tested_y,incomplete_alloys,0)
y4_temp = np.delete(o4_tested_y,incomplete_alloys,0)
y5_temp = np.delete(o5_tested_y,incomplete_alloys,0)
y=np.concatenate((y1_temp,y2_temp,y3_temp,y4_temp,y5_temp),axis=1)
train_y=y

y_pareto_curr,index=Pareto_finder(train_y,goal)
hv_curr = (HV_Calc(goal,ref,y_pareto_curr)).reshape(1,1)


candidates=[]
improvements=[]
indices=[]

x_test=space
p_test=probs
N_test=x_test.shape[0]


## load test priors here or query prior models

YS_pb_prior_tests=pd.DataFrame(pd.read_csv('YS_pb_prior.csv', header=None)).to_numpy().reshape(-1)/nz1
YS_prior_tests=pd.DataFrame(pd.read_csv('YS_prior.csv', header=None)).to_numpy().reshape(-1)/nz1
EUTS_prior_tests=pd.DataFrame(pd.read_csv('EUTS_prior.csv', header=None)).to_numpy().reshape(-1)/nz3
UTStoYS_prior_tests=pd.DataFrame(pd.read_csv('UTStoYS_prior.csv', header=None)).to_numpy().reshape(-1)/nz2

test_priors = [[YS_pb_prior_tests,YS_prior_tests],[UTStoYS_prior_tests],[EUTS_prior_tests],[np.zeros(N_test)],[np.zeros(N_test)]]

# test_priors = [[[],[]],[[]],[[]],[[]],[[]]]

# for j in range(len(prior_existence)):
#     for jj in range(len(prior_existence[j])):
#         if prior_existence[j][jj]:
#             test_priors[j][jj]=priors_models[j][jj](x_test)/normal[j]
#         else:
#             test_priors[j][jj]=np.zeros([N_test])


for i in range(N_GP):
    pd.DataFrame(np.array(i).reshape(1,1)).to_csv("current_GP.csv", header=None, index=None)
                
    
    ##### if prior==True for a model, condition the GP on data-priors
    
    GP_o1_GT=gp_model(o1_GT_x, o1_GT_y.reshape(o1_GT_x.shape[0])-priors_values[0][0], lhp[i], (np.max(o1_GT_y)*0.625)**2, o1_GT_sd.reshape(-1), N_dim, 'SE' , mean=0)
    GP_o1_is1=gp_model(o1_is1_x, o1_is1_y.reshape(o1_is1_x.shape[0])-priors_values[0][1], lhp[i], (np.max(o1_is1_y)*0.625)**2, o1_is1_sd.reshape(-1), N_dim, 'SE' , mean=0)
    o1 = [GP_o1_GT,GP_o1_is1]
    
    GP_o2_GT=gp_model(o2_GT_x, o2_GT_y.reshape(o2_GT_x.shape[0])-priors_values[1][0], lhp[i], (np.max(o2_GT_y)*0.625)**2, o2_GT_sd.reshape(-1), N_dim, 'SE' , mean=0)
    o2 = [GP_o2_GT]
    
    GP_o3_GT=gp_model(o3_GT_x, o3_GT_y.reshape(o3_GT_x.shape[0])-priors_values[2][0], lhp[i], (np.max(o3_GT_y)*0.625)**2, o3_GT_sd.reshape(-1), N_dim, 'SE' , mean=0)
    o3 = [GP_o3_GT]
    
    GP_o4_GT=gp_model(o4_GT_x, o4_GT_y.reshape(o4_GT_x.shape[0])-priors_values[3][0], lhp[i], (np.max(o4_GT_y)*0.625)**2, o4_GT_sd.reshape(-1), N_dim, 'SE' , mean=0)
    o4 = [GP_o4_GT]
    
    GP_o5_GT=gp_model(o5_GT_x, o5_GT_y.reshape(o5_GT_x.shape[0])-priors_values[4][0], lhp[i], (np.max(o5_GT_y)*0.625)**2, o5_GT_sd.reshape(-1), N_dim, 'SE' , mean=0)
    o5 = [GP_o5_GT]
    
    models = [o1,o2,o3,o4,o5]
    
    fused_means=[]
    fused_vars=[]
    fused_sigs=[]
    
    ### if prior==False, then test_priors is vector of zeros
    
    for j in range(N_obj):
        if len(models[j])==1:
            y_t,var_t = models[j][0].predict_var(x_test)
            fused_means.append(deepcopy(y_t)+test_priors[j][0])
            fused_vars.append(deepcopy(var_t))
        else:
            y_t=[]
            var_t=[]
            for z in range(len(models[j])):
                y_temp,var_temp = models[j][z].predict_var(x_test)
                y_t.append(deepcopy(y_temp)+test_priors[j][z])
                var_t.append(deepcopy(var_temp))
            for k in range(1,len(models[j])):
                var_t[k] = var_t[k] + (y_t[0]-y_t[k])**2
            m,v=reification(y_t,var_t)
            fused_means.append(deepcopy(m))
            fused_vars.append(deepcopy(v))
    
    fused_means=np.transpose(np.array(fused_means))
    fused_vars=np.transpose(np.array(fused_vars))
    fused_sigs=abs(fused_vars)**0.5
                
    
    n_jobs=multiprocessing.cpu_count()
    def calc(ii):
        e = EHVI(fused_means[ii].reshape(1,-1),fused_sigs[ii].reshape(1,-1),goal,ref,y_pareto_curr)
        return e

    Ehvi=Parallel(n_jobs)(delayed(calc)(np.array([jj])) for jj in range(N_test))
    
    ### neutral acquisition function
    Ehvi=np.array(Ehvi)
    
    ### constraint-aware and risk-aware acquisition function
    # risk_aware_prob = test_prob**10
    Ehvi = p_test*Ehvi.ravel()
        
    x_star=np.argmax(Ehvi)
    candidates.append(deepcopy(x_test[x_star]))
    improvements.append(deepcopy(Ehvi[x_star]))
    indices.append(x_star)
    
    
kmedoids = KMedoids(n_clusters=Batch_size, method='pam', max_iter=500, random_state=0).fit(candidates)
x_query=kmedoids.cluster_centers_


pd.DataFrame(np.array(indices).reshape(-1,1)).to_csv("all_candidates_indices.csv", header=None, index=None)
pd.DataFrame(x_query).to_csv("x_query.csv", header=None, index=None)
pd.DataFrame(np.array(candidates)).to_csv("all_candidates.csv", header=None, index=None)
pd.DataFrame((np.array(improvements)).reshape(N_GP,1)).to_csv("all_improvements.csv", header=None, index=None)


medoid_indices = kmedoids.medoid_indices_.astype('int')
pd.DataFrame(np.array(medoid_indices).reshape(-1,1)).to_csv("x_query_indices.csv", header=None, index=None)











