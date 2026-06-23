#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from matplotlib import pyplot as plt
import re
from operator import itemgetter
from matplotlib import cm
import time
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

#default setting
# from matplotlib import RcParams
# latex_style_times = RcParams({'font.family': 'serif',
#                'font.serif': ['Times'],
#                'text.usetex': True,
#                'axes.linewidth':2.0
#                #'figsize':[6.4,4.8]
#                })

# #plt.style.use(latex_style_times)
# plt.rcParams.update({'font.size': 15})
# plt.rcParams["figure.figsize"]=[6.4,4.8]
# plt.rcParams["figure.dpi"]=300

def read_data(filename,skip_header):
  data=[]
  with open(filename,'r') as in_file:
    count=1
    for line in in_file:
      ll=line.split()
      if count > skip_header:
        if len(ll)>8:
          ll[7]=float(ll[7])
        data.append(ll)
      count +=1
  return data

def extract_element_conc(strings):
  element,concentration=[],[]
  r=re.compile('([A-Za-z]+)([-?\d+\.\d+|\d+]+)')
  while (strings !=''):
    m=r.match(strings)
    element.append(m.group(1))
    concentration.append(float(m.group(2)))
    short_st= m.group(1)+m.group(2)
    strings=strings.replace(short_st,'')
  concentration /= np.sum(concentration)
  return [element,concentration]

def cal_intermediate_quantities(strings,dictionary):
  r=.5 #ratio for elemental volume and voronoi volume
  alloy=extract_element_conc(strings)
  conc=alloy[1]
  G,V,poi=np.zeros(len(alloy[0])), np.zeros(len(alloy[0])), np.zeros(len(alloy[0]))
  for j in range(0,len(alloy[0])):
     k=0
     while (alloy[0][j] !=dictionary[k][0] and k<len(dictionary)):
       k +=1
     V[j]= r*float(dictionary[k][1])+(1-r)*float(dictionary[k][2])
     G[j],poi[j]=dictionary[k][3],dictionary[k][4]
  G_bar,V_bar,poi_bar=np.dot(G,conc),np.dot(V,conc),np.dot(poi,conc)
  V_var=np.dot((V-V_bar)**2,conc)
  return [G_bar,poi_bar,V_var,V_bar]

def cal_intermediate_quantitiesV2(alloy,dictionary):
  r=1 #ratio for elemental volume and voronoi volume
  #alloy=extract_element_conc(strings)
  conc=alloy[1]
  #print("alloy: ",alloy)
  G,V,poi=np.zeros(len(conc)), np.zeros(len(conc)), np.zeros(len(conc))
  for j in range(0,len(conc)):
     k=0
     while (alloy[0][j] !=dictionary[k][0] and k<len(dictionary)):
       k +=1
     V[j]= r*float(dictionary[k][1])+(1-r)*float(dictionary[k][2])
     G[j],poi[j]=dictionary[k][3],dictionary[k][4]
  G_bar,V_bar,poi_bar=np.dot(G,conc),np.dot(V,conc),np.dot(poi,conc)
  V_var=np.dot((V-V_bar)**2,conc)
  return [G_bar,poi_bar,V_var,V_bar]

def cal_tau0_Eb(alloy): #system
  alpha,f_tau,f_Eb= 0.125,0.35,5.7
  dictionary=read_data("table.dat",0) 
  inter_quan=cal_intermediate_quantitiesV2(alloy,dictionary) #(system,dictionary)
  G_bar,poi_bar,V_var,V_bar=inter_quan[0],inter_quan[1],inter_quan[2],inter_quan[3]
  b=(4*V_bar)**(1.0/3)/2.0**0.5
  #G_bar,poi_bar,V_var,V_bar=85,inter_quan[1],inter_quan[2],inter_quan[3] 
  #b=2.53
  K_tau=G_bar*((1+poi_bar)/(1-poi_bar))**(4.0/3)*b**(-4)
  K_Eb =G_bar*((1+poi_bar)/(1-poi_bar))**(2.0/3)*b #**(-2)
  tau0=0.051*alpha**(-1.0/3)*K_tau*f_tau*V_var**(2.0/3)
  Eb  =0.274*alpha**(1.0/3)*K_Eb*f_Eb*V_var**(1.0/3)
  tau0_in_MPa=tau0*10**3
  Eb_in_eV=Eb*0.01/1.6 #6.02
  return [tau0_in_MPa,Eb_in_eV]

def cal_tau_T(system,T):
  kbT=1.38*10**(-23)*T/(1.602*10**(-19))
  init=cal_tau0_Eb(system)
  tau0,Eb=init[0],init[1]
  #print(Eb)
  epsilon0=1e4
  epsilon=0.001 #10**(-3)
  #tau_T=tau0*(1-(kbT/Eb*np.log(epsilon0/epsilon))**(2/3.0))
  tau_T=tau0*np.e**(-1.0/0.51*kbT/Eb*np.log(epsilon0/epsilon))
  return [Eb,tau_T]




#------------------------------------

def cal_taus_HT(list,concs,T):  
  M=3.06
  taus=[]
  for i in range(len(concs)):
    alloy=[list,concs[i]/100.]
    tau_i=cal_tau_T(alloy,T)
    taus.append(M*tau_i[1])
  return taus
def print_taus(list,concs,taus):
  line=""
  for i in range(len(list)):
    line += list[i]+"     "
  line += "tau/MPa"
  print(line)
  for i in range(len(concs)):
    line_i=""
    for j in range(len(concs[i])):
      line_i += str(concs[i][j])+" "
    #line_i += str(concs_i[len(concs[i])])+"	" #+'\n'
    line_i += str(taus[i])
  return None

def print_taus2(list,concs,taus):
  def merge(concs,taus):
    merged=[]
    for i in range(len(taus)):
      concs_i=concs[i].tolist()
      concs_i.append(taus[i])
      merged.append(concs_i)
    return merged
  merged=merge(concs,taus)
  #print(merged[0])
#   merged.sort(key=itemgetter(4),reverse=True) #sort(merged,key=takeTen)
  line=""
  for i in range(len(list)):
    line += list[i]+"     "
  line += "tau/MPa"
  #print(line)  
  for i in range(len(merged)):
    line_i=""
    for j in range(len(merged[i])):
      line_i += str(merged[i][j])+" "
    print(line_i)
  return None





# In[ ]:





# In[12]:


import concurrent.futures
import time
import datetime

max_numbers = [10000000, 10000000, 10000000, 10000000, 10000000]

class Task:
    def __init__(self, max_number):
        self.max_number = max_number
        self.interrupt_requested = False

    def __call__(self):
        print("Started:", datetime.datetime.now(), self.max_number)
        last_number = 0;
        for i in range(1, self.max_number + 1):
            if self.interrupt_requested:
                print("Interrupted at", i)
                break
            last_number = i * i
        print("Reached the end")
        return last_number

    def interrupt(self):
        self.interrupt_requested = True

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(max_numbers)) as executor:
        tasks = [Task(num) for num in max_numbers]
        for task, future in [(i, executor.submit(i)) for i in tasks]:
            try:
                print(future.result(timeout=1))
            except concurrent.futures.TimeoutError:
                print("this took too long...")
                task.interrupt()


# if __name__ == '__main__':
#     main()


# In[ ]:





# In[ ]:




