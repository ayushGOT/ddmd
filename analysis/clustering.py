#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy import linalg as LA
from matplotlib.ticker import FuncFormatter
# %run ~/.mpl_set.py

import networkx as nx

from sknetwork.clustering import Louvain


# In[2]:


#get_ipython().system('jupyter nbconvert --to script clustering.ipynb')


# In[2]:


#pip install scikit-network


# In[2]:


df=pd.read_pickle("result.pkl")


# In[3]:


df


# In[4]:


len(df.cluster_label.unique())


# In[5]:


dtrajs = []    # a 2-D matrix to store the cluster no. of each frame of each simulation
for sys_label in sorted(df.sys_label.unique()): 
    sub_df = df[df['sys_label'] == sys_label]
    dtrajs.append(sub_df['cluster_label'].values)


# In[6]:


#sub_df['cluster_label'].values


# In[7]:


np.shape(sub_df['cluster_label'].values)


# In[8]:


print(np.shape(dtrajs))
type(dtrajs)


# In[9]:


dtrajs[0][2]


# In[10]:


def get_trans_count(dtrajs, lag=1):
    adj_sparse = {}      # dictionary
    for dtraj in dtrajs: 
        for i in range(len(dtraj) - lag): 
            transition = (dtraj[i], dtraj[i+lag])
            if transition in adj_sparse: 
                adj_sparse[transition] += 1 
            else: 
                adj_sparse[transition] = 1

    n_states = len(set(np.concatenate(dtrajs)))   # no. of macrostates i.e. clusters
    print(n_states)
    trans_count = np.zeros((n_states, n_states),dtype=int)
    for edges in adj_sparse: 
        #print(edges)
        
        trans_count[edges] = int(adj_sparse[edges])
        #print(trans_count[edges])
        print(type(adj_sparse[edges]))
        print(type(trans_count[edges]))

    trans_count = trans_count + trans_count.T    # why this? detailed balance?
    return trans_count 

def get_trans_mat(dtrajs, lag=1): 
    trans_count = get_trans_count(dtrajs, lag=lag)
    trans_mat = trans_count / np.sum(trans_count, axis=0)
    return trans_mat


# In[11]:


b={}
print(type(b))
b[1,1]=1
b[2,2]=2
c=np.zeros((2,2))
print(b,c)


# In[12]:


a=1.
print(type(a))
type(a)


# In[13]:


trans_mat_norm = get_trans_mat(dtrajs)
G = nx.from_numpy_array(trans_mat_norm, create_using=nx.DiGraph())
#G = nx.from_numpy_array(trans_mat_norm, create_using=nx.Graph())


# In[14]:


G


# In[15]:


cluster_info = []
for label in df.cluster_label.unique()[:]: 
    sub_df = df[df.cluster_label == label] 
    cluster_info.append({'label': int(label),  
                         'rmsd_mean': sub_df.rmsd.mean(), 
                         'rmsd_std': sub_df.rmsd.std(),
                         'Q_mean': sub_df.Q.mean(), 
                         'Q_std': sub_df.Q.std(),
                        })
#     print(sub_df.dist.count(), sub_df.dist.std(), sub_df.dist.mean())

cluster_info = pd.DataFrame(cluster_info)
cluster_info = cluster_info.sort_values('label').reset_index(drop=True)
cluster_info


# In[16]:


col_names = sorted([col for col in cluster_info.columns])
for i in sorted(G.nodes()):
    for col in col_names:
        G.nodes[i][col] = cluster_info[col][i]


# In[103]:


np.set_printoptions(threshold=1000)


# In[17]:


trans_count = get_trans_count(dtrajs)  # 500 X 500 matrix having info about transitions from cluster/state i to j. Here, 500 is the no. of clusters obtained through k-means
np.savetxt('trans_count.txt', trans_count)
print(trans_count.shape,type(trans_count))
print(trans_count[0])
louvain = Louvain(modularity='newman', random_state=42)
labels = louvain.fit_predict(trans_count)
print(labels.shape,type(labels))
#labels.toarray()
#len((labels))


# In[18]:


len(set(labels))


# In[19]:


for i in sorted(G.nodes()):
    G.nodes[i]['mod'] = labels[i]

cluster_info['mod'] = labels


# In[21]:


nx.write_gexf(G, 'kmeans.gexf')
#nx.write_gpickle(G, 'kmeans.pkl')


# In[22]:


df['mod'] = [labels[i] for i in df['cluster_label']]


# In[29]:


df.to_pickle('./df_kmeans.pkl')


# In[30]:


dtrajs_mod = []
for sys_label in sorted(df.sys_label.unique()): 
    sub_df = df[df['sys_label'] == sys_label]
    dtrajs_mod.append(sub_df['mod'].values)


# In[31]:


trans_mat_mod = get_trans_mat(dtrajs_mod)
trans_mat_mod.shape
G_mod = nx.from_numpy_array(trans_mat_mod, create_using=nx.DiGraph())


# In[32]:


cluster_info = []
for label in df['mod'].unique()[:]: 
    sub_df = df[df['mod'] == label] 
    cluster_info.append({'mod': int(label), 
                         'rmsd_mean': sub_df.rmsd.mean(), 
                         'rmsd_std': sub_df.rmsd.std(),
                         'Q_mean': sub_df.Q.mean(), 
                         'Q_std': sub_df.Q.std(),
                        })

cluster_info = pd.DataFrame(cluster_info)
cluster_info = cluster_info.sort_values('mod').reset_index(drop=True)


# In[33]:


col_names = sorted([col for col in cluster_info.columns])
for i in sorted(G_mod.nodes()):
    for col in col_names:
        G_mod.nodes[i][col] = cluster_info[col][i]


# In[35]:


nx.write_gexf(G_mod, 'kmeans_mod.gexf')
#nx.write_gpickle(G_mod, 'kmeans_mod.pkl')


# In[36]:


cluster_info.to_pickle('./df_kmeans_mod.pkl')

