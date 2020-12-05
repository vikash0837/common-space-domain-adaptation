#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
from sklearn.manifold import TSNE
city = np.load("city.npy")
kitti = np.load("kitti.npy")
city_embedded = TSNE(n_components=2).fit_transform(city)
print(city_embedded.shape)
kitti_embedded = TSNE(n_components=2).fit_transform(kitti)
print(kitti_embedded.shape)


# In[12]:


plt.figure(figsize=(8,8))
plt.scatter(city_embedded[:,0], city_embedded[:,1],c='red',label='city')
plt.scatter(kitti_embedded[:,0], kitti_embedded[:,1],c='black',label='kitti')
plt.legend()
plt.savefig("city_kitt.png")


# In[4]:


city_syn = np.load("city_syn2.npy")
kitti_syn = np.load("kitti_syn2.npy")
city_syn_embedded = TSNE(n_components=2).fit_transform(city_syn)
print(city_syn_embedded.shape)
kitti_syn_embedded = TSNE(n_components=2).fit_transform(kitti_syn)
print(kitti_syn_embedded.shape)


# In[13]:


plt.figure(figsize=(8,8))
plt.scatter(city_syn_embedded[:,0], city_syn_embedded[:,1],c='red',label='city_syn')
plt.scatter(kitti_syn_embedded[:,0], kitti_syn_embedded[:,1],c='black',label='kitti_syn')
plt.legend()
plt.savefig("city_syn_kitti_syn.png")


# In[ ]:




