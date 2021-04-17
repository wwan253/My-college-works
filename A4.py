# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:34:45 2021

@author: Weixu Wang
"""

# In[1]:


import numpy as np
import pandas as pd
from collections import defaultdict
import math
import time
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from scipy import sparse
import gc


# In[2]:


review_df = pd.read_json('review.json', lines = True)


# In[3]:


review_df.sort_values('date', inplace = True)


# In[4]:


del review_df['review_id']
del review_df['date']


# In[5]:


train, test = np.split(review_df, [int(.8*len(review_df))])
print('Train rec:', train.shape)
print('test rec:', test.shape)


# In[6]:


del review_df
gc.collect()


# In[7]:


busid = CategoricalDtype(sorted(train.business_id.unique()), ordered=True)
useid = CategoricalDtype(sorted(train.user_id.unique()), ordered=True)
busmap = {busid.categories[i]:i for i in range(len(busid.categories))}
busmap = defaultdict(lambda:-1, busmap)
usemap = {useid.categories[i]:i for i in range(len(useid.categories))}
usemap = defaultdict(lambda:-1, usemap)


# In[8]:


ratm = csr_matrix((train["stars"],
                  (train.user_id.astype(useid).cat.codes,
                   train.business_id.astype(busid).cat.codes)),
                 shape=(useid.categories.size,
                        busid.categories.size))


# In[9]:


from sklearn.metrics.pairwise import cosine_similarity
st = time.time()
sim_sk = cosine_similarity(ratm.T, dense_output = False)
ed = time.time()


# In[10]:


print('Time taken: {:.2f}s'.format(ed-st))


# In[11]:


def get_sim_pair(sim, pair):
    for p in pair:
        i, j = busmap[p[0]], busmap[p[1]]
        yield sim[i, j]    


# In[36]:


print(*list(get_sim_pair(sim_sk, 
                         [('rjZ0L-P1SRLJfiK5epnjYg', 
                           'cL2mykvopw-kodSyc-z5jA'),
                          ('6H8xfhoZ2IGa3eNiY5FqLA', 
                           'XZbuPXdyA0ZtTu3AzqtQhg'),
                          ('rfwJFFzW6xW2qYfJh14OTA', 
                           'G58YATMKnn-M-RUDWg3lxw'),
                          ('0QSnurP5Ibor2zepJmEIlw', 
                           '6-lmL3sC-axuh8y1SPSiqg')])),
     sep='\n')


# In[15]:


st = time.time()
sim_mat = ratm.T * ratm
l2 = np.sqrt(1 / sim_mat.diagonal())
sim_mat = sim_mat.multiply(l2.reshape(-1, 1)).T.multiply(l2.reshape(-1, 1))
ed = time.time()
sim_mat = sim_mat.tocsr()


# In[16]:


print('Time taken: {:.2f}s'.format(ed-st))


# In[38]:


print(*list(get_sim_pair(sim_mat,
                         [('rjZ0L-P1SRLJfiK5epnjYg',
                           'cL2mykvopw-kodSyc-z5jA'),
                          ('6H8xfhoZ2IGa3eNiY5FqLA', 
                           'XZbuPXdyA0ZtTu3AzqtQhg'),
                          ('rfwJFFzW6xW2qYfJh14OTA', 
                           'G58YATMKnn-M-RUDWg3lxw'),
                          ('0QSnurP5Ibor2zepJmEIlw', 
                           '6-lmL3sC-axuh8y1SPSiqg')])),
      sep='\n')


# In[39]:


print('Difference:', np.abs(sim_mat - sim_sk).sum())
del sim_sk
del sim_mat


# In[80]:


class CF:
    def __init__(self):
        import math
        import numpy as np
        from collections import defaultdict
        from scipy.sparse import csr_matrix
        self.valmaps = {}
        self.rmat = None
        self.allsim = None
    
    def train(self, u, p, r, simfunc):
        # map index
        uids = CategoricalDtype(sorted(np.unique(u)), ordered=True)
        pids = CategoricalDtype(sorted(np.unique(p)), ordered=True)
        
        self.valmaps['p'] = {pids.categories[i] : i 
                             for i in range(len(pids.categories))}
        self.valmaps['p'] = defaultdict(lambda:-1, self.valmaps['p'])
        
        self.valmaps['u'] = {uids.categories[i] : i 
                             for i in range(len(uids.categories))}
        self.valmaps['u'] = defaultdict(lambda:-1, self.valmaps['u'])
        
        
        self.rmat = csr_matrix((r, (np.vectorize(self.valmaps['u'].get)(u), 
                                    np.vectorize(self.valmaps['p'].get)(p))),
                               shape=(uids.categories.size, 
                                      pids.categories.size))
        
        self.allsim = simfunc(self.rmat)
    
    def load(self, model):
        self.valmaps, self.rmat, self.allsim = model
    
    def save(self):
        return self.valmaps, self.rmat, self.allsim
    
    def pred(self, u, p):
        pairs = zip(
            np.vectorize(lambda x:self.valmaps['u'][x])(u), 
            np.vectorize(lambda x:self.valmaps['p'][x])(p))
        re = [0] * len(u)
        i = -1
        for ui, pj in pairs:
            i += 1
            if ui == -1 or pj == -1:
                continue
            hist = self.rmat[ui,].nonzero()[1]
            sims = self.allsim[pj, hist]
            norm = sims.sum()
            if norm == 0:
                continue
            
            if len(hist) > 20:
                p = np.argpartition(-sims, 20)[:20]
                hist = hist[p]
                sims = sims[p]
                
            star = self.rmat[ui, [*hist]]
            re[i] = (star*sims).sum() / norm
            
        return np.array(re)
    
    def test_RMSE(self, u, p, r):
        return math.sqrt(np.mean((self.pred(u, p) - r)**2))
    


# In[52]:


cf = CF()
cf.train(train.user_id.values,
              train.business_id.values,
              train.stars.values,
              lambda A:cosine_similarity(A.T))
mod1 = cf.save()


# In[81]:


rmse1 = cf.test_RMSE(test.user_id.values,
                     test.business_id.values,
                     test.stars.values)
print(rmse1)


# In[151]:


class CFB(CF):
    def __init__(self):
        super().__init__()
        self.bg = None
        self.bus = None
        self.bps = None
    
    def load(self, model):
        self.valmaps, self.rmat, self.allsim,            self.bg, self.bus, self.bps = model
    
    def save(self):
        return (self.valmaps, self.rmat, self.allsim,                self.bg, self.bus, self.bps)
    
    def train(self, u, p, r, simfunc):
        super().train(u, p, r, simfunc)
        self.bg = np.mean(r)
        self.bus = np.array(
            self.rmat.sum(1).flatten())[0] / self.rmat.getnnz(1) - self.bg
        self.bps = np.array(
            self.rmat.sum(0).flatten())[0] / self.rmat.getnnz(0) - self.bg
    
    def pred(self, u, p):
        pairs = zip(
            np.vectorize(lambda x:self.valmaps['u'][x])(u), 
            np.vectorize(lambda x:self.valmaps['p'][x])(p))
        re = [None] * len(u)
        i = -1
        for ui, pj in pairs:
            i += 1
            bu = self.bg + self.bus[ui]
            re[i] = bu + self.bps[pj]
            if ui == -1 or pj == -1:
                continue
            hist = self.rmat[ui,].nonzero()[1]
            sims = self.allsim[pj, hist]
            norm = sims.sum()
            if norm == 0:
                continue
           
            if len(hist) > 20:
                p = np.argpartition(-sims, 20)[:20]
                hist = hist[p]
                sims = sims[p]
                
            star = self.rmat[ui, [*hist]].toarray()[0]

            re[i] += ((star-bu-self.bps[hist])*sims).sum() / norm 
            
        return np.array(re)
    


# In[152]:


cf = CFB()
cf.train(train.user_id.values,
              train.business_id.values,
              train.stars.values,
              lambda A:cosine_similarity(A.T))
mod2 = cf.save()
rmse2 = cf.test_RMSE(test.user_id.values,
                     test.business_id.values,
                     test.stars.values)
print(rmse2)


# In[153]:


train, valid = np.split(train, [int(len(train)*7/8)])
print('Train rec:', train.shape)
print('Train rec:', valid.shape)
print('test rec:', test.shape)


# The SGD equation can be rearranged to have less steps to be calculated on-the-fly
# 
# $$
# \begin{aligned}
# q_{i}^{(t+1)} &\gets q_i^{(t)} - \eta\nabla_{qi}\mathcal{L} \\
# q_{i}^{(t+1)} &\gets q_i^{(t)} - \eta\left(-2(r_{ij}-q_i^{(t)}\cdot p_j^{(t)})p_j^{(t)} + 2\lambda_1 q_i^{(t)}\right)\\
# q_{i}^{(t+1)} &\gets q_i^{(t)} - \eta\left(2\lambda_1 q_i^{(t)} - 2(r_{ij}-q_i^{(t)}\cdot p_j^{(t)})p_j^{(t)}\right)\\
# q_{i}^{(t+1)} &\gets q_i^{(t)} - 2\eta\left(\lambda_1 q_i^{(t)} - (r_{ij}-q_i^{(t)}\cdot p_j^{(t)})p_j^{(t)}\right)\\\\
# let\ \mathscr{d} = r_{ij}-q_i^{(t)}\cdot p_j^{(t)}\ \ Then\\\\
# q_{i}^{(t+1)} &\gets q_i^{(t)} - 2\eta\left(\lambda_1 q_i^{(t)} - \mathscr{d} p_j^{(t)}\right)\\
# p_{j}^{(t+1)} &\gets p_j^{(t)} - 2\eta\left(\lambda_2 p_j^{(t)} - \mathscr{d} q_i^{(t)}\right)\\\\
# Let\ \ \eta` = 2\eta\\
# Given\ \ \lambda_1 = \lambda_2 = 0.3 = \lambda \\\\
# q_{i}^{(t+1)} &\gets q_i^{(t)} - \eta`\left(\lambda q_i^{(t)} - \mathscr{d} p_j^{(t)}\right)\\
# p_{j}^{(t+1)} &\gets p_j^{(t)} - \eta`\left(\lambda p_j^{(t)} - \mathscr{d} q_i^{(t)}\right)
# \end{aligned}
# $$

# In[535]:


class LFM:
    def __init__(self, k=None, seed=0):
        import numpy as np
        from scipy import sparse
        self.k = k
        self.Q = None
        self.P = None
        self.seed = seed
        self.valmaps = {}
        
    def epoch(self, data, eta_, lamb):
        Q = self.Q
        P = self.P
        for i, j, r in data:
            d = r - Q[i].dot(P[j])
            Q[i] -= eta_*(lamb*Q[i] - d*P[j])
            P[j] -= eta_*(lamb*P[j] - d*Q[i])

    def parse_data(self, u, p, r):
        uids = CategoricalDtype(sorted(np.unique(u)), ordered=True)
        pids = CategoricalDtype(sorted(np.unique(p)), ordered=True)
        
        self.valmaps['p'] = {pids.categories[i] : i 
                             for i in range(len(pids.categories))}
        self.valmaps['p'] = defaultdict(lambda:-1, self.valmaps['p'])
        
        self.valmaps['u'] = {uids.categories[i] : i 
                             for i in range(len(uids.categories))}
        self.valmaps['u'] = defaultdict(lambda:-1, self.valmaps['u'])
        
        data = csr_matrix((r, (np.vectorize(self.valmaps['u'].get)(u), 
                                    np.vectorize(self.valmaps['p'].get)(p))),
                               shape=(uids.categories.size, 
                                      pids.categories.size))
        return data
            
    def train(self, u, p, r, eta_, lamb, eps):
        data = self.parse_data(u, p, r)
        np.random.seed(self.seed)
        self.Q = np.random.rand(data.shape[0], self.k)
        self.P = np.random.rand(data.shape[1], self.k)
        data = [*zip(*sparse.find(data))]
        for ep in range(eps):
            self.epoch(data, eta_, lamb)
            print('=', end='')
        print()
        
    def save(self):
        return (self.k, self.Q, self.P, self.valmaps, self.seed)
    
    def load(self, model):
        self.k, self.Q, self.P, self.valmaps, self.seed = model
            
    def pred(self, u, p):
        _u = np.vectorize(lambda x:self.valmaps['u'][x])(u)
        _p = np.vectorize(lambda x:self.valmaps['p'][x])(p)
        
        re = np.zeros(len(_u))
        seen = np.intersect1d(np.where(_u != -1)[0], 
                              np.where(_p != -1)[0])
        re.flat[seen] += (self.Q[_u[seen]]*self.P[_p[seen]]).sum(1)
        
        return re
        
    def test_RMSE(self, u, p, r):
        return math.sqrt(np.mean((self.pred(u, p) - r)**2))


# In[489]:


for f in [8, 16, 32, 64]:
    lfm = LFM(f)
    lfm.train(train.user_id.values,
              train.business_id.values,
              train.stars.values,
              eta_ = 2 * 0.01,
              lamb = 0.3,
              eps = 20)
    print('f =', f, 'Train RMSE:', lfm.test_RMSE(train.user_id.values,
                                                 train.business_id.values,
                                                 train.stars.values),
          'Validation RMSE:', lfm.test_RMSE(valid.user_id.values,
                                            valid.business_id.values,
                                            valid.stars.values))


# In[547]:


lfm = LFM(32)
lfm.train(train.user_id.values,
          train.business_id.values,
          train.stars.values,
          eta_ = 2 * 0.01,
          lamb = 0.3,
          eps = 20)
mod3 = lfm.save()


# In[563]:


class LFMb(LFM):
    def __init__(self, k=None, seed=0):
        super().__init__(k,seed)
        self.bus = None
        self.bps = None
        self.bg = None
    
    def save(self):
        return super().save(), self.bus, self.bps, self.bg
    
    def load(self, model):
        mod, self.bus, self.bps, self.bg = model
        super().load(mod)
        
    def epoch(self, data, eta_, lamb):
        bus = list(self.bus)
        bps = list(self.bps)
        bg = self.bg
        Q = self.Q
        P = self.P
        for i, j, r in data:
            d = r - Q[i].dot(P[j]) - bus[i] - bps[j] - bg
            Q[i] -= eta_*(lamb*Q[i] - d*P[j])
            P[j] -= eta_*(lamb*P[j] - d*Q[i])
            bus[i] -= eta_*(lamb*bus[i] - d)
            bps[j] -= eta_*(lamb*bps[j] - d)
        self.bus = np.array(bus)
        self.bps = np.array(bps)
    
    def train(self, u, p, r, eta_, lamb, eps):
        data = self.parse_data(u, p, r)
        np.random.seed(self.seed)
        self.Q = np.random.rand(data.shape[0], self.k)
        self.P = np.random.rand(data.shape[1], self.k)
        self.bus = np.zeros(data.shape[0], dtype=np.float64)
        self.bps = np.zeros(data.shape[1], dtype=np.float64)
        self.bg = np.mean(r)
        data = [*zip(*sparse.find(data))]
        for ep in range(eps):
            self.epoch(data, eta_, lamb)
            print('=', end='')
        print()
        
    def pred(self, u, p):
        _u = np.vectorize(lambda x:self.valmaps['u'][x])(u)
        _p = np.vectorize(lambda x:self.valmaps['p'][x])(p)
        
        re = np.repeat(self.bg, len(_u))
        seen_u = np.where(_u != -1)[0]
        seen_p = np.where(_p != -1)[0]
        seen = np.intersect1d(seen_u, seen_p)
        re.flat[seen] += (self.Q[_u[seen]]*self.P[_p[seen]]).sum(1)
        re.flat[seen_u] += self.bus[_u[seen_u]]
        re.flat[seen_p] += self.bps[_p[seen_p]]
        
        return re


# In[543]:


for f in [8, 16, 32, 64]:
    lfm = LFMb(f)
    lfm.train(train.user_id.values,
              train.business_id.values,
              train.stars.values,
              eta_ = 2 * 0.01,
              lamb = 0.3,
              eps = 20)
    print('f =', f, 'Train RMSE:', lfm.test_RMSE(train.user_id.values,
                                                 train.business_id.values,
                                                 train.stars.values),
          'Validation RMSE:', lfm.test_RMSE(valid.user_id.values,
                                            valid.business_id.values,
                                            valid.stars.values))


# In[564]:


lfm = LFMb(8)
lfm.train(train.user_id.values,
          train.business_id.values,
          train.stars.values,
          eta_ = 2 * 0.01,
          lamb = 0.3,
          eps = 20)
mod4 = lfm.save()


# In[550]:


lmf = LFM()
lmf.load(mod3)
pa = lmf.test_RMSE(test.user_id.values,
                   test.business_id.values,
                   test.stars.values)


# In[574]:


lmf = LFMb()
lmf.load(mod4)
pb = lmf.test_RMSE(test.user_id.values,
                   test.business_id.values,
                   test.stars.values)


# In[575]:


print('RMSE of latent factor model without bias:', pa)
print('RMSE of latent factor model with bias:', pb)

