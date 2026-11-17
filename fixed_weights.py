"""Rebuild Fig. 2 data with Monte Carlo uncertainty and pooled estimates."""
import numpy as np
import pandas as pd
from models import bank, replay, combine
rng=np.random.default_rng(777)
innovations=.4*rng.normal(size=(10000,60))
a=bank(15,1);d=bank(15,2)-a
rows=[]
for rho in (0.,.4,.85):
    noise=innovations.copy()
    for j in range(1,60):
        noise[:,j]=rho*noise[:,j-1]+np.sqrt(1-rho**2)*innovations[:,j]
    stats=replay(20+noise)
    w=combine(stats,.01)[1];pooled=combine(stats,.01,mode='pooled')[1]
    S=.16*rho**abs(np.subtract.outer(np.arange(15),np.arange(15)))
    C=.16*rho**(14-np.arange(15)[:,None]+np.arange(1,11))
    G=(d*(S@d)).sum(0)
    latent=-(d*(S@a)).sum(0)/G
    proxy=latent+(d*C).sum(0)/G
    for j in range(10):
        m=w[:,j].mean();margin=1.96*w[:,j].std(ddof=1)/np.sqrt(len(w))
        rows.append((rho,j+1,proxy[j],latent[j],m,m-margin,m+margin,pooled.mean()))
pd.DataFrame(rows,columns=['rho','h','proxy','range','mean_fitted','lower','upper','pooled']).to_csv('results/weight_diagnostic.csv',index=False)
