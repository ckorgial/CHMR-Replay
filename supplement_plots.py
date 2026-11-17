from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from colour_figures import export
folder=Path('supplement')
rh=pd.read_csv(folder/'scores.csv')
w=pd.read_csv(folder/'weights.csv')
fig,axes=plt.subplots(2,3,figsize=(10,5),sharex=True,sharey=True)
for ax,(suite,group) in zip(axes.flat,[('MR','Cruise'),('MR','Queue'),('MR','Cut-in'),('PA','all'),('KITTI','all')]):
    for name,p in w[(w.suite==suite)&(w.group==group)].groupby('method'):
        ax.plot(p.h,p['mean'],label=name)
        if name in ['Matched','Pooled']:ax.fill_between(p.h,p.lower,p.upper,alpha=.15)
    if suite=='KITTI' and w[w.suite=='KITTI'].empty:
        ax.text(.5,.5,'KITTI not rerun\n(raw data unavailable)',ha='center',va='center',transform=ax.transAxes,fontsize=10)
    ax.set(title=suite+' '+group,xlabel='Horizon (frames)',ylabel='Quadratic weight')
axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center')
fig.tight_layout();export(fig,'weight_by_regime')
st=pd.read_csv(folder/'stability.csv')
fig,axes=plt.subplots(2,2,figsize=(9.3,6.6))
labels={'W':'FIR history length W (frames)',
        'K':'Replay origins K',
        'ridge':r'Ridge penalty $\lambda$ (m$^2$)',
        'beta':r'Recency factor $\beta$'}
for ax,f in zip(axes.flat,['W','K','ridge','beta']):
    for name,data in st[st.factor==f].groupby('suite'):
        grouped=(data.groupby('value').variance.mean() if f=='K'
                 else data.groupby('value').rmse.apply(lambda arr:np.sqrt(np.mean(arr**2))))
        ax.plot(grouped.index,grouped.to_numpy(),'o-',label=name,markersize=4)
    ax.set(xlabel=labels[f],ylabel='Mean fitted-weight variance' if f=='K' else 'RMSE (m)')
    ax.grid(axis='y',alpha=.2)
    if f=='ridge': ax.set_xscale('log')
axes[0,0].legend(frameon=False)
fig.tight_layout();export(fig,'stability_2x2')
fig,axes=plt.subplots(1,3,figsize=(10,3))
subs=pd.read_csv('results/subgroups.csv')
for i,ax in enumerate(axes):
    for m,p in subs[(subs.suite=='MR')&(subs['subset']==f'group {i}')&(subs.method.isin(['One-step','Pooled','CHMR','CA','AR-CA']))].groupby('method'):
        ax.plot(p.horizon,p.rmse,label=m)
    ax.set(xlabel='Horizon (frames)',ylabel='RMSE (m)',title=['Cruise','Queue','Cut-in'][i],ylim=(0,1.1))
axes[0].legend(fontsize=7);fig.tight_layout();export(fig,'regime_filters')
if (folder/'sequences.csv').exists():
    seq=pd.read_csv(folder/'sequences.csv')
    fig,ax=plt.subplots(figsize=(6,3))
    for m,p in seq[seq.method.isin(['CHMR','Pooled','Equal'])].groupby('method'):
        ax.plot(p.sequence,p.sequence_rmse,'o-',label=m)
    ax.set(xlabel='KITTI sequence',ylabel='Sequence RMSE (m)');ax.legend();fig.tight_layout();export(fig,'sequence_errors')
fig,axes=plt.subplots(1,2,figsize=(8,3))
c=pd.read_csv(folder/'correlation.csv')
for rho,p in c[(c.K==20)&(c.method=='matched')].groupby('rho'):
    axes[0].plot(p.h,p.proxy-p.latent,label=str(rho))
axes[0].set(xlabel='Horizon (frames)',ylabel='Proxy minus latent optimum');axes[0].legend(title='Correlation',fontsize=7)
s=pd.read_csv(folder/'stress.csv')
s[s.scenario.str.startswith('unseen')].pivot(index='method',columns='scenario',values='rmse').plot.bar(ax=axes[1],rot=35)
axes[1].set(ylabel='RMSE (m)',xlabel='');axes[1].legend(fontsize=7);fig.tight_layout();export(fig,'correlation_stress')
