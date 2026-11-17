"""Independent algebra and causal-index checks; additional matched-budget controls."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype':42,'font.size':10})
from colour_figures import curves
from models import bank,combine,replay,synthetic,kalman
from kitti import extract
from scientific_config import selected_suites, cache_directory
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--cache',default=str(cache_directory()))
args=parser.parse_args()
out=Path('results')
rng=np.random.default_rng(1517)
x=rng.normal(size=(4,60))
for p in (1,2):
    for h in (1,5,10):
        coef=np.polyfit(np.arange(-14,1)*.1,x[0,-15:],p)
        assert np.isclose(np.polyval(coef,h*.1),x[0,-15:]@bank(15,p)[:,h-1])
stats=replay(x)
for h in range(1,11):
    dif=[];res=[]
    for o in range(30,50):
        values=[np.polyval(np.polyfit(np.arange(-14,1)*.1,x[0,o-14:o+1],p),.1*h) for p in (1,2)]
        dif.append(values[1]-values[0]);res.append(x[0,o+h]-values[0])
    assert np.isclose(stats[0][0,h-1],np.mean(np.square(dif)))
    assert np.isclose(stats[1][0,h-1],np.mean(np.multiply(dif,res)))
# Curved motion: empirical numerator difference must match the covariance term.
W=15;h=3;rho=.7;sigma=.4
S=sigma**2*rho**abs(np.subtract.outer(np.arange(W+h),np.arange(W+h)))
a=bank(W,1)[:,h-1];d=bank(W,2)[:,h-1]-a
t=.1*np.arange(W+h); motion=16-t+.5*t*t
err=rng.multivariate_normal(np.zeros(W+h),S,100000)
observed=motion+err;D=observed[:,:W]@d
shift=(d@S[:W,-1])/((d@motion[:W])**2+d@S[:W,:W]@d)
emp=np.mean(D*err[:,-1])/np.mean(D**2)
assert abs(shift-emp)<.008
pd.DataFrame([{'analytic':shift,'Monte_Carlo':emp,'N':len(err)}]).to_csv(out/'curved_identity.csv',index=False)

rows=[]
for suite in selected_suites():
    i=['MR','PA','KITTI'].index(suite)
    print(f'[diagnostics] matched-budget checks for {suite}',flush=True)
    test=np.load(out/f'{suite}_predictions.npz')
    if suite=='KITTI':
        ya,ra,seq=extract(args.cache,out);y,r=ya[seq%3==0],ra[seq%3==0]
    else: y,r,_,_=synthetic(suite,500,100+i)
    sd,st=replay(y),replay(test['history'])
    mse=lambda pred,ref:np.mean((pred-ref)**2)
    for method in ('one','pooled','matched'):
        curve=[(.00001*10**k) for k in range(10)]
        opts=[(mse(combine(sd,v,mode=method)[0],r[:,60:]),v) for v in curve]
        opts.append((mse(sd[2]+.5*sd[3],r[:,60:]),np.inf))
        _,v=min(opts)
        pred=combine(st,v,mode=method)[0] if np.isfinite(v) else st[2]+.5*st[3]
        rows.append((suite,method,v,np.sqrt(mse(pred,test['reference'][:,60:]))))
pd.DataFrame(rows,columns=['suite','method','lambda','rmse']).to_csv(out/'extended_grid.csv',index=False)

# Horizon and regime curves use saved forecasts only; no configuration selection.
if 'MR' in selected_suites():
    curves('regimes')
# Fixed configuration across correlations, common innovations and motion draws.
rows=[]
for rho in (0.,.4,.85):
    y,r,_,_=synthetic('MR',2000,200,rho=rho)
    stats=replay(y)
    for mode in ('one','matched','pooled'):
        p,w=combine(stats,.001,mode=mode)
        rows.append((rho,mode,np.sqrt(np.mean((p-r[:,60:])**2)),w.mean()))
pd.DataFrame(rows,columns=['rho','mode','rmse','weight']).to_csv(out/'correlation_sweep.csv',index=False)
