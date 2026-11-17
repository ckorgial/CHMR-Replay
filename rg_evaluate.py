"""Run from the project directory after run.py benchmark. Exports paired primary evidence."""
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from models import synthetic, combine, replay, kalman, imm
from rg import forecast
from kitti import extract
import os

cfg=json.loads(Path('protocol.json').read_text())
out=Path('rg_results')
out.mkdir(exist_ok=True)
settings=pd.read_csv('results/settings.csv').set_index('suite')
metrics=[]; horizons=[]; gates=[]; cuts=[]; strata=[]; influences=[]
def evaluate(label,suite,y,r,groups,s,prior,clusters=None):
    start=time.perf_counter()
    p,gate=forecast(y,s,prior,cfg)
    if y.shape[1]==60:
        p['IMM']=imm(y,[s.qcv,s.qca],s.sigma,s.stay)
    elapsed=time.perf_counter()-start
    h=r.shape[1]
    ids,ix=np.unique(np.arange(len(y)) if clusters is None else clusters,return_inverse=True)
    draws=np.random.default_rng(907).multinomial(len(ids),np.ones(len(ids))/len(ids),cfg['paired_bootstrap'])
    def sums(v):return draws@np.bincount(ix,weights=v,minlength=len(ids))
    denominator=sums(np.full(len(y),h))
    boot={}; points={}
    for name,pred in p.items():
        error=pred[:,:h]-r
        boot[name]=np.sqrt(sums((error**2).sum(1))/denominator)
        points[name]=np.sqrt(np.mean(error**2))
        for j in range(h):
            horizons.append((suite,label,name,j+1,np.sqrt(np.mean(error[:,j]**2)),np.abs(error[:,j]).mean()))
        for g in np.unique(groups):
            m=groups==g
            strata.append((suite,label,name,g,m.sum(),np.sqrt(np.mean(error[m]**2))))
        for side,mask in [('false',r<=15),('useful',r>15),('near false',(r<=15)&(r>=14))]:
            total=mask.sum(1); yes=((pred[:,:h]>15)&mask).sum(1)
            den=sums(total); ratios=sums(yes)[den>0]/den[den>0] if np.any(den>0) else []
            ci=np.quantile(ratios,[.025,.975]) if len(ratios) else [np.nan]*2
            cuts.append((suite,label,name,side,yes.sum(),total.sum(),*ci))
        if clusters is not None:
            for k in ids:
                influences.append((label,name,k,np.sqrt(np.mean(error[clusters!=k]**2))))
    for name in p:
        for ref in ['CA','AR-CA','CHMR','Fixed mix','G','Equal']:
            lo,hi=np.quantile(boot[name]-boot[ref],[.025,.975])
            metrics.append((suite,label,name,ref,len(y),len(ids),points[name],points[name]-points[ref],100*(points[name]/points[ref]-1),lo,hi,np.abs(p[name][:,:h]-r).mean(),prior,elapsed))
    for lo,hi in zip(np.arange(4)/4,np.arange(1,5)/4):
        m=(gate[:,:h]>=lo)&((gate[:,:h]<hi) if hi<1 else True)
        wins=np.abs(p['CA'][:,:h]-r)<np.abs(p['PC'][:,:h]-r)
        gates.append((suite,label,lo,hi,m.sum(),wins[m].mean() if m.any() else np.nan))
    if label in ('confirmation','original','masked'):
        np.savez_compressed(out/f'{suite}_{label}.npz',reference=r,group=groups,cluster=ix,gate=gate,**p)
def run(suite,i,stage,cache):
    s=settings.loc[suite].copy()
    if suite=='KITTI':
        os.environ['CHMR_OFFLINE']='1'
        yy,rr,seq=extract(cache,out); dev=seq%3==0; yd,rd=yy[dev],rr[dev]
    else:yd,rd,*_=synthetic(suite,500,cfg['development'][i])
    base=combine(replay(yd),s['lambda'])[0]
    d=kalman(yd,3,s.qca,s.sigma)-base
    prior=float(np.clip(np.sum(d*(rd[:,60:]-base))/np.sum(d*d),0,1))
    if suite=='KITTI':
        evaluate('masked',suite,yy[~dev],rr[~dev,60:],seq[~dev],s,prior,seq[~dev]);return
    if stage=='main':
        for label,seeds,n in [('original',[cfg['original'][i]],2000),('confirmation',[cfg['confirmation'][i],cfg['additional_seeds'][i]],1000)]:
            data=[synthetic(suite,n,seed) for seed in seeds]
            y,r,g,_=[np.concatenate(x) for x in zip(*data)]
            evaluate(label,suite,y,r[:,60:],g,s,prior)
    else:
        for label in cfg['stress_conditions']:
            rho=float(label.split()[1]) if label.startswith('rho ') else None
            y,r,g,_=synthetic(suite,cfg['stress_n'],cfg['stress_seed']+i,rho=rho)
            st=s.copy(); u=(np.arange(70)-59)*.1; truth=r[:,:60].copy()
            if label=='low noise':y=truth+.5*(y-truth)
            if label=='high noise':y=truth+2*(y-truth)
            if label=='range noise':y=truth+(r[:,:60]/20)*(y-truth)
            offset=np.zeros(70)
            if label=='abrupt pre-loss braking':offset=-np.maximum(u+.5,0)**2
            if label=='gradual pre-loss braking':offset=-np.maximum(u+1,0)**3/6
            if label=='post-loss braking':offset=-np.maximum(u,0)**2
            if label=='variable velocity':offset=np.random.default_rng(845).uniform(-2,2,(len(y),1))*u
            y=y+offset[...,:60]; r=r+offset
            if label=='CA q x10':st.qca*=10
            if label=='short history':y=y[:,-45:]
            evaluate(label,suite,y,r[:,60:65 if label=='short outage' else 70],g,st,prior)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['main','stress','kitti']);ap.add_argument('--cache',default='cache')
    args=ap.parse_args()
    for i,suite in enumerate(['KITTI'] if args.stage=='kitti' else ['MR','PA']):run(suite,i,args.stage,args.cache)
    for name,rows,columns in [
        ('metrics',metrics,'suite split method comparator encounters clusters rmse difference percent ci_low ci_high mae prior batch_seconds'),
        ('horizons',horizons,'suite split method h rmse mae'),('screen',cuts,'suite split method outcome events eligible low high'),
        ('regimes',strata,'suite split method group encounters rmse'),('gate',gates,'suite split lower upper count ca_win_fraction'),
        ('influence',influences,'split method omitted rmse')]:
        pd.DataFrame(rows,columns=columns.split()).to_csv(out/f'{args.stage}_{name}.csv',index=False)
    files=['protocol.json','rg.py','rg_evaluate.py','models.py','results/settings.csv']
    (out/f'{args.stage}_manifest.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},indent=2))
