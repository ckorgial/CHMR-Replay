"""Run from the project directory. Results are regenerated from raw data and seeds."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from colour_figures import curves
from models import synthetic, replay, combine, kalman, imm, bank
from kitti import extract
from scientific_config import selected_suites, cache_directory

O=Path('results'); O.mkdir(exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--cache',default=str(cache_directory()))
args=parser.parse_args()
rmse=lambda p,r: np.sqrt(np.mean((p-r)**2))
summ=[]; sensitivity=[]; screening=[]; settings=[]; intervals=[]; slices=[]
grid=[.001,.01,.1,1.,10.]
for suite in selected_suites():
    si=['MR','PA','KITTI'].index(suite)
    print(f'[benchmark] {suite}: generating/loading development and held-out trajectories', flush=True)
    if suite=='KITTI':
        ya,ra,seq=extract(args.cache,O)
        if not len(seq):
            raise ValueError('No KITTI windows recovered under the documented protocol')
        dev=seq%3==0
        yd,rd=ya[dev],ra[dev]
        y,r=ya[~dev],ra[~dev];cluster=seq[~dev]; group=cluster
        changes=np.full(len(y),np.nan)
    else:
        yd,rd,_,_=synthetic(suite,500,100+si)
        y,r,group,changes=synthetic(suite,2000,200+si)
        cluster=np.arange(len(r))
    noise=yd-rd[:,:60]; sigma=np.sqrt(np.mean(noise**2))
    rho=np.clip(np.sum(noise[:,1:]*noise[:,:-1])/np.sum(noise[:,:-1]**2),-.98,.98)
    target=r[:,60:];devtarget=rd[:,60:]
    sd,st=replay(yd),replay(y)
    def choose(mode):
        return min(grid,key=lambda v:rmse(combine(sd,v,mode=mode)[0],devtarget))
    lam=choose('matched'); qgrid=[.01,.1,1.,10.]
    pred={'Linear':st[2],'Quadratic':st[2]+st[3],'Equal':st[2]+.5*st[3]}
    qvalues=[]
    for name,order,corr in [('CV',2,None),('CA',3,None),('AR-CV',2,rho),('AR-CA',3,rho)]:
        q=min(qgrid,key=lambda q:rmse(kalman(yd,order,q,sigma,corr),devtarget))
        qvalues.append(q);pred[name]=kalman(y,order,q,sigma,corr)
    stay=min([.85,.95,.99],key=lambda s:rmse(imm(yd,qvalues[:2],sigma,s),devtarget))
    pred['IMM']=imm(y,qvalues[:2],sigma,stay)
    for name,mode in [('One-step','one'),('Pooled','pooled'),('CHMR','matched')]:
        pred[name]=combine(st,lam,mode=mode)[0]
    alt={}
    for mode in ('one','pooled'):
        alt[mode+' tuned']=combine(st,choose(mode),mode=mode)[0]
    # Conditional on a stationary calibration model, remove the population cross moment.
    h=np.arange(1,11)
    cross=sigma**2*rho**(14-np.arange(15)[:,None]+h)
    delta=((bank(15,2)-bank(15,1))*cross).sum(0)
    alt['Covariance correction']=combine((st[0],st[1]-delta,st[2],st[3]),lam)[0]
    global_weight=np.clip((sd[1].mean(0)+lam*.5)/(sd[0].mean(0)+lam),0,1)
    alt['Development weights']=st[2]+st[3]*global_weight
    alt['No ridge']=combine(st,0)[0]
    alt['Linear prior']=combine(st,lam,0)[0]
    for name,kw in [('Recent origins',{'fresh':True}),('Decay',{'beta':.85}),('W20',{'W':20}),('K30',{'K':30})]:
        alt[name]=combine(replay(y,**kw),lam)[0]
    for v in grid: alt['lambda '+str(v)]=combine(st,v)[0]
    for name,p in alt.items(): sensitivity.append((suite,name,rmse(p,target)))
    settings.append((suite,len(yd),len(y),sigma,rho,lam,choose('one'),choose('pooled'),*qvalues,stay))
    for method,p in pred.items():
        e=p-target
        summ.append((suite,method,rmse(p,target),np.maximum(e,0).mean(),rmse(p[:,:3],target[:,:3]),rmse(p[:,-3:],target[:,-3:])))
        conditions=[('all',np.ones_like(target,bool),target),('near',None,target)]
        if suite=='PA': conditions.append(('stress',np.ones_like(target,bool),target-np.arange(1,11)[None,:]**2*.01))
        for condition,mask,ref in conditions:
            for g in (12,15,18):
                m=(abs(ref-g)<=1) if mask is None else mask
                low=(ref<=g)&m; high=(ref>g)&m
                screening.append((suite,method,condition,g,low.sum(),((p>g)&low).sum(),high.sum(),((p>g)&high).sum()))
        for subset,m in [('group '+str(i),group==i) for i in np.unique(group)]+([('early change',changes<-1.75),('recent change',changes>=-1.75)] if suite=='PA' else []):
            for h in range(10): slices.append((suite,method,subset,h+1,rmse(p[m,h],target[m,h])))
    rng=np.random.default_rng(300+si)
    # The same sampled clusters select both methods; retain all horizons in each cluster.
    ids,inv=np.unique(cluster,return_inverse=True)
    draws=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    def totals(v):
        return draws@np.bincount(inv,weights=v,minlength=len(ids))
    total=totals(np.full(len(y),10))
    for name in ('One-step','Pooled','Linear','AR-CV','AR-CA'):
        for metric in ('RMSE','FCR','UCR'):
            def boot(p):
                if metric=='RMSE': return np.sqrt(totals(((p-target)**2).sum(1))/total)
                ok=(target<=15) if metric=='FCR' else (target>15)
                den=totals(ok.sum(1));return np.divide(100*totals(((p>15)&ok).sum(1)),den,out=np.full(den.shape,np.nan),where=den>0)
            d=boot(pred['CHMR'])-boot(pred[name]); valid=np.isfinite(d)
            ci=np.quantile(d[valid],[.025,.975]) if valid.any() else [np.nan]*2
            intervals.append((suite,name,metric,*ci,valid.sum()))
    np.savez_compressed(O/f'{suite}_predictions.npz',history=y,reference=r,cluster=cluster,group=group,change=changes,**pred,**alt)
    print(f'[benchmark] {suite}: {len(y)} test windows; CHMR RMSE={rmse(pred["CHMR"], target):.6f} m', flush=True)
def save(rows,headers,name):
    pd.DataFrame(rows,columns=headers.split()).to_csv(O/(name+'.csv'),index=False)
save(summ,'suite method rmse positive early late','performance')
save(sensitivity,'suite variant rmse','ablations')
save(screening,'suite method subset g eligible_low false_clear eligible_high useful_clear','thresholds')
save(settings,'suite dev test sigma rho lambda one_lambda pooled_lambda qcv qca qarcv qarca stay','settings')
save(intervals,'suite comparator metric lower upper finite','paired_intervals')
save(slices,'suite method subset horizon rmse','subgroups')

# Fixed linear range and common random numbers isolate the covariance mechanism.
rng=np.random.default_rng(777); innovations=rng.normal(size=(10000,60))*.4
horizons=np.arange(1,11);a=bank(15,1);d=bank(15,2)-a;diag=[]
for rho in (0.,.4,.85):
    n=innovations.copy()
    for t in range(1,60): n[:,t]=rho*n[:,t-1]+np.sqrt(1-rho**2)*innovations[:,t]
    s=replay(20+n);fitted=combine(s,.01)[1].mean(0)
    S=.16*rho**abs(np.subtract.outer(np.arange(15),np.arange(15)))
    C=.16*rho**(14-np.arange(15)[:,None]+horizons)
    denom=(d*(S@d)).sum(0)
    latent=-(d*(S@a)).sum(0)/denom
    proxy=latent+(d*C).sum(0)/denom
    for h,x,z,w in zip(horizons,proxy,latent,fitted):diag.append((rho,h,x,z,w))
save(diag,'rho h proxy range mean_fitted','weight_diagnostic')

curves('weights')
