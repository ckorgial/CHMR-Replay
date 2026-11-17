"""Held-out weight diagnostics. Run after experiment.py and diagnostics.py."""
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from models import bank, synthetic, combine, replay
from kitti import extract
from scientific_config import selected_suites,cache_directory

out=Path('results')
supp=Path('supplement')
supp.mkdir(exist_ok=True)
settings=pd.read_csv(out/'settings.csv').set_index('suite')
records={}
def record(table, **fields):
    records.setdefault(table, []).append(fields)

def oracle(r,scale,rho):
    """Conditional on each full trajectory AND its simulated marginal scale."""
    h=np.arange(1,11)
    a=bank(15,1)
    d=bank(15,2)-a
    lag=np.abs(np.subtract.outer(np.arange(15),np.arange(15)))
    corr=rho**lag
    windows=sliding_window_view(r[:,:60],15,axis=1)[:,16:36]
    signal=windows@d
    err=r[:,np.arange(30,50)[:,None]+h]-windows@a
    var=scale[:,None]**2
    denom=(signal**2).mean(1)+var*(d*(corr@d)).sum(0)
    latent=((signal*err).mean(1)-var*(d*(corr@a)).sum(0))/denom
    shift=var*(d*rho**(14-np.arange(15)[:,None]+h)).sum(0)/denom
    return latent,latent+shift

def draws(ids,seed):
    ids,inv=np.unique(ids,return_inverse=True)
    rng=np.random.default_rng(seed)
    return rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),2000),inv

def total(x,bs):
    counts,inv=bs
    return counts@np.bincount(inv,x,minlength=counts.shape[1])

rmse=lambda e:np.sqrt(np.mean(e**2))
for suite in selected_suites():
    i=['MR','PA','KITTI'].index(suite)
    print(f'[report] derived scores, weighting, ablations, and bootstrap: {suite}',flush=True)
    z=np.load(out/f'{suite}_predictions.npz')
    y,r,g,cl=[z[k] for k in ('history','reference','group','cluster')]
    target=r[:,60:]
    if suite=='KITTI':
        all_y,all_r,seq=extract(cache_directory(),out)
        yd,rd=all_y[seq%3==0],all_r[seq%3==0]
        np.testing.assert_allclose(y,all_y[seq%3!=0])
    else:
        yd,rd,*_=synthetic(suite,500,100+i)
        repeat,_,_,_,scale=synthetic(suite,2000,200+i,return_scale=True)
        np.testing.assert_array_equal(y,repeat)
        latent,proxy=oracle(r,scale,.4 if i==0 else 0.)
    lam=settings.loc[suite,'lambda']
    train=replay(yd)
    stats=replay(y)
    fitted,wm=combine(stats,lam)
    pooled,wp=combine(stats,lam,mode='pooled')
    dev_pool=combine(train,lam,mode='pooled')[0]
    delta=combine(train,lam)[0]-dev_pool
    gamma=np.clip(np.sum(delta*(rd[:,60:]-dev_pool))/np.sum(delta**2),0,1)
    forecasts={k:z[k] for k in ('Linear','Quadratic','Equal','One-step','Pooled','CHMR','CV','CA','AR-CV','AR-CA','IMM')}
    forecasts['Partial']=pooled+gamma*(fitted-pooled)
    if suite!='KITTI':
        forecasts['Latent replay oracle']=stats[2]+np.clip(latent,0,1)*stats[3]
    bs=draws(cl,920+i)
    n=total(np.ones(len(y)),bs)
    ref=(pooled-target)**2
    for method,p in forecasts.items():
        sq=(p-target)**2
        for h in range(11):
            a=sq.mean(1) if h==0 else sq[:,h-1]
            b=ref.mean(1) if h==0 else ref[:,h-1]
            diff=np.sqrt(total(a,bs)/n)-np.sqrt(total(b,bs)/n)
            lo,hi=np.quantile(diff,[.025,.975])
            record('scores',suite=suite,method=method,h=h,rmse=np.sqrt(a.mean()),diff_vs_pooled=np.sqrt(a.mean())-np.sqrt(b.mean()),lower=lo,upper=hi,gamma=gamma)
    groups={'all':np.ones(len(y),bool)}
    if suite=='MR':
        groups.update({name:g==j for j,name in enumerate(('Cruise','Queue','Cut-in'))})
    for name,mask in groups.items():
        subbs=draws(cl[mask],950+i)
        size=total(np.ones(mask.sum()),subbs)
        weights={'Matched':wm,'Pooled':np.broadcast_to(wp,wm.shape)}
        if suite!='KITTI': weights.update(latent=latent,proxy=proxy)
        for method,w in weights.items():
            for j,col in enumerate(w[mask].T):
                low,high=np.quantile(total(col,subbs)/size,[.025,.975])
                record('weights',suite=suite,group=name,method=method,h=j+1,mean=col.mean(),sd=col.std(ddof=1),median=np.median(col),q25=np.quantile(col,.25),q75=np.quantile(col,.75),at0=np.mean(col==0),at1=np.mean(col==1),lower=low,upper=high,mse_to_unrestricted_latent=np.mean((col-latent[mask,j])**2) if suite!='KITTI' else np.nan)
    for factor,values in dict(K=[10,20,30],W=[10,15,20,25],beta=[.85,.95,1.],ridge=[.00001,.0001,.001,.01,.1,1.,10.,100.,1000.,10000.]).items():
        for val in values:
            st=stats if factor=='ridge' else replay(y,**{factor:val})
            pred,w=combine(st,val if factor=='ridge' else lam)
            for j in range(10):
                record('stability',suite=suite,factor=factor,value=val,h=j+1,rmse=rmse(pred[:,j]-target[:,j]),variance=w[:,j].var(),clipped=np.mean((w[:,j]==0)|(w[:,j]==1)))
    for mode,t in [('original',target)]+([('braking',target-.01*np.arange(1,11)**2)] if suite=='PA' else []):
        for band,mask in [('all',np.ones_like(t,bool)),('near',abs(t-15)<=1)]:
            for label,eligible in [('false',t<=15),('useful',t>15)]:
                selected=eligible&mask
                den=selected.sum(1)
                num=((fitted>15)&selected).sum(1)
                d=total(den,bs);v=d>0
                rates=total(num,bs)[v]/d[v]
                ci=np.quantile(rates,[.025,.975]) if len(rates) else [np.nan]*2
                record('screening',suite=suite,scenario=mode,band=band,rate=label,num=num.sum(),den=den.sum(),lower=ci[0],upper=ci[1],valid=v.sum())
    if suite=='KITTI':
        for seq in np.unique(cl):
            for m,p in forecasts.items():
                record('sequences',sequence=seq,windows=(cl==seq).sum(),method=m,sequence_rmse=rmse((p-target)[cl==seq]),remaining_rmse=rmse((p-target)[cl!=seq]),remaining_diff_vs_pooled=rmse((p-target)[cl!=seq])-rmse((pooled-target)[cl!=seq]))
for rho in (0.,.2,.4,.6,.85):
    y,r,_,_,scale=synthetic('MR',2000,200,rho=rho,return_scale=True)
    latent,proxy=oracle(r,scale,rho)
    for K in (20,30):
        st=replay(y,K=K)
        bankdiff=bank(15,2)-bank(15,1)
        cross=(bankdiff*rho**(14-np.arange(15)[:,None]+np.arange(1,11))).sum(0)
        for method,v,mode in [('matched',0,'matched'),('pooled',0,'pooled'),('known scale',scale[:,None]**2,'matched'),('calibrated scale',settings.loc['MR','sigma']**2,'matched')]:
            p,w=combine((st[0],st[1]-v*cross,*st[2:]),.001,mode=mode)
            w=np.broadcast_to(w,p.shape)
            for j in range(10):
                record('correlation',rho=rho,K=K,method=method,h=j+1,rmse=rmse(p[:,j]-r[:,60+j]),mean=w[:,j].mean(),variance=w[:,j].var(),clipped=np.mean((w[:,j]==0)|(w[:,j]==1)),proxy=proxy[:,j].mean(),latent=latent[:,j].mean())
rng=np.random.default_rng(645)
u=.1*(np.arange(70)-59)
innov=rng.normal(size=(2000,60))
for name,v,acc,start,rho,bias,sigma in [('constant',0,0,0,0,0,.3),('velocity',-1,0,0,.4,0,.3),('acceleration',-1,1,-6,.4,0,.3),('late onset',-1,1,-.3,.4,0,.3),('unseen acceleration',-1,2,.1,.4,0,.3),('unseen braking',-1,-2,.1,.4,0,.3),('biased',-1,0,0,.85,.5,.3),('low noise',-1,0,0,.4,0,.1),('high noise',-1,0,0,.4,0,.6)]:
    r=25+v*u+.5*acc*np.maximum(0,u-start)**2
    err=innov.copy()
    for j in range(1,60): err[:,j]=rho*err[:,j-1]+np.sqrt(1-rho*rho)*innov[:,j]
    st=replay(r[:60]+bias+sigma*err)
    for m in ('one','pooled','matched','equal','linear','quadratic'):
        p=combine(st,.001,mode=m)[0] if m in ('one','pooled','matched') else st[2]+{'equal':.5,'linear':0,'quadratic':1}[m]*st[3]
        record('stress',scenario=name,method=m,rmse=rmse(p-r[60:]))
for table,rows in records.items():
    pd.DataFrame(rows).to_csv(supp/(table+'.csv'),index=False)
