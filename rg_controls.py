"""Secondary controls fixed after the primary protocol. No change to the primary estimator."""
from pathlib import Path
import json, os, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from models import synthetic,replay,combine,bank
from rg import forecast,noise_estimate,cross_moment
from kitti import extract

def smooth(a,b,lam,tau=.1):
    """Cyclic coordinate descent for box-constrained tridiagonal convex quadratic."""
    w=np.clip((b+lam/2)/(a+lam),0,1)
    degree=np.array([1]+[2]*8+[1])
    for _ in range(3000):
        old=w.copy()
        for h in range(10):
            neighbors=(w[:,h-1] if h else 0)+(w[:,h+1] if h<9 else 0)
            w[:,h]=np.clip((b[:,h]+lam/2+tau*neighbors)/(a[:,h]+lam+tau*degree[h]),0,1)
        if np.max(abs(old-w))<1e-10:return w
    raise RuntimeError('Adjacent-horizon quadratic failed to converge')

def main(cache):
    cfg=json.loads(Path('protocol.json').read_text());rows=[];errors=[]
    settings=pd.read_csv('results/settings.csv').set_index('suite')
    for i,suite in enumerate(settings.index):
        s=settings.loc[suite];lam=s['lambda']; real=suite=='KITTI'
        if real:
            os.environ['CHMR_OFFLINE']='1'
            y,r,seq=extract(cache,'rg_results');dev=seq%3==0
            yd,rd=y[dev],r[dev];test=[('original',y[~dev],r[~dev],None)]
        else:
            yd,rd,*_=synthetic(suite,500,100+i)
            test=[]
            for label,seeds,n in [('original',[200+i],2000),('confirmation',[64001+i,74001+i],1000)]:
                batches=[synthetic(suite,n,j,return_scale=True) for j in seeds]
                y,r,_,_,sc=[np.concatenate(z) for z in zip(*batches)]
                test.append((label,y,r,sc))
        sd=replay(yd);dev_p,dev_w=combine(sd,lam);pool=combine(sd,lam,mode='pooled')[0]
        def slope(x,y):
            return np.clip(np.sum(x*y)/np.sum(x*x),0,1) if np.any(x) else 1.
        gamma=slope(dev_p-pool,rd[:,60:]-pool)
        fixed=np.clip(np.sum(sd[3]*(rd[:,60:]-sd[2]),axis=0)/np.sum(sd[3]**2,axis=0),0,1)
        from models import kalman
        pi=slope(kalman(yd,3,s.qca,s.sigma)-dev_p,rd[:,60:]-dev_p)
        if real:
            y,r,seq=extract(cache,'rg_results',gap=True);m=seq%3!=0
            test.append(('unmatched first frame',y[m],r[m],None))
        for label,y,r,sc in test:
            a,b,f,d=replay(y); w=np.clip((b+lam/2)/(a+lam),0,1)
            wp=combine((a,b,f,d),lam,mode='pooled')[1]
            weights={'Adjacent':smooth(a,b,lam),'Fixed horizon':fixed,'Global partial':wp+gamma*(w-wp)}
            c=cross_moment(*noise_estimate(y,.5));p=np.clip((b.mean(1,keepdims=True)-c.mean(1,keepdims=True)+lam/2)/(a.mean(1,keepdims=True)+lam),0,1)
            weights.update({'CHMR':w,'PC':np.clip((b-c+lam/2+.1*p)/(a+lam+.1),0,1)})
            if sc is not None:
                rho=.4 if suite=='MR' else 0.
                oracle=cross_moment(sc**2,np.full(len(y),rho))
                weights['Known covariance']=np.clip((b-oracle+lam/2)/(a+lam),0,1)
                A,B,_,_=replay(r[:,:60]);D=bank(15,2)-bank(15,1);L=bank(15,1)
                cov=rho**abs(np.subtract.outer(np.arange(15),np.arange(15)))
                opt=(B-sc[:,None]**2*(D*(cov@L)).sum(0))/(A+sc[:,None]**2*(D*(cov@D)).sum(0))
                for name,v in weights.items():
                    for j,x in enumerate(((v-np.clip(opt,0,1))**2).mean(0)):errors.append((suite,label,name,j+1,x))
            preds={name:f+d*v for name,v in weights.items()}
            if label.startswith('unmatched'):preds.update(forecast(y,s,pi,cfg)[0])
            for name,v in preds.items():rows.append((suite,label,name,len(y),np.sqrt(np.mean((v-r[:,60:])**2))))
    pd.DataFrame(rows,columns='suite split method n rmse'.split()).to_csv('rg_results/controls.csv',index=False)
    pd.DataFrame(errors,columns='suite split method h weight_mse'.split()).to_csv('rg_results/weights.csv',index=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='cache');main(ap.parse_args().cache)
