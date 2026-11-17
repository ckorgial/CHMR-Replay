import numpy as np
from scipy.linalg import block_diag
from numpy.lib.stride_tricks import sliding_window_view

DT=.1
H=10
def synthetic(suite,n,seed,rho=None,return_scale=False):
    """Sample trajectories conditional on positive range and forward speeds."""
    if suite not in ('MR','PA') or not isinstance(n,(int,np.integer)) or n<=0:
        raise ValueError('synthetic needs suite MR/PA and a positive encounter count')
    if rho is not None and not -1 < rho < 1:
        raise ValueError('AR(1) correlation must be strictly between -1 and 1')
    rng=np.random.default_rng(seed)
    u=(np.arange(70)-59)*DT
    rr=[]; yy=[]; groups=[]; changes=[]; scales=[]
    while len(rr)<n:
        if suite=='MR':
            group=rng.choice(3,p=[.45,.35,.20])
            ego=rng.uniform(*( (20,28) if group!=1 else (10,16) ))
            v=rng.uniform(-2,2)
            a=np.zeros(70)
            if group==0:
                for t in range(1,70): a[t]=.95*a[t-1]+.2*np.sqrt(DT)*rng.normal()
            elif group==1: a[:]=rng.uniform(-1.8,-.6)
            else: a=rng.uniform(-2,2)*np.exp(-np.arange(70)*DT/1.5)
            velocity=v+np.r_[0,np.cumsum(a[:-1])*DT]
            distance=np.r_[0,np.cumsum(velocity[:-1]*DT+.5*a[:-1]*DT**2)]
            r=rng.uniform(12,30)+distance-distance[59]
            sigma=rng.uniform(*[(.35,.55),(.20,.35),(.25,.45)][group])
            b=np.nan
        else:
            group=rng.integers(3); A=(-1.5,0,1.5)[group]
            b=rng.uniform(-3,-.5); v=rng.uniform(-3,1)
            r=rng.uniform(12,20)+v*u+.5*A*(np.maximum(u-b,0)**2-b*b)
            ego=20.;velocity=v+A*np.maximum(u-b,0)
            sigma=.3
        if r.min()<=0 or (ego+velocity).min()<=0: continue
        corr=(.4 if suite=='MR' else 0.) if rho is None else rho
        noise=rng.normal(size=60)*sigma
        for t in range(1,60): noise[t]=corr*noise[t-1]+np.sqrt(1-corr**2)*noise[t]
        rr.append(r);yy.append(r[:60]+noise);groups.append(group);changes.append(b);scales.append(sigma)
    result = np.array(yy),np.array(rr),np.array(groups),np.array(changes)
    return (*result, np.array(scales)) if return_scale else result

def bank(W,p):
    if W <= p or p < 0:
        raise ValueError("Polynomial bank needs at least p + 1 distinct samples")
    x=np.vander(np.arange(1-W,1)*DT,p+1,increasing=True)
    return np.linalg.pinv(x).T@np.vander(np.arange(1,11)*DT,p+1,increasing=True).T

def replay(y,W=15,K=20,beta=1.,fresh=False):
    y=np.asarray(y)
    if y.ndim!=2 or y.shape[1]!=60 or not np.isfinite(y).all():
        raise ValueError('replay needs a finite (encounters, 60) observation array')
    if not isinstance(W,(int,np.integer)) or not isinstance(K,(int,np.integer)) or W<3 or K<1:
        raise ValueError('W and K must be integers with W>=3 and K>=1')
    if not 0<beta<=1:
        raise ValueError('Recency factor beta must satisfy 0 < beta <= 1')
    windows=sliding_window_view(y,W,axis=1)
    f1,f2=[windows@bank(W,p) for p in (1,2)]
    hh=np.arange(1,11)
    origins=(59-hh if fresh else np.full(10,49))[None,:]-np.arange(K-1,-1,-1)[:,None]
    if origins.min()<W-1:
        raise ValueError(f'W={W}, K={K} exceed available complete replay history (max K = {60-10-W+1})')
    idx=origins-W+1
    linear=f1[:,idx,hh-1]; diff=f2[:,idx,hh-1]-linear
    residual=y[:,origins+hh]-linear
    weights=beta**np.arange(K-1,-1,-1);weights/=weights.sum()
    return (np.einsum('nkh,k->nh',diff**2,weights),
            np.einsum('nkh,k->nh',residual*diff,weights),f1[:,-1],f2[:,-1]-f1[:,-1])

def combine(stats,lam=.1,prior=.5,mode='matched'):
    if lam<0 or not 0<=prior<=1 or mode not in ('matched','one','pooled'):
        raise ValueError('Require lambda>=0, prior in [0,1], and a recognized replay mode')
    a,b,base,diff=stats
    if mode=='one': a,b=a[:,:1],b[:,:1]
    if mode=='pooled': a,b=a.mean(1,keepdims=True),b.mean(1,keepdims=True)
    w=np.clip(np.divide(b+lam*prior,a+lam,out=np.full_like(a,prior),where=a+lam>0),0,1)
    return base+w*diff,w

def dynamics(order,q):
    from math import factorial
    F=np.array([[DT**(j-i)/factorial(j-i) if j>=i else 0 for j in range(order)] for i in range(order)])
    Q=np.array([[q*DT**(2*order-i-j-1)/((2*order-i-j-1)*factorial(order-i-1)*factorial(order-j-1)) for j in range(order)] for i in range(order)])
    return F,Q

def kalman(y,order,q,sigma,rho=None):
    if order not in (2,3) or q<0 or sigma<=0 or (rho is not None and not -1<rho<1):
        raise ValueError('Invalid state dimension, process noise, measurement noise, or AR parameter')
    F,Q=dynamics(order,q)
    h=np.eye(order)[0]; R=sigma**2
    P=np.diag([400.,25.,4.][:order])
    if rho is not None:
        F=block_diag(F,rho);Q=block_diag(Q,R*(1-rho**2));P=block_diag(P,R)
        h=np.r_[h,1.];R=0.
    x=np.zeros((len(y),len(h))); x[:,0]=20
    for t in range(y.shape[1]):
        if t: x=x@F.T;P=F@P@F.T+Q
        gain=P@h/(h@P@h+R)
        x+=(y[:,t]-x@h)[:,None]*gain
        P=P-np.outer(gain,h@P);P=(P+P.T)/2
    return np.stack([(x@np.linalg.matrix_power(F,j).T)[:,0] for j in range(1,11)],axis=1)

def imm(y,qs,sigma,stay):
    """Common [range, velocity, acceleration] coordinates; CV discards acceleration."""
    if len(qs)!=2 or min(qs)<0 or sigma<=0 or not 0<stay<1:
        raise ValueError('IMM requires nonnegative [qCV,qCA], sigma>0, and 0<stay<1')
    F0,Q0=dynamics(2,qs[0]); F1,Q1=dynamics(3,qs[1])
    F=np.array([block_diag(F0,0.),F1]);Q=np.array([block_diag(Q0,0.),Q1])
    T=np.array([[stay,1-stay],[1-stay,stay]])
    n=len(y); x=np.zeros((n,2,3));x[:,:,0]=20
    P=np.tile(np.diag([400.,25.,4.]),(n,2,1,1)); P[:,0,2,2]=0
    prob=np.full((n,2),.5);pred=[]
    for t in range(70):
        if t:
            mass=prob@T; mix=prob[:,:,None]*T/mass[:,None,:]
            mean=np.einsum('nij,nid->njd',mix,x)
            diff=x[:,:,None,:]-mean[:,None,:,:]
            cov=np.einsum('nij,nijab->njab',mix,P[:,:,None,:,:]+diff[...,None]*diff[...,None,:])
            x=np.einsum('mij,nmj->nmi',F,mean);P=F@cov@F.transpose(0,2,1)+Q
            prob=mass
        if t<60:
            v=y[:,t,None]-x[:,:,0]; S=P[:,:,0,0]+sigma**2
            gain=P[:,:,:,0]/S[:,:,None];x+=gain*v[:,:,None]
            P=P-gain[...,None]*P[:,:,None,0,:]
            logp=np.log(np.maximum(prob,1e-300))-.5*(np.log(S)+v*v/S)
            prob=np.exp(logp-logp.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        else: pred.append((prob*x[:,:,0]).sum(1))
    return np.stack(pred,axis=1)
