"""Independent polynomial, replay, covariance integral, and edge-case tests."""
import numpy as np
from scipy.integrate import quad
from math import factorial
from models import bank,replay,combine,dynamics,imm,kalman
import pandas as pd
from pathlib import Path
rng=np.random.default_rng(104)
for p in (1,2):
    a=bank(15,p);x=np.arange(-14,1)*.1
    X=np.vander(x,p+1,increasing=True);Q,R=np.linalg.qr(X)
    expected=Q@np.linalg.solve(R.T,np.vander(np.arange(1,11)*.1,p+1,increasing=True).T)
    np.testing.assert_allclose(a,expected,atol=1e-13)
    np.testing.assert_allclose((a*a).sum(0)[[0,9]],([.295,1.099] if p==1 else [.793,18.814]),atol=.0005)
l=np.polyfit(x,x*x,1);q=x*x-np.polyval(l,x)
t=np.arange(1,11)*.1
np.testing.assert_allclose(bank(15,2)-bank(15,1),np.outer(q/(q@q),t*t-np.polyval(l,t)),atol=1e-13)
try: bank(2,2)
except ValueError: pass
else: raise AssertionError('rank-deficient design')
for a,b,lam,expected in [(0,0,.1,.5),(1e-18,0,1e-18,.25),(1,10,.1,1),(1,-2,.1,0)]:
    result=combine(tuple(np.array([[float(v)]]) for v in [a,b,0,1]),lam)[1].item()
    assert abs(result-expected)<1e-12
Y=rng.normal(size=(3,60))*1e8
assert np.isfinite(combine(replay(Y))[0]).all()
for order in (2,3):
    F,Q=dynamics(order,2)
    for i,j in np.ndindex(Q.shape):
        powers=[order-i-1,order-j-1]
        exact=quad(lambda u:2*u**sum(powers)/np.prod([factorial(k) for k in powers]),0,.1)[0]
        np.testing.assert_allclose(Q[i,j],exact)
hist=rng.normal(20,.4,(5,60))
for d in (2,3):np.testing.assert_allclose(kalman(hist,d,.1,.4),kalman(hist,d,.1,.4,0),atol=1e-10)
assert imm(hist,[.1,.1],.4,.95).shape==(5,10)
# If a benchmark was already executed, validate its output schema and finiteness.
for table, key in [('performance', 'method'), ('ablations', 'variant')]:
    path = Path('results') / f'{table}.csv'
    if path.exists():
        frame = pd.read_csv(path)
        required = {'suite', key, 'rmse'}
        if not required.issubset(frame.columns):
            raise AssertionError(f'Missing columns in {path}: {required - set(frame.columns)}')
        if frame.empty or not np.isfinite(frame['rmse']).all():
            raise AssertionError(f'Empty or nonfinite values in {path}')
        if frame.duplicated(['suite', key]).any():
            raise AssertionError(f'Duplicate rows in {path}')
print('QR, residualization, gains, clipping, tiny ridge, rank guard, extreme noise, '
      'process covariance, white-noise augmentation and output-schema checks passed.')

# Independent report-level checks, including the conditional curved-motion shift.
from scipy.linalg import qr
from numpy.linalg import solve
from math import factorial
x=np.arange(-14,1)*.1
for order in (1,2):
    X=np.vander(x,order+1,increasing=True)
    Q,R=qr(X,mode='economic')
    future=np.vander(np.arange(1,11)*.1,order+1,increasing=True)
    np.testing.assert_allclose(bank(15,order),Q@solve(R.T,future.T),atol=2e-13)
for order in (2,3):
    _,cov=dynamics(order,2.)
    for i,j in np.ndindex(cov.shape):
        mi,mj=order-i-1,order-j-1
        computed=quad(lambda s: 2*s**(mi+mj)/(factorial(mi)*factorial(mj)),0,.1)[0]
        np.testing.assert_allclose(cov[i,j],computed,atol=2e-14)
# Use separate random stream and avoid treating MC sampling error as floating-point error.
mc_rng=np.random.default_rng(1517)
W,h,rho,sigma=15,3,.7,.4
cov=sigma**2*rho**abs(np.subtract.outer(np.arange(W+h),np.arange(W+h)))
a=bank(W,1)[:,h-1];d=bank(W,2)[:,h-1]-a
motion=16-np.arange(W+h)*.1+.5*(np.arange(W+h)*.1)**2
noise=mc_rng.multivariate_normal(np.zeros(W+h),cov,size=100000)
D=(motion[:W]+noise[:,:W])@d
analytic=(d@cov[:W,-1])/((d@motion[:W])**2+d@cov[:W,:W]@d)
measured=np.mean(D*noise[:,-1])/np.mean(D**2)
assert abs(analytic-measured)<.008,(analytic,measured)
print(f'Curved-motion conditional shift: analytic={analytic:.8f}, MC={measured:.8f}')
# Property: horizon-matched weights cannot use target ages after the loss origin.
for W,K in [(15,20),(20,30),(25,20)]:
    origins=np.arange(59-10-K+1,59-10+1)
    assert origins[0]>=W-1
    assert np.all(origins[:,None]+np.arange(1,11)[None,:]<=59)
# Independent mathematical proof that late horizons dominate pooled curvature.
x=np.arange(-14,1)*.1
lin=np.polyval(np.polyfit(x,x*x,1),np.arange(1,11)*.1)
curv=(np.arange(1,11)*.1)**2-lin
fraction=np.sum(curv[-3:]**2)/np.sum(curv**2)
assert abs(fraction-.64956)<1e-4, fraction
print(f'Late-horizon curvature contribution: {fraction:.5%} (ages 8-10)')

# Explicit argument-guard tests for scientifically invalid configurations.
for fn,part in [
    (lambda: replay(np.ones((1,60)),W=15,K=40),'too many common origins'),
    (lambda: replay(np.ones((1,60)),beta=-.5),'invalid recency factor'),
    (lambda: combine(replay(np.ones((1,60))),lam=-1),'negative ridge'),
    (lambda: bank(2,2),'rank-deficient polynomial bank'),
]:
    try: fn()
    except ValueError: pass
    else: raise AssertionError(f'Expected to reject {part}')
print('Invalid-parameter guards passed.')
