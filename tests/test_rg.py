import unittest,json
from pathlib import Path
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.optimize import minimize
from models import synthetic,bank,combine,replay
from rg import forecast,cross_moment,noise_estimate
from rg_controls import smooth
class TestRG(unittest.TestCase):
    def test_ar(self):
        rho=.65;var=.4**2; rng=np.random.default_rng(812)
        cov=var*rho**abs(np.subtract.outer(np.arange(25),np.arange(25)))
        e=rng.multivariate_normal(np.zeros(25),cov,100000)
        d=bank(15,2)-bank(15,1)
        v=(e[:,:15]@d)*e[:,15:]
        np.testing.assert_array_less(abs(v.mean(0)-cross_moment(np.array([var]),np.array([rho]))[0]),5*v.std(0)/np.sqrt(len(e)))
        np.testing.assert_array_equal(cross_moment(np.array([var]),np.array([0.])),0)
    def test_smooth(self):
        a=np.linspace(.1,1,10);b=np.sin(np.arange(10))
        w=smooth(a[None],b[None],.2)[0]
        loss=lambda x:np.sum(a*x*x-2*b*x+.2*(x-.5)**2)+.1*np.sum(np.diff(x)**2)
        v=minimize(loss,np.full(10,.5),bounds=[(0,1)]*10,tol=1e-12)
        self.assertAlmostEqual(loss(w),v.fun,places=9)
    def test_prefix(self):
        y,r,*_=synthetic('PA',10,99)
        s=pd.Series({'lambda':.1,'qca':1,'sigma':.3,'qarca':1,'rho':.1,'qcv':1,'qarcv':1})
        cfg=json.loads(Path('protocol.json').read_text())
        p,g=forecast(y,s,.9,cfg)
        np.testing.assert_allclose(p['CHMR'],combine(replay(y),.1)[0],atol=1e-12)
        self.assertTrue(((g>=0)&(g<=1)).all())
        fut=np.column_stack([y,r[:,60:]]); fut[:,60:]=np.nan
        np.testing.assert_array_equal(forecast(fut[:,:60],s,.9,cfg)[0]['PCG'],p['PCG'])
        np.testing.assert_allclose(noise_estimate(np.ones((1,60)),.5)[1],0)
if __name__=='__main__':unittest.main()
