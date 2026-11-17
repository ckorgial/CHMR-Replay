"""Frozen causal extension. All prediction functions receive observed histories only."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from models import bank, kalman


def cross_moment(variance, rho, width=15):
    lags = (width-1-np.arange(width))[:, None] + np.arange(1, 11)
    contrast = bank(width, 2) - bank(width, 1)
    return variance[:, None] * (rho[:, None, None]**lags * contrast).sum(axis=1)


def noise_estimate(y, shrink):
    t = np.linspace(-1, 1, y.shape[1])
    design = np.vander(t, 3)
    residual = y - (y @ np.linalg.pinv(design).T) @ design.T
    energy = (residual[:, :-1]**2).sum(1)
    rho = np.divide((residual[:, :-1]*residual[:, 1:]).sum(1), energy,
                    out=np.zeros_like(energy), where=energy>1e-12)
    return (residual**2).sum(1)/(y.shape[1]-3), shrink*np.clip(rho, -.95, .95)


def forecast(y, s, prior, cfg):
    """No labels, regime IDs, future measurements or oracle parameters are accepted."""
    width, count = 15, 20
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] < width+10+count-1 or not np.isfinite(y).all():
        raise ValueError("Insufficient finite replay history")
    windows = sliding_window_view(y, width, axis=1)
    origins = np.arange(y.shape[1]-10-count, y.shape[1]-10)
    idx = origins-width+1
    linear = windows @ bank(width, 1)
    diff = windows @ (bank(width, 2)-bank(width, 1))
    target = y[:, origins[:, None]+np.arange(1, 11)]
    d = diff[:, idx]
    e = target-linear[:, idx]
    a, b = (d*d).mean(1), (d*e).mean(1)
    ridge = s['lambda']
    def weight(a, b):
        return np.clip((b+.5*ridge)/(a+ridge), 0, 1)
    def poly(w):
        return linear[:, -1]+w*diff[:, -1]
    ca = kalman(y, 3, s.qca, s.sigma)
    ca_replay = np.stack([kalman(y[:, :o+1], 3, s.qca, s.sigma) for o in origins], axis=1)
    # Historical fixed FIR anchor avoids scoring hindsight-fitted replay weights.
    anchor = linear[:, idx]+.5*d
    delta = ca_replay-anchor
    kappa = cfg['gate_prior_strength_m2']
    gate = np.clip(((delta*(target-anchor)).mean(1)+kappa*prior)/
                   ((delta**2).mean(1)+kappa), 0, 1)
    correction = cross_moment(*noise_estimate(y, cfg['ar_shrink']))
    p = {'CA':ca, 'AR-CA':kalman(y,3,s.qarca,s.sigma,s.rho),
         'CV':kalman(y,2,s.qcv,s.sigma),'AR-CV':kalman(y,2,s.qarcv,s.sigma,s.rho),
         'Linear':poly(0), 'Quadratic':poly(1), 'Equal':poly(.5),
         'One-step':poly(weight(a[:,:1],b[:,:1])), 'Hard pool':poly(weight(a.mean(1)[:,None], b.mean(1)[:,None]))}
    for corrected in (False,True):
        score = b-correction if corrected else b
        pooled = weight(a.mean(1, keepdims=True), score.mean(1, keepdims=True))
        for shared in (False,True):
            tau = cfg['pool_penalty_m2'] if shared else 0
            w = np.clip((score+.5*ridge+tau*pooled)/(a+ridge+tau),0,1)
            base = poly(w)
            suffix = ('P' if shared else '')+('C' if corrected else '')
            p[suffix or 'CHMR'] = base
            p[suffix+'G'] = base+gate*(ca-base)
    p['Fixed mix'] = p['CHMR']+prior*(ca-p['CHMR'])
    p['Half mix'] = (p['CHMR']+ca)/2
    return p, gate
