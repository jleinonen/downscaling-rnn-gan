import numpy as np


def crps_ensemble(observation, forecasts):
    fc = forecasts.copy()
    fc.sort(axis=-1)
    obs = observation
    fc_below = fc<obs[...,None]
    crps = np.zeros_like(obs)

    for i in range(fc.shape[-1]):
        below = fc_below[...,i]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[...,i][below])

    for i in range(fc.shape[-1]-1,-1,-1):
        above  = ~fc_below[...,i]
        k = fc.shape[-1]-1-i
        weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
        crps[above] += weight * (fc[...,i][above]-obs[above])

    return crps
