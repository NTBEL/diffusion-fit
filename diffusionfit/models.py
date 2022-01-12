"""
"""

import numpy as np
import numba

@numba.njit(cache=True)
def gaussian(r, E, gamma):
    """Gaussian diffusion function.
    """
    return E * np.exp(-(r/gamma)**2)

@numba.njit(cache=True)
def anisotropic_gaussian(x, y, E, gamma_x, gamma_y):
    """Anisotropic Gaussian diffusion function.
    """
    return E * np.exp(-(x/gamma_x)**2) * np.exp(-(y/gamma_y)**2)

@numba.njit(cache=True)
def point_clark(r, Emax, beta, gamma):
    """Point-Clark diffusion distribution function for receptor-based sensors.
    """
    return Emax / (1 + beta*np.exp((r/gamma)**2))

@numba.njit(cache=True)
def point_source(t, r, D, Q=1e8, alpha=1):
    """Concentration for diffusion from an instantaneous point source.
    """
    a = (4*np.pi*D*t)**(3/2)
    b = np.exp(-r**2/(4*D*t))
    return (Q/alpha) * b / a

@numba.njit(cache=True)
def point_source_loss(t, r, D, Q=1e8, alpha=1, kprime=0):
    """Concentration for diffusion from an instantaneous point source.
    """
    a = (4*np.pi*D*t)**(3/2)
    b = np.exp(-r**2/(4*D*t))
    c = np.exp(-kprime*t)
    return (Q/alpha) * b * c / a

@numba.njit(cache=True)
def log_intensity_withloss(t, k, B, t0):
    return -k*(t+t0) - np.log(t+t0) + B

@numba.njit(cache=True)
def log_intensity_noloss(t, B, t0):
    return - np.log(t+t0) + B
