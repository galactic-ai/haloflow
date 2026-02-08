import numpy as np

from scipy.integrate import cumulative_trapezoid

from functools import partial

# Schechter function: phi(M) = phi_M * (M/M_star)**alpha * np.exp(-M/M_star)

def schechter_logmass(log_m, log_mstar=12.27454237366621, alpha=-2.0143401113429573, phi_star=4.989766469669057e-05):
    # phi(log_m) = phi_star * ln(10) * 10**(log_m - log_mstar)**(alpha+1) * e**-10**(log_m - log_mstar)
    # delta = log_m - log_mstar; x = 10**delta; phi(log_m) = phi_star * ln(10) * x**(alpha+1) * e**-x
    delta = log_m - log_mstar
    x = 10**delta
    term1 = phi_star * np.log(10)
    term2 = x**(alpha+1)
    term3 = np.exp(-x)
    return term1 * term2 * term3


def double_schechter_logmass(log_m, log_mstar, alpha1, phi1, alpha2, phi2):
    # phi(log_m) = ln(10) * e**-10**(log_m - log_mstar) * [phi1 * 10**((log_m - log_mstar)*(alpha1+1)) + phi2 * 10**((log_m - log_mstar)*(alpha2+1))]
    delta = log_m - log_mstar
    x = 10**delta
    term1 = np.log(10) * np.exp(-x)
    term2 = phi1 * x**(alpha1 + 1) + phi2 * x**(alpha2 + 1)
    return term1 * term2

def mu_counts(params, mh_bin, lo, hi, ngrid=8000, type='single_schechter'):
    if type == 'single_schechter':
        phi1, logMstar, a1 = params
        func = partial(schechter_logmass, log_mstar=logMstar, alpha=a1, phi_star=phi1)
    elif type == 'double_schechter':
        phi1, logMstar, a1, phi2, a2 = params
        func = partial(double_schechter_logmass, log_mstar=logMstar, alpha1=a1, phi1=phi1, alpha2=a2, phi2=phi2)
    else:
        raise ValueError("type must be 'single_schechter' or 'double_schechter'")
    
    g = np.linspace(lo, hi, ngrid)
    pdf = func(g)

    C = cumulative_trapezoid(pdf, g, initial=0)
    C_edges = np.interp(mh_bin, g, C)
    mu = np.diff(C_edges)
    return np.maximum(mu, 1e-30)


def nll_counts(params, p_hmf, mh_bin, lo, hi, type='single_schechter'):
    mu = mu_counts(params, mh_bin, lo, hi, type=type)
    y = p_hmf.astype(np.float64)
    return np.sum(mu - y*np.log(mu))           # Poisson NLL up to constant
