import numpy as np
from scipy.optimize import curve_fit

import haloflow.data as D

def schechter_logmass(log_m, log_mstar=12.27454237366621, alpha=-2.0143401113429573, phi_star=4.989766469669057e-05):
    # phi(log_m) = phi_star * ln(10) * 10**(log_m - log_mstar)**(alpha+1) * e**-10**(log_m - log_mstar)
    # delta = log_m - log_mstar; x = 10**delta; phi(log_m) = phi_star * ln(10) * x**(alpha+1) * e**-x
    delta = log_m - log_mstar
    x = 10**delta
    term1 = phi_star * np.log(10)
    term2 = x**(alpha+1)
    term3 = np.exp(-x)
    return term1 * term2 * term3

all_sims = ['TNG50', 'TNG100', 'Eagle100', 'Simba100']
sim = None
obs = 'mags'

y, X, domains, counts = [], [], [], []


for i, s in enumerate(all_sims):
    if s == sim:
        continue
    y_t, X_t = D.hf2_centrals("train", obs=obs, sim=s)

    domains.append(np.full(y_t.shape[0], i))
    y.append(y_t)
    X.append(X_t)
    counts.append(np.full(y_t.shape[0], y_t.shape[0]))
    print(y_t.shape[0])

y = np.concatenate(y)
X = np.concatenate(X)
domains = np.concatenate(domains)
counts = np.concatenate(counts)
weights = counts / np.unique(counts).sum()
print(weights)

log_masses = y[:, 0]
print(min(log_masses), max(log_masses))
bins = np.linspace(min(log_masses), max(log_masses), 20)
print(bins)
# mf = schechter_logmass(log_masses, log_mstar=np.median(log_masses), phi_star=1, alpha=-1.5)

# check with histogram
hist, bin_edges = np.histogram(log_masses, bins=bins)
hist = hist / 100**3 / 3.125 / (bin_edges[1] - bin_edges[0])
print(hist.shape, bin_edges.shape)
print(np.median(log_masses))

# fit 
params, _ = curve_fit(schechter_logmass, bin_edges[:-1], hist, p0=[np.median(log_masses), 1, 0.5])
log_mstar, alpha, phi_star = params
print(log_mstar, alpha, phi_star)

import matplotlib.pyplot as plt

# plt.scatter(10**log_masses, mf)
plt.scatter(10**bin_edges[:-1], hist)
plt.plot(10**bin_edges[:-1], schechter_logmass(bin_edges[:-1]))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Mass [M☉]')
plt.ylabel('Φ(log M) [Mpc⁻³ dex⁻¹]')
plt.title('Schechter Mass Function')
plt.grid(True)
plt.show()