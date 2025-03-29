import numpy as np

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

