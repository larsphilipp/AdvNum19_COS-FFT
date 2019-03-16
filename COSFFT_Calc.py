# ============================== INFORMATION ==============================

# Title:       COS-FFT for Heston Model
# Authors:
# Date:


# ============================== PREPARATION ==============================

# Import libraries
import numpy as np
from scipy import special
from scipy.special import erf
from COSFFT_Functions import *

# Input Parameters
r     = 0.1        # risk-free rate
mu    = r          # model parameters
sigma = 0.3    
S0    = 100        # Today's stock price
tau   = 0.25       # Time to expiry in years
q     = 0
K     = np.arange(70, 131, dtype = np.float)

# Calculate Black Scholes
C_BS, p, d1, d2 = blackS(S0, K, r, tau, sigma, q)


# ============================= STEP 1: SETUP =============================

# Parameters
scalea = -10   # how many standard deviations?
scaleb = 10 
a      = scalea * np.sqrt(tau) * sigma # lowerBound
b      = scaleb * np.sqrt(tau) * sigma
bma    = b-a
N      = 50
k      = np.arange(0, N, dtype = np.float)
u      = k * np.pi / bma
a      = 0.15  # Initial vola of underyling at time 0 (also called u0)
bj     = 0.5   # The speed of mean reversion also called lambda b1 =kappa+lam-rho*sigma
v_bar  = 0.05  # mean level of variance
uj     = 0.5   # in the original paper it is 0.5 and -0.5 -> *2 equals 1, so may be not relevant (not included in Fang papr)
volvol = 0.05  # Volatility of the volatiltiy process (if 0 then constant Vol like BS)
rho    = 0.2   #covariance between the log stock and the variance process


# ========================= STEP 2: PREPARE TERMS =========================

# Value for a Call
Vk = 2 / bma * (cosSerExp(a, b, 0, b, k) - cosSer1(a, b, 0, b, k))

# Integrated Function
#u = k * np.pi / bma
characteristicFunction = charFuncBSM(u, mu, sigma, tau)
#characteristicFunction = charFuncHeston(mu, u, tau, a, bj, v_bar, uj, rho, volvol)

# Assign space for Call Prices
C_COS = np.zeros((np.size(K)))


# ========================= STEP 3: CALCULATE PRICES =========================

for m in range(0, np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j*k*np.pi*(x-a)/bma) # not needed unter Heston (is already included) could be moved out (see Fang p.22)
    Fk = np.real(np.multiply(characteristicFunction, addIntegratedTerm)) 
    Fk[0] = 0.5 * Fk[0]						# weigh first term 1/2
    C_COS[m] = K[m] * np.sum(np.multiply(Fk, Vk)) * np.exp(-r * tau)
    
print (C_COS)
print (C_BS)