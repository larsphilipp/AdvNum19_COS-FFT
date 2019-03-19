## Title:        All Functions
## Author:       Elisa FLeissner, Lars Stauffenegger
## Email:        elisa.fleissner@student.unisg.ch,
##               lars.stauffenegger@student.unisg.ch,
## Place, Time:  Zürich, 19.03.19
## Description:  All Functions for Option Pricing with Black Scholes, COS Method and Heston Model
## Improvements: -
## Last changes: -

# In[1]: Packages, Settings
import numpy as np
from scipy.special import erf
np.seterr(divide='ignore', invalid='ignore')


# In[2]: Standard Normal Cumulative Distribution Function
def stdnCdf(a):
    cdf = 0.5 + 0.5 * erf(a / np.sqrt(2))
    return cdf


# In[3]: Black Scholes Model
def blackS(S, X, r, T, sigma, q):
    #Calculates Black-Scholes european option prices.
    #  Peter.Gruber@unisg.ch, February 2007
    #  Based on code by Paul.Soderlind@unisg.ch
    #if arg == 6:     # if dividend is specified, correct for it
    S = S * np.exp(-q * T)
    
    d1 = np.divide( ( np.log(np.divide(S, X) ) + (r + 1/2 * np.power(sigma, 2)) * T ), ( sigma * np.sqrt(T)) )
    d2 = d1 - sigma * np.sqrt(T)
    c  = np.multiply(S, stdnCdf(d1)) - np.multiply(np.multiply(X, np.exp(-r*T)), stdnCdf(d2))
    p  = c + np.multiply(X, np.exp(-r*T)) - S                  #put-call parity
    return c,p,d1,d2


# In[4]: Truncation Range     
def truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol):
        c1 = mu * tau + (1 - np.exp(-lm * tau)) * (v_bar - sigma)/(2 * lm) - v_bar * tau / 2

        c2 = 1/(8 * np.power(lm,3)) * (volvol * tau * lm * np.exp(-lm * tau) \
            * (sigma - v_bar) * (8 * lm * rho - 4 * volvol) \
            + lm * rho * volvol * (1 - np.exp(-lm * tau)) * (16 * v_bar - 8 * sigma) \
            + 2 * v_bar * lm * tau * (-4 * lm * rho * volvol + np.power(volvol,2) + 4 * np.power(lm,2)) \
            + np.power(volvol,2) * ((v_bar - 2 * sigma) * np.exp(-2*lm*tau) \
            + v_bar * (6 * np.exp(-lm*tau) - 7) + 2 * sigma) \
            + 8 * np.power(lm,2) * (sigma - v_bar) * (1 - np.exp(-lm*tau)))

        a = c1 - L * np.sqrt(np.abs(c2))
        b = c1 + L * np.sqrt(np.abs(c2))
        return a, b


# In[5]: Cosine Expansion
def cosSerExp(a,b,c,d,k):
    bma = b-a
    uu  = k*np.pi/bma
    chi = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu*(d-a)))*np.exp(d)-np.multiply(uu,np.sin(uu*(c-a)))*np.exp(c)))
    return chi


def cosSer1(a,b,c,d,k):
    bma    = b-a
    uu     = k*np.pi/bma
    uu[0]  = 1      # to avoid case differentiation (done 2 lines below)
    psi    = np.divide(1,uu) * ( np.sin(uu*(d-a)) - np.sin(uu*(c-a)) )
    psi[0] = d-c
    return psi


# In[6]: Characteristic Functions
def charFuncBSM(s,mu,sigma, T):
    # phi = E[exp(ius)]
    # In the BS-Case, this is
    phi = np.exp((mu - 0.5 * np.power(sigma,2)) * 1j * np.multiply(T,s) - 0.5 * np.power(sigma,2) * T * np.power(s,2))  #vector-compatible in s
    return phi

def charFuncHestonFOH(mu, r, u, tau, sigma, v_bar, lm, rho, volvol):
    d = np.sqrt(np.power(lm - 1j * rho * volvol * u, 2) + np.power(volvol,2) * (np.power(u,2) + u * 1j))
    g = (lm - 1j * rho * volvol * u - d) / (lm - 1j * rho * volvol * u + d)
    #C = np.divide(lm * mu, np.power(volvol,2)) * ( (lm - 1j * rho * volvol * u - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    C = np.divide(lm * v_bar, np.power(volvol,2)) * ( (lm - 1j * rho * volvol * u - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    D = 1j * r * u * tau + sigma / np.power(volvol,2) * (np.divide((1 - np.exp(-d * tau)), (1 - g * np.exp(-d * tau)))) * (lm - 1j * rho * volvol * u - d) 
    phi = np.exp(D) * np.exp(C)
    return phi