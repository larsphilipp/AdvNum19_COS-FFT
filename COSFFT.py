# COS-FFT for Heston Model

import numpy as np
from scipy.special import erf

## Input Parameters
r     = 0.1        # risk-free rate
mu    = r          # model parameters
sigma = 0.3    
S0    = 100        # Today's stock price
tau     = 0.25       # Time to expiry in years
q     = 0

#K=70:130
C_BS,p,d1,d2 = blackS(S0,K,r,tau,sigma,q)


## Step 1. Setup --------------------------------------------------------------
scalea = -10 # how many standard deviations?
scaleb = 10 
a      = scalea*np.sqrt(tau)*sigma # lowerBound
b      = scaleb*np.sqrt(tau)*sigma
bma    = b-a
N      = 50
k      = np.arange(0, N, dtype=np.float)
u      = k * np.pi / bma
K      = np.arange(70, 131, dtype=np.float)
a      = 0.15  # Initial vola of underyling at time 0 (also called u0)
bj     = 0.5   # The speed of mean reversion also called lambda b1 =kappa+lam-rho*sigma
v_bar  = 0.05 # mean level of variance
uj     = 0.5 # in the original paper it is 0.5 and -0.5 -> *2 equals 1, so may be not relevant (not included in Fang papr)
volvol = 0.05 # Volatility of the volatiltiy process (if 0 then constant Vol like BS)
rho = 0.2   #covariance between the log stock and the variance process

## Step 2: Prepare terms ------------------------------------------------------
# Value for a Call
Vk = 2 / bma * (cosSerExp(a,b,0,b,k) - cosSer1(a,b,0,b,k))

# Integrated Function
u = k * np.pi / bma
#characteristicFunction = charFuncBSM(u, mu, sigma, T)
characteristicFunction = charFuncHeston(mu, u, tau, a, bj, v_bar, uj, rho, volvol)

# Assign Space for Call Prices
C_COS = np.zeros((np.size(K)))

## Step 3: Calculate prices ---------------------------------------------------
for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * u * (x-a)) # not needed unter Heston (is already included) could be moved out (see Fang p.22)
    Fk = np.real(np.multiply(characteristicFunction, addIntegratedTerm)) 
    Fk[0]=0.5*Fk[0]						# weigh first term 1/2
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,Vk)) * np.exp(-r*T)
    
#### --------------------------------------------------------------------------
## ========== Functions of the model ==========
def charFuncBSM(s,mu,sigma, tau):
    # phi = E[exp(ius)]
    # Characteristic Function for the Black Scholes Model
    phi = np.exp((mu - 0.5 * np.power(sigma,2)) * 1j * np.multiply(tau,s) - 0.5 * np.power(sigma,2) * tau * np.power(s,2))  #vector-compatible in s
    return phi

def charFuncHeston(r, u, tau, a, bj, v, uj, rho, sigma):
    d = np.sqrt(np.power(rho * sigma * u * 1j, 2) - np.power(sigma,2) * (2*uj*u*1j - np.power(u,2)))
    g = (bj - rho*sigma*u*1j + d) / (bj - rho*sigma*u*1j - d)
    C = r * u * 1j * tau + a/np.power(sigma,2) * ( (bj - rho * sigma * 1j + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1-g) ))
    D = (bj - rho * sigma * u * 1j + d) / np.power(sigma,2) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    phi = np.exp(C + D * v + 1j * u)
    return phi

def blackS(S,X,r,T,sigma,q):
    #Calculates Black-Scholes european option prices.
    #  Peter.Gruber@unisg.ch, February 2007
    #  Based on code by Paul.Soderlind@unisg.ch
    #if args==6:     # if dividend is specified, correct for it
    S = S * np.exp(-q*T)
    
    d1 = np.divide( ( np.log(np.divide(S,X) ) + (r+1/2* np.power(sigma,2))*T ), (sigma*np.sqrt(T)) )
    d2 = d1 - sigma*np.sqrt(T)
    c  = np.multiply(S, stdnCdf(d1)) - np.multiply(np.multiply(X, np.exp(-r*T)), stdnCdf(d2))
    p  = c + np.multiply(X, np.exp(-r*T)) - S                  #put-call parity
    
    return c,p,d1,d2

def stdnCdf(a):
    cdf = 0.5 + 0.5 * erf(a / np.sqrt(2))
    return cdf

def ftcall(model, charfn, data):
    alpha = model.FFT.alpha
    r = data.r
    tau = data.tau
    nu = model.FFT.nu
    phi = np.divide( np.multiply(np.exp(-r*tau), charfn), np.multiply((alpha + 1j*nu), (alpha + 1 + 1j*nu)))
    return phi

def cosSerExp(a,b,c,d,k):
      # cosine series coefficients of exp(y) Oosterle (22)
      # INPUT     a,b ... 1x1     arguments in the cosine
      #           c,d ... 1x1     integration boundaries in (20)
      #           k   ... FFT.Nx1 values of k
    bma = b-a
    uu  = k*np.pi/bma
    chi = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu*(d-a)))*np.exp(d)-np.multiply(uu,np.sin(uu*(c-a)))*np.exp(c)))
    return chi

def cosSer1(a,b,c,d,k):
  # cosine series coefficients of 1 Oosterle (23)
  # INPUT     a,b ... 1x1     arguments in the cosine
  #           c,d ... 1x1     integration boundaries in (20)
  #           k   ... FFT.Nx1 values of k
    bma    = b-a
    uu  = k*np.pi/bma
    uu[0]  = 1      # to avoid case differentiation (done 2 lines below)
    psi    = np.divide(1,uu) * ( np.sin(uu*(d-a)) - np.sin(uu*(c-a)) )
    psi[0] = d-c
    return psi
  

