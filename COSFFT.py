# COS-FFT for Heston Model

import numpy as np
from scipy.special import erf

## Input Parameters
r     = 0.1        # risk-free rate
mu    = r          # model parameters
sigma = 0.3    
S0    = 100        # Today's stock price
T     = 0.25       # Time to expiry in years
q     = 0

#K=70:130
C_BS,p,d1,d2 = blackS(S0,K,r,T,sigma,q)
#subplot(2,1,1)
#plot(K,C_BS,'r')
#hold on

## Step 1. Setup
scalea = -10 # how many standard deviations?
scaleb = 10 
a      = scalea*np.sqrt(T)*sigma
b      = scaleb*np.sqrt(T)*sigma
bma    = b-a
N      = 50
k      = np.arange(0, N, dtype=np.float)
gamma  = k*np.pi/bma
K      = np.arange(70, 131, dtype=np.float)

## Step 2: Prepare Uk terms
Uk = 2/bma * ( cosSerExp(a,b,0,b,k) - cosSer1(a,b,0,b,k) )
charfn = cf(gamma, mu, sigma, T) # phi

C_COS = np.zeros((np.size(K)))

## Step 3: Calculate prices
for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    Fk = np.real(np.multiply(charfn, np.exp(1j*k*np.pi*(x-a)/bma)))
    Fk[0]=0.5*Fk[0]						# weigh first term 1/2
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,Uk)) * np.exp(-r*T)

#subplot(2,1,1)
#hold on
#plot(K,C_COS,'k:')
#subplot(2,1,2)
#hold on
#semilogy(K,abs(C_BS-C_COS)./C_BS,'k--')
#axis([S0*.7 S0*1.3 1E-16 1])


## ========== Private functions of the model ==========
def cf(s,mu,sigma, T):
    # phi = E[exp(ius)]
    # In the BS-Case, this is
    phi = np.exp((mu - 0.5 * np.power(sigma,2)) * 1j * np.multiply(T,s) - 0.5 * np.power(sigma,2) * T * np.power(s,2))  #vector-compatible in s
    return phi


def blackS(S,X,r,T,sigma,q):
    #Calculates Black-Scholes european option prices.
    #
    #  Usage:      [c,p,d1,d2] = blackS( S,X,r,T,sigma,[q] )
    #
    #  Inputs:     S      scalar or nx1 vector, possible current stock prices
    #              X      scalar or nx1 vector, strike price
    #              r      scalar, riskfree interest rate (continuously compounded)
    #              T      scalar, time to expiry of option
    #              sigma  scalar or nx1 vector, std in stock price evolution
    #              [q]    scalar, dividend yield (continuously compounded), optional
    #
    #  Output:     c      nx1 vector, call option prices
    #              p      nx1 vector, put option prices
    #              d1     nx1 vector
    #              d2     nx1 vector
    #
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
  

