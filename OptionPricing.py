## Title:        Option Pricing
## Author:       Elisa FLeissner, Lars Stauffenegger
## Email:        elisa.fleissner@student.unisg.ch,
##               lars.stauffenegger@student.unisg.ch,
## Place, Time:  Zürich, 24.03.19
## Description:  Option Pricing with Black Scholes, COS Method and Heston Model
## Improvements: -
## Last changes: -


# In[1]: Packages, Settings
import quandl
import numpy as np
import pandas as pd
import AllFunctions as func
import matplotlib.pyplot as plt

## TODO: quandl price import, Hardcoded for 1 security


# In[2]: Import data from quandl
quandl.ApiConfig.api_key = "mrMTRoAdPycJSyzyjxPN"
ticker = "AAPL"
database = "EOD"
ID = database + "/" + ticker
stock_data = quandl.get(ID, rows = 500)
log_ret = np.log(stock_data.Close) - np.log(stock_data.Close.shift(1))
log_ret.drop(log_ret.index[:1], inplace = True)

mean_data = np.mean(log_ret)
var_data = np.var(log_ret)
sd_data = np.sqrt(var_data)
stock_today = stock_data.Close.tail(1)


# In[2]: Parameter
# According to Fang, 2010, p. 30
r       = 0         # Risk-free rate
mu      = mean_data         # Mean rate of drift
sigma   = sd_data       # Initial Vola of underyling at time 0; also called u0 or a
S0      = int(stock_today)       # Today's stock price
tau     = 30 / 365  # Time to expiry in years
q       = 0         # Divindend Yield
lm      = 1.5768    # The speed of mean reversion
v_bar   = var_data    # Mean level of variance of the underlying
volvol  = 0.5751    # Volatility of the volatiltiy process (if 0 then constant Vol like BS)
rho     = -0.5711   # Covariance between the log stock and the variance process

# Range of Strikes
mini    = int(stock_today * 0.8)
maxi    = int(stock_today * 1.2)
K       = np.arange(mini, maxi, dtype = np.float)

# Truncation Range
L       = 120
a, b    = func.truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol)
bma     = b-a

# Number of Points
N       = 15
k       = np.arange(np.power(2,N))

# Input for the Characterstic Function Phi
u       = k * np.pi/bma


# In[3]: Black Scholes Option Pricing
C_BS, p, d1, d2 = func.blackScholes(S0, K, r, tau, sigma, q)
print (C_BS)


# In[4]: COS-FFT Value Function for Put
UkPut = 2 / bma * ( func.cosSer1(a,b,a,0,k) - func.cosSerExp(a,b,a,0,k) )
UkCall = 2 / bma * ( func.cosSerExp(a,b,0,b,k) - func.cosSer1(a,b,0,b,k) )


# In[5]: COS with BS-Characterstic Function
charactersticFunctionBS = func.charFuncBSM(u, mu, sigma, tau)

C_COS = np.zeros((np.size(K)))

for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(np.multiply(charactersticFunctionBS, addIntegratedTerm))
    Fk[0]=0.5 * Fk[0] 
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,UkCall)) * np.exp(-r * tau)
    
print (C_COS)


# In[6]: COS with Fang & Oosterlee (2008) Version of Heston's Characteristic Function
charactersticFunctionFOH = func.charFuncHestonFO(mu, r, u, tau, sigma, v_bar, lm, rho, volvol)

C_COS_HFO = np.zeros((np.size(K)))
P_COS_HFO = np.zeros((np.size(K)))
C_COS_PCP = np.zeros((np.size(K)))

for m in range(0, np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(charactersticFunctionFOH * addIntegratedTerm)
    Fk[0] = 0.5 * Fk[0]						
    C_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkCall)) * np.exp(-r * tau)
    P_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau)
    C_COS_PCP[m] = P_COS_HFO[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau)

print(C_COS_HFO)
print(P_COS_HFO)
print(C_COS_PCP)


# In[7]: Plotting
plt.plot(K, C_BS, "g.", K, C_COS, "b.", K, C_COS_HFO, "r.")
plt.axvline(x = S0)
plt.show()
print("C_BS = green, C_COS = blue, C_COS_HFO = red")

## End