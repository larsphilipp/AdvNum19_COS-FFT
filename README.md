<div align="right">
Advanced Numerical Methods and Data Analysis - FS19-8,780
<br>
University of St. Gallen, 24.03.2019
<br>
</div>

-------------



# COS-FFT Project Description

**Elisa Fleissner** &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  elisa.fleissner@student.unisg.ch <br>
**Lars Stauffenegger** &nbsp; &nbsp; &nbsp;lars.stauffenegger@student.unisg.ch  <br>

## <div id="0">Overview</div>

1. <a href="#A2">Introduction</a>
2. <a href="#B2">Black Scholes Formula</a>
3. <a href="#C2">Cosine transform method</a>
4. <a href="#D2">Characteristic functions</a>
5. <a href="#E2">Data</a>
6. <a href="#F2">Results</a>
7. <a href="#G2">Concluding remarks</a>
8. <a href="#H2">References</a>


## <div id="A2"> <a href="#0">Introduction  </a> </div>

This is the documentation for the "COS-FFT" assignment of the class **Advanced Numerical Methods and Data Analysis** taught by Prof. Peter Gruber at the University of St. Gallen in Spring 2019. We - Elisa Fleissner and Lars Stauffenegger - are in the 2nd Semester of our Master studies and worked as a group with the aim to use the Cosine transform method as presented in [Fang & Oosterlee (2008)](http://mpra.ub.uni-muenchen.de/9319/) combined with the [Heston model](tbd) to value plain-vanilla European Call options. To validate our results, we implemented the Black Scholes model in our calculations. For all calculations we used Python3 language.

### Project plan ###
...

### Parameters and set-up ###
We first need to download all necessary modules in Python. We split the file into one containing the formulas (`AllFunctions.py`) and one for the calculations (`OptionPricing.py`).

<details> <summary>Click to see the code</summary> <p>

```python
import numpy as np
from scipy.special import erf
np.seterr(divide='ignore', invalid='ignore')
import AllFunctions as func # Only used in the OptionPricing.py file
```
</details> </p>

The decision on the parameters is crucial, at least for some. We will elaborate in a later paragraph, how we derived certain parameters. Here we simply state the most important parameters we used.

<details> <summary>Click to see the code</summary> <p>

```python
# According to Fang 2010, p. 30
# According to Fang 2010, p. 30
r       = 0         # Risk-free rate
mu      = r         # Mean rate of drift
sigma   = 0.0175    # Initial Vola of underyling at time 0; also called u0 or a
S0      = 100       # Today's stock price
tau     = 30 / 365  # Time to expiry in years
q       = 0         # Divindend Yield
lm      = 1.5768    # The speed of mean reversion
v_bar   = 0.0398    # Mean level of variance of the underlying
volvol  = 0.5751    # Volatility of the volatiltiy process (if 0 then constant Vol like BS)
rho     = -0.5711   # Covariance between the log stock and the variance process

# Range of Strikes
K       = np.arange(70, 131, dtype = np.float)

# Truncation Range
L       = 120
a, b    = func.truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol)
bma     = b-a

# Number of Points
N       = 15
k       = np.arange(np.power(2,N))

# Input for the Characterstic Function Phi
u       = k*np.pi/bma
```
</details> </p>

We downloaded the stock price data from... Strike price range of ... Vola?, mu?, r?, q?

## <div id="B2"> <a href="#0">Black Scholes Formula  </a> </div>

As a starting point for valuing European Call options, we decided to apply the Black Scholes option pricing formula to the data at hand. <br>
Black-Scholes pricing formula for a call: <br>
![equation](http://latex.codecogs.com/gif.latex?C(S,&space;t)&space;=&space;S\Phi&space;d_1&space;-&space;Ke^{-r(t-t)}\Phi&space;d_2) <br>
![equation](http://latex.codecogs.com/gif.latex?d_1&space;=&space;\frac{ln(S/K)&space;&plus;&space;(r&space;&plus;&space;\sigma^2/2)*(T-t)}{\sigma&space;*\sqrt(T-t)}) <br>
![equation](http://latex.codecogs.com/gif.latex?d_2&space;=&space;d_1&space;-&space;\sigma&space;\sqrt(T-t)) <br>

Before implementing the Black-Scholes formula in Python, we need to define a function that calculates the cumulative density function of the Standard Normal Distribution.

<details> <summary>Click to see the code</summary> <p>
    
```python
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
```
</details> </p>

## <div id="C2"> <a href="#0">Cosine transform method  </a> </div>

Fang & Oosterlee (2008) presented a new way to price (complex) options using a Fourier-based methods for numerical integration. Until the publication of their results, the Fast Fourier Transform method was known for its computational efficiency in option pricing. The authors introduce the COS method, which will further increase the speed of the calculations. Compared to other methods, which also show high computational speed, the COS method can compute option prices for a vector of strikes and provides an efficient way to recover the density from the characteristic function.

### Truncation range ###
During our project we became aware of the truncation range. As Call options' payoffs rise with increasing stock price a cancellation error can be introduced when valuing call options. This effect does not occur for Put options. This is why we will use Put options to calculate the truncation range and then use the Put-Call-Parity to transfer the findings to the Call options (Fang, 2010, p. 28). <br> <br>

Put-Call-Parity: <br>
![equation](http://latex.codecogs.com/gif.latex?v^{call}(\textup{x},&space;t_0)&space;=&space;v^{put}(\textup{x},&space;t_0)&plus;S_0e^{-qT}-Ke^{-rT}) <br> <br>

For the exact derivation of our Python code, please see Alistair et al. (2008, p. 836).

<details> <summary>Click to see the code</summary> <p>
    
```python
def truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol):
        c1 = mu * tau + (1 - np.exp(-lm * tau)) * (v_bar - sigma)/(2 * lm) - v_bar * tau / 2

        c2 = 1/(8 * np.power(lm,3)) * (volvol * tau * lm * np.exp(-lm * tau) \
            * (sigma - v_bar) * (8 * lm * rho - 4 * volvol) \
            + lm * rho * volvol * (1 - np.exp(-lm * tau)) * (16 * v_bar - 8 * sigma) \
            + 2 * v_bar * lm * tau * (-4 * lm * rho * volvol + np.power(volvol,2) + 4 * np.power(lm,2)) \
            + np.power(volvol,2) * ((v_bar - 2 * sigma) * np.exp(-2 * lm * tau) \
            + v_bar * (6 * np.exp(-lm * tau) - 7) + 2 * sigma) \
            + 8 * np.power(lm,2) * (sigma - v_bar) * (1 - np.exp(-lm * tau)))

        a = c1 - L * np.sqrt(np.abs(c2))
        b = c1 + L * np.sqrt(np.abs(c2))
        return a, b    
```
</details> </p>

### Cosine series expansion ###
Based on the equations (22) and (23) in Fang & Oosterlee (2008), we implemented functions for the Cosine expansion.

<details> <summary>Click to see the code</summary> <p>
    
```python
def cosSerExp(a,b,c,d,k):
    bma = b-a
    uu  = k * np.pi/bma
    chi = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu * (d-a))) * np.exp(d)-np.multiply(uu,np.sin(uu * (c-a))) * np.exp(c)))
    return chi


def cosSer1(a,b,c,d,k):
    bma    = b-a
    uu     = k * np.pi/bma
    uu[0]  = 1      # to avoid case differentiation (done 2 lines below)
    psi    = np.divide(1,uu) * ( np.sin(uu * (d-a)) - np.sin(uu * (c-a)) )
    psi[0] = d-c
    return psi
```
</details> </p>

These Cosine expansions are now used to calculate the payoff series coefficients of the option.
![equation](http://latex.codecogs.com/gif.latex?U_k^{call}&space;=&space;\frac{2}{b-a}(\chi&space;_k(0,b)-\psi&space;_k(0,b))) <br>
![equation](http://latex.codecogs.com/gif.latex?U_k^{call}&space;=&space;\frac{2}{b-a}(-\chi&space;_k(0,b)+\psi&space;_k(0,b))) <br>
Note: <br>
![equation](http://latex.codecogs.com/gif.latex?V_k&space;=&space;U_k&space;K)

<details> <summary>Click to see the code</summary> <p>
    
```python
UkPut = 2 / bma * ( func.cosSer1(a,b,a,0,k) - func.cosSerExp(a,b,a,0,k) )
UkCall = 2 / bma * ( func.cosSerExp(a,b,0,b,k) - func.cosSer1(a,b,0,b,k) )
```

</details> </p>

## <div id="D2"> <a href="#0">Characteristic functions  </a> </div>

### Black-Scholes characteristic function ###
<details> <summary>Click to see the code</summary> <p>
    
```python
charactersticFunctionBS = func.charFuncBSM(u, mu, sigma, tau)

C_COS = np.zeros((np.size(K)))

for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j*k*np.pi*(x-a)/bma)
    Fk = np.real(np.multiply(charactersticFunctionBS, addIntegratedTerm))
    Fk[0]=0.5*Fk[0] 
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,UkCall)) * np.exp(-r*tau)
    
print (C_COS)
```

</details> </p>

### Heston characteristic function ###
From Fang (2010) Eq. (2.32) we implemented the characteristic function for the Heston model. 

<details> <summary>Click to see the code</summary> <p>
    
```python
def charFuncHestonFOH(mu, r, u, tau, sigma, v_bar, lm, rho, volvol):
    d = np.sqrt(np.power(lm - 1j * rho * volvol * u, 2) + np.power(volvol,2) * (np.power(u,2) + u * 1j))
    g = (lm - 1j * rho * volvol * u - d) / (lm - 1j * rho * volvol * u + d)
    #C = np.divide(lm * mu, np.power(volvol,2)) * ( (lm - 1j * rho * volvol * u - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    C = np.divide(lm * v_bar, np.power(volvol,2)) * ( (lm - 1j * rho * volvol * u - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    D = 1j * r * u * tau + sigma / np.power(volvol,2) * (np.divide((1 - np.exp(-d * tau)), (1 - g * np.exp(-d * tau)))) * (lm - 1j * rho * volvol * u - d) 
    phi = np.exp(D) * np.exp(C)
    return phi    
```
</details> </p>

To use the Heston model for option pricing, we combine the above results and validate our Call option price via the Put-Call-Parity. 

<details> <summary>Click to see the code</summary> <p>
    
```python
charactersticFunctionFOH = func.charFuncHestonFOH(mu, r, u, tau, sigma, v_bar, lm, rho, volvol)

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
```
</details> </p>

## <div id="E2"> <a href="#0">Data  </a> </div>
- Which data to use? 
- Which parameters to use for the data?

## <div id="F2"> <a href="#0">Results  </a> </div>
- Computing time
- Efficiency?
- Plot?

## <div id="G2"> <a href="#0">Concluding remarks  </a> </div>
As mentioned in this [Blog entry](https://chasethedevil.github.io/post/the-cos-method-for-heston/), limitations in the COS method are inaccuracy for very small prices. We observed this phenomenon ourselves when comparing the results from our COS-Heston calculations to the Black-Scholes option prices.

## <div id="H2"> <a href="#0">References  </a> </div>

Fang, F. (2010).*The COS Method: An Efficient Fourier Method for Pricing Financial Derivatives*. Doctor thesis. 


### Sample equation ###
![equation](http://latex.codecogs.com/gif.latex?C(S,&space;t)&space;=&space;S\Phi&space;d_1&space;-&space;Ke^{-r(t-t)}\Phi&space;d_2) <br>

### Sample subsection ###
<details> <summary>Click to see the code</summary> <p>
    
```python
    
```

</details> </p>
