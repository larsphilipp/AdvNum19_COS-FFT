<div align="right">
Advanced Numerical Methods and Data Analysis - FS19-8,780
<br>
University of St. Gallen, 24.03.2019
<br>
</div>

-------------



# AdvNum19_COS-FFT Project Description

**Elisa Fleissner** &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  elisa.fleissner@student.unisg.ch <br>
**Lars Stauffenegger** &nbsp; &nbsp; &nbsp;lars.stauffenegger@student.unisg.ch  <br>

## <div id="0">Overview</div>

1. <a href="#A2">Introduction</a>
2. <a href="#B2">Black Scholes Formula</a>
3. <a href="#C2">Cosine transform method</a>
4. <a href="#D2">Heston model</a>
5. <a href="#E2">Data</a>
6. <a href="#F2">Results</a>
7. <a href="#G2">Concluding remarks</a>


## <div id="A2"> <a href="#0">Introduction  </a> </div>

This is the documentation for the "COS-FFT" assignment of the class **Advanced Numerical Methods and Data Analysis** taught by Prof. Peter Gruber at the University of St. Gallen in Spring 2019. We - Elisa Fleissner and Lars Stauffenegger - are in the 2nd Semester of our Master studies and worked as a group with the aim to use the Cosine transform method as presented in [Fang & Oosterlee (2008)] (http://mpra.ub.uni-muenchen.de/9319/) combined with the [Heston model] (tbd) to value plain-vanilla European Call options. To validate our results, we implemented the Black Scholes model in our calculations. For all calculations we used Python3 language.

### Project plan ###
...

### Parameters and set-up ###
We first need to download all necessary modules in Python. We split the file into one containing the formulas (`AllFunctions.py`) and one for the calculations (`OptionPricing.py`).

```python
import numpy as np
from scipy.special import erf
np.seterr(divide='ignore', invalid='ignore')
import AllFunctions as func # Only used in the OptionPricing.py file
```

The decision on the parameters is crucial, at least for some. We will elaborate in a later paragraph, how we derived certain parameters. Here we simply state the most important parameters we used.

```python
r     = 0          # risk-free rate
mu    = r          # model parameters
sigma = 0.15    
S0    = 100        # Today's stock price
tau   = 30 / 365   # Time to expiry in years
q     = 0
K     = np.arange(70, 131, dtype = np.float)
```

We downloaded the stock price data from... Strike price range of ... Vola?, mu?, r?, q?

## <div id="B2"> <a href="#0">Black Scholes Formula  </a> </div>

As a starting point for valuing European Call options, we decided to apply the Black Scholes option pricing formula to the data at hand. <br>
Black-Scholes pricing formula for a call: <br>
![equation](http://latex.codecogs.com/gif.latex?C(S,&space;t)&space;=&space;S\Phi&space;d_1&space;-&space;Ke^{-r(t-t)}\Phi&space;d_2) <br>
![equation](http://latex.codecogs.com/gif.latex?d_1&space;=&space;\frac{ln(S/K)&space;&plus;&space;(r&space;&plus;&space;\sigma^2/2)*(T-t)}{\sigma&space;*\sqrt(T-t)}) <br>
![equation](http://latex.codecogs.com/gif.latex?d_2&space;=&space;d_1&space;-&space;\sigma&space;\sqrt(T-t)) <br>

Before implementing the Black-Scholes formula in Python, we need to define a function that calculates the cumulative density function of the Standard Normal Distribution. 

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

## <div id="C2"> <a href="#0">Cosine transform method  </a> </div>
A further development from the
![equation](http://latex.codecogs.com/gif.latex?Concentration%3D%5Cfrac%7BTotalTemplate%7D%7BTotalVolume%7D)

## <div id="D2"> <a href="#0">Heston model  </a> </div>
## <div id="E2"> <a href="#0">Data  </a> </div>
## <div id="F2"> <a href="#0">Results  </a> </div>
## <div id="G2"> <a href="#0">Concluding remarks  </a> </div>



