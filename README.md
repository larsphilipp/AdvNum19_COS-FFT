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
2. <a href="#E2">Data</a>
3. <a href="#B2">Black Scholes Formula</a>
4. <a href="#C2">Cosine transform method</a>
5. <a href="#D2">Characteristic functions</a>
6. <a href="#F2">Results</a>
7. <a href="#G2">Concluding remarks</a>
8. <a href="#H2">References</a>


## <div id="A2"> <a href="#0">Introduction  </a> </div>

This is the documentation for the "COS-FFT" assignment of the class **Advanced Numerical Methods and Data Analysis** taught by Prof. Peter Gruber at the University of St. Gallen in Spring 2019. We - Elisa Fleissner and Lars Stauffenegger - are in the 2nd Semester of our Master studies and worked as a group with the aim to use the Cosine transform method as presented in [Fang & Oosterlee (2008)](http://mpra.ub.uni-muenchen.de/9319/) combined with the Heston model to value plain-vanilla European Call options. To validate our results, we implemented the Black Scholes model in our calculations. For all calculations we used Python3 language.

### Project plan ###
The minimum requierements of the project consisted of the picing of plain vanilla options for one underlying and a range of strikes using the Heston model in the COS method. We focused on the technical implementation of the Heston characteristic function as this required already a deep understanding of the underlying reading. Fang and Oosterle (2008) use a different notation and slightly rearranged formulas of the Heston model compared to the original paper by Heston in 1993. This required both of us to first read both papers as well as further literature such as a paper by Wang (2017) to better understand error sources (e.g. truncation error). In this type of an exercise, coding work cannot really be split, we hence worked alternately on the code using Github for version control and documentation. Our target was to have a working model with automated stock data inputs calculating Call as well as Put option prices using different approaches: the Black Scholes Model, the COS Method with the Characterstic function used in Black Scholes and the COS Method applying Heston's characteristic function.

### Setup ###
We first needed to download all necessary modules in Python. We split the file into one containing the formulas (`AllFunctions.py`) and one for the calculations (`OptionPricing.py`).

<details> <summary>Click to see the code</summary> <p>

```python
import quandl
import numpy as np
import AllFunctions as func
import matplotlib.pyplot as plt
from scipy.special import erf
```
</details> </p>

## <div id="E2"> <a href="#0">Data and parameters  </a> </div>

### Stock data ###
To price the options, the code downloads historical data from [Quandl](www.quandl.com) for one company (Apple). We decided to download the last 500 trading days to have a large enough sample for estimating the variance. We did not make an estimate for rho and the vola and therefore used the original parameters from the Fang & Oosterlee (2008) paper. 

<details> <summary>Click to see the code</summary> <p>

```python
# Import from quandl
quandl.ApiConfig.api_key = "mrMTRoAdPycJSyzyjxPN"
ticker     = "AAPL"
database   = "EOD"
identifier = database + "/" + ticker
stockData  = quandl.get(identifier, rows = 500)

# Return and Volatility
logReturn = np.log(stockData.Close) - np.log(stockData.Close.shift(1))
logReturn.drop(logReturn.index[:1], inplace = True)
tradingDaysCount   = 252
annualisedMean     = np.mean(logReturn) * tradingDaysCount
annualisedVariance = np.var(logReturn) * tradingDaysCount
annualisedStdDev   = np.sqrt(annualisedVariance)
lastPrice          = stockData.Close.tail(1)

```
</details> </p>

### Parameters ###
The decision on the parameters is crucial, at least for some. We tried to implement most parameters from the data we downloaded from Quandl, but we did not estimate the parameters `volvol` and `rho`. For the range of strike prices we decided to use a percentage of 20% around the current stock price to have a sample of in-the-money, at-the-money and out-of-the-money options to price. 

<details> <summary>Click to see the code</summary> <p>

```python
# Volvol and rho according to Fang, 2010, p. 30
r      = 0                  # assumption Risk-free rate
mu     = r #annualisedMean  # Mean rate of drift
sigma  = annualisedStdDev   # Initial Vola of underyling at time 0; also called u0 or a
S0     = lastPrice[0]       # Today's stock price
tau    = 30 / 365           # Time to expiry in years
q      = 0                  # Divindend Yield
lm     = 1.5768             # The speed of mean reversion
v_bar  = annualisedVariance # Mean level of variance of the underlying
volvol =  0.5751            # Volatility of the volatiltiy process
rho    = -0.5711            # Covariance between the log stock and the variance process

# Range of Strikes
mini    = int(S0 * 0.8)
maxi    = int(S0 * 1.2)
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
```
</details> </p>

### Truncation range ###
The choice of the truncation range can be essential for the pricing of the options. We figured out that our initial range determination (simply a multiple of the time-to-maturity adjusted standard deviation) delivered an unsatifying ouput for example for low volality. Hence we decided to check for another approach and implemented a more sopisticated determination of the truncation range as presented in Fitt et al. (2010). <br>

For the exact derivation of our Python code, please see Fitt et al. (2010, p. 836).

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


## <div id="B2"> <a href="#0">Black Scholes Formula  </a> </div>

As a starting point for valuing European Call options, we decided to apply the Black Scholes option pricing formula to the data at hand. <br>
Black-Scholes pricing formula for a call: <br><br>
![equation](http://latex.codecogs.com/gif.latex?C(S,&space;t)&space;=&space;S\Phi&space;d_1&space;-&space;Ke^{-r(t-t)}\Phi&space;d_2) <br>
![equation](http://latex.codecogs.com/gif.latex?d_1&space;=&space;\frac{ln(S/K)&space;&plus;&space;(r&space;&plus;&space;\sigma^2/2)*(T-t)}{\sigma&space;*\sqrt(T-t)}) <br>
![equation](http://latex.codecogs.com/gif.latex?d_2&space;=&space;d_1&space;-&space;\sigma&space;\sqrt(T-t)) <br>

Before implementing the Black-Scholes formula in Python, we need to define a function that calculates the cumulative density function of the Standard Normal Distribution. Then we can use the Black-Scholes formula to derive the price of the Call option. We store these results in `C_BS` which we will later use for comparison.

<details> <summary>Click to see the code</summary> <p>
    
```python
# Cumulative Distribution Function
def StdNormCdf(z):
    phi = 0.5 * (1 + erf(z/np.sqrt(2)))
    return phi

# Black Scholes Model
def blackScholes(S, K, r, tau, sigma, q):
    S = S * np.exp(-q * tau)
    d1 = np.divide((np.log(np.divide(S, K)) + (r + 1/2 * np.power(sigma, 2)) * tau), (sigma * np.sqrt(tau)))
    d2 = d1 - sigma * np.sqrt(tau)
    call  = np.multiply(S, StdNormCdf(d1)) - np.multiply(np.multiply(K, np.exp(-r * tau)), StdNormCdf(d2))
    put  = call + np.multiply(K, np.exp(-r * tau)) - S
    return call, put
    
C_BS, P_BS = func.blackScholes(S0, K, r, tau, sigma, q)
print(C_BS)
```
</details> </p>

## <div id="C2"> <a href="#0">Cosine transform method  </a> </div>

Fang & Oosterlee (2008) presented a new way to price (complex) options using a Fourier-based methods for numerical integration. Until the publication of their results, the Fast Fourier Transform method was known for its computational efficiency in option pricing. The authors introduce the COS method, which will further increase the speed of the calculations. Compared to other methods, which also show high computational speed, the COS method can compute option prices for a vector of strikes and provides an efficient way to recover the density from the characteristic function.

### Cosine series expansion ###
The equations (22) and (23) in Fang & Oosterlee (2008) were translated into Python from the Matlab code provided by Prof. Gruber as part of the lecture.

<details> <summary>Click to see the code</summary> <p>
    
```python
def cosSerExp(a, b, c, d, k):
    bma = b-a
    uu  = k * np.pi/bma
    chi = np.multiply(np.divide(1, (1 + np.power(uu,2))), (np.cos(uu * (d-a)) * np.exp(d) - np.cos(uu * (c-a)) * np.exp(c) + np.multiply(uu,np.sin(uu * (d-a))) * np.exp(d)-np.multiply(uu,np.sin(uu * (c-a))) * np.exp(c)))
    return chi


def cosSer1(a, b, c, d, k):
    bma    = b-a
    uu     = k * np.pi/bma
    uu[0]  = 1
    psi    = np.divide(1,uu) * ( np.sin(uu * (d-a)) - np.sin(uu * (c-a)) )
    psi[0] = d-c
    return psi
```
</details> </p>

These Cosine expansions are now used to calculate the payoff series coefficients of the option.<br>
![equation](http://latex.codecogs.com/gif.latex?U_k^{call}&space;=&space;\frac{2}{b-a}(\chi&space;_k(0,b)-\psi&space;_k(0,b))) <br>
![equation](http://latex.codecogs.com/gif.latex?U_k^{put}&space;=&space;\frac{2}{b-a}(-\chi&space;_k(a,0)+\psi&space;_k(a,0))) <br>
Note: <br>
![equation](http://latex.codecogs.com/gif.latex?V_k&space;=&space;U_k&space;K)

<details> <summary>Click to see the code</summary> <p>
    
```python
UkPut  = 2 / bma * ( func.cosSer1(a,b,a,0,k) - func.cosSerExp(a,b,a,0,k) )
UkCall = 2 / bma * ( func.cosSerExp(a,b,0,b,k) - func.cosSer1(a,b,0,b,k) )
```

</details> </p>

## <div id="D2"> <a href="#0">Characteristic functions  </a> </div>

### Black-Scholes characteristic function ###
To calculate the prices with the COS method (without Heston) we first applied the characteristic function from the Black Scholes model. 

<details> <summary>Click to see the code</summary> <p>
    
```python
def charFuncBSM(s, mu, sigma, T):
    phi = np.exp((mu - 0.5 * np.power(sigma,2)) * 1j * np.multiply(T,s) - 0.5 * np.power(sigma,2) * T * np.power(s,2))
    return phi

charactersticFunctionBS = func.charFuncBSM(u, mu, sigma, tau)

C_COS = np.zeros((np.size(K)))

for m in range(0,np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(np.multiply(charactersticFunctionBS, addIntegratedTerm))
    Fk[0]=0.5 * Fk[0] 
    C_COS[m] = K[m] * np.sum(np.multiply(Fk,UkCall)) * np.exp(-r * tau)
    
print (C_COS)
```

</details> </p>

### Heston characteristic function ###
From Fang (2010) Eq. (2.32) we implemented the characteristic function for the Heston model. 

<details> <summary>Click to see the code</summary> <p>
    
```python
def charFuncHestonFO(mu, r, u, tau, sigma, v_bar, lm, rho, volvol):
    d = np.sqrt(np.power(lm - 1j * rho * volvol * u, 2) + np.power(volvol,2) * (np.power(u,2) + u * 1j))
    g = (lm - 1j * rho * volvol * u - d) / (lm - 1j * rho * volvol * u + d)
    C = np.divide(lm * v_bar, np.power(volvol,2)) * ( (lm - 1j * rho * volvol * u - d) * tau - 2 * np.log(np.divide((1 - g * np.exp(-d * tau)) , (1-g)) ))
    D = 1j * r * u * tau + sigma / np.power(volvol,2) * (np.divide((1 - np.exp(-d * tau)), (1 - g * np.exp(-d * tau)))) * (lm - 1j * rho * volvol * u - d) 
    phi = np.exp(D) * np.exp(C)
    return phi
```
</details> </p>

To use the Heston model for call option pricing we determined the Put option prices first and calculated the corresponding Call option prices using the Put-Call-Parity. As Call options' payoffs rise with increasing stock price a cancellation error can be introduced when valuing call options. This effect does not occur for Put options. (Fang, 2010, p. 28).  <br>
Put-Call-Parity: <br><br>
![equation](http://latex.codecogs.com/gif.latex?v^{call}(\textup{x},&space;t_0)&space;=&space;v^{put}(\textup{x},&space;t_0)&plus;S_0e^{-qT}-Ke^{-rT}) <br> <br>

<details> <summary>Click to see the code</summary> <p>
    
```python
charactersticFunctionHFO = func.charFuncHestonFO(mu, r, u, tau, sigma, v_bar, lm, rho, volvol)

C_COS_HFO = np.zeros((np.size(K)))
P_COS_HFO = np.zeros((np.size(K)))
C_COS_PCP = np.zeros((np.size(K)))

for m in range(0, np.size(K)):
    x  = np.log(S0/K[m])
    addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
    Fk = np.real(charactersticFunctionHFO * addIntegratedTerm)
    Fk[0] = 0.5 * Fk[0]						
    C_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkCall)) * np.exp(-r * tau)
    P_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau)
    C_COS_PCP[m] = P_COS_HFO[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau)

print(C_COS_HFO)
print(P_COS_HFO)
print(C_COS_PCP)
```
</details> </p>


## <div id="F2"> <a href="#0">Results  </a> </div>

To visualise the results, we plot the option prices (y-axis) compared to the strike prices (x-axis). The standard and the COS-Black-Scholes prices are identical. One can see that especially at-the-money call option prices under Heston exceed the ones using the Black Scholes characterstic function. A comparison is difficult as some of the Heston input parameters were simply taken from the Fang & Oosterle (2008) paper and might not represent accurate estimates for our underlying. It seems that the shift from constant volatility (Black Scholes) to stochastic volatility (Heston) causes the vega-sensitive options to increase in price.

<details> <summary>Click to see the code</summary> <p>

```python
plt.plot(K, C_BS, "g.", K, C_COS, "b.", K, C_COS_HFO, "r.")
plt.axvline(x = S0)
plt.show()
print("C_BS = green, C_COS = blue, C_COS_HFO = red")
```
</details> </p>


<div align="center">
    <img src="/img.png" width = "400px" </img> 
</div>


## <div id="G2"> <a href="#0">Concluding remarks  </a> </div>
This project was rather technical and it took a lot of effort to get all formulas correct. However, once running, the COS-FFT method combined with the Heston model provides a powerful tool to price many options with very high efficiency. In a next step, it would also be interesting to value other options such as digital or barrier options. <br>
There is also some criticism on this method. As mentioned in this [Blog entry](https://chasethedevil.github.io/post/the-cos-method-for-heston/), limitations in the COS method are inaccuracy for very small prices.

## <div id="H2"> <a href="#0">References  </a> </div>

Fang, F. (2010). *The COS Method: An Efficient Fourier Method for Pricing Financial Derivatives*. Doctor thesis. <br>
Fang, F. & Oosterlee, K. (2008). *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*. <br>
Fitt, A., Norbury, J., Ockendon, H. & Wilson, E. (2010). *Progress in Industrial Mathematics at ECMI 2008*. Springer Berlin. <br>
Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. <br>
Wang, C. (2017). *Pricing European Options by Stable Fourier-Cosine Series Expansions*. <br>
