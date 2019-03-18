# found on Github
# https://github.com/khrapovs/fangoosterlee
from __future__ import division, print_function
import numpy as np
import numexpr as ne


### Readin Classes
class HestonParam(object):

    def __init__(self, lm=1.5, mu=.12**2, eta=.57, rho=-.2, sigma=.12**2):
        
        self.lm = lm
        self.mu = mu
        self.eta = eta
        self.rho = rho
        self.sigma = sigma
        
        
class Heston(object):

    def __init__(self, param, riskfree, maturity):
        self.param = param
        self.riskfree = riskfree
        self.maturity = maturity

    # Characterstic Function (Heston)
    def charfun(self, arg):
        lm, mu, eta = self.param.lm, self.param.mu, self.param.eta
        rho, sigma = self.param.rho, self.param.sigma

        d = np.sqrt((lm - 1j * rho*eta*arg)**2 + (arg**2 + 1j*arg) * eta**2)
        g = (lm - 1j * rho * eta * arg - d) / (lm - 1j * rho * eta * arg + d)

        phi = np.exp(1j * arg * self.riskfree * self.maturity + sigma/eta**2 \
            * (1 - np.exp(-d * self.maturity)) \
            / (1 - g * np.exp(-d * self.maturity)) \
            * (lm - 1j * rho * eta * arg - d))

        phi = phi * np.exp(lm * mu / eta**2 * \
            (self.maturity * (lm - 1j * rho * eta * arg - d) \
            - 2 * np.log((1-g * np.exp(-d * self.maturity)) / (1 - g))))

        return phi

    # Determination of a and b
    def cos_restriction(self):
        lm, mu, eta = self.param.lm, self.param.mu, self.param.eta
        rho, sigma = self.param.rho, self.param.sigma

        L = 12
        c1 = self.riskfree * self.maturity \
            + (1 - np.exp(-lm * self.maturity)) \
            * (mu - sigma)/2/lm - mu * self.maturity / 2

        c2 = 1/(8 * lm**3) \
            * (eta * self.maturity * lm * np.exp(-lm * self.maturity) \
            * (sigma - mu) * (8 * lm * rho - 4 * eta) \
            + lm * rho * eta * (1 - np.exp(-lm * self.maturity)) \
            * (16 * mu - 8 * sigma) + 2 * mu * lm * self.maturity \
            * (-4 * lm * rho * eta + eta**2 + 4 * lm**2) \
            + eta**2 * ((mu - 2 * sigma) * np.exp(-2*lm*self.maturity) \
            + mu * (6 * np.exp(-lm*self.maturity) - 7) + 2 * sigma) \
            + 8 * lm**2 * (sigma - mu) * (1 - np.exp(-lm*self.maturity)))

        a = c1 - L * np.abs(c2)**.5
        b = c1 + L * np.abs(c2)**.5

        return a, b

# Plugging together the cos method
def cosmethod(model, moneyness, call, npoints):
# def cosmethod(model, moneyness=0., call=True, npoints=2**10):
    if not hasattr(model, 'charfun'):
        raise Exception('Characteristic function is not available!')
    if not hasattr(model, 'cos_restriction'):
        raise Exception('COS restriction is not available!')

    # (nobs, ) arrays
    alim, blim = model.cos_restriction()
    # (npoints, nobs) array
    kvec = np.arange(npoints)[:, np.newaxis] * np.pi / (blim - alim)
    # (npoints, ) array
    unit = np.append(.5, np.ones(npoints-1))
    # Arguments
    argc = (kvec, alim, blim, 0, blim)
    argp = (kvec, alim, blim, alim, 0)
    # (nobs, ) array
    put = np.logical_not(call)
    # (npoints, nobs) array
    umat = 2 / (blim - alim) * (call * xfun(*argc) - put * xfun(*argp))
    # (npoints, nobs) array
    pmat = model.charfun(kvec)
    # (npoints, nobs) array
    xmat = np.exp(-1j * kvec * (moneyness + alim))
    # (nobs, ) array
    return np.exp(moneyness) * np.dot(unit, pmat * umat * xmat).real

# Cos Expansion
def xfun(k, a, b, c, d):
    out0 = ne.evaluate(("(cos(k * (d-a)) * exp(d) - cos(k * (c-a)) * exp(c)"
        "+ k * (sin(k * (d-a)) * exp(d) - sin(k * (c-a)) * exp(c)))"
        "/ (1 + k**2)"))
    k1 = k[1:]
    out1 = ne.evaluate("(sin(k1 * (d-a)) - sin(k1 * (c-a))) / k1")

    out1 = np.vstack([(d - c) * np.ones_like(a), out1])

    return out0 - out1

if __name__ == '__main__':

    pass
    

#### TEST ---------------------------------------------------------------------
K = np.arange(70, 131, dtype = np.float)
price = 100
riskfree, maturity = 0, 30/365

lm = 1.5768
mu = riskfree #.12**2
eta = .5751
rho = -0.5711  #-.0
sigma = 0.15 #.12**2
param = HestonParam(lm=lm, mu=mu, eta=eta, rho=rho, sigma=sigma)

premium = np.zeros((np.size(K)))

for m in range(0, np.size(K)):
    strike = K[m]
    moneyness = np.log(strike/price) - riskfree * maturity
    #moneyness = np.log(price/strike) - riskfree * maturity
    model = Heston(param, riskfree, maturity)
    premium[m] = price * (cosmethod(model, moneyness, True, 2**10))

print(premium)


### END