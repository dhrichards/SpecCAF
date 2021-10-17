import numpy as np
import scipy.stats as stats


class parameters:
    def __init__(self,confidence=0.8):

        self.confidence = confidence

        # Raw data from simple shear and compression inversion Richards et al. 2020
        self.rawT = np.array  ( [-30, -13.6, -10.2, -9.5, -30.3, -7, -5.5])
        self.rawlamb = 2*np.array( [0.173, 0.198, 0.126, 0.343, 0.153, 0.139, 0.178])
        self.rawbeta = 2*np.array([0.62, 4.25, 5.92, 2.75, 0.763, 4.12, 5.51])
        self.rawiota = np.array([1.23, 1.93, 1.54, 1.98, 0.993, 1.65, 1.59])

        self.plamb, self.lambcov = np.polyfit(self.rawT,self.rawlamb, 1, cov=True)
        self.pbeta, self.betacov = np.polyfit(self.rawT,self.rawbeta, 1, cov=True)
        self.piota, self.iotacov = np.polyfit(self.rawT,self.rawiota, 1, cov=True)


    def lamb(self,T):
        return np.polyval(self.plamb,T)

    def beta(self,T):
        return np.polyval(self.pbeta,T)

    def iota(self,T):
        return np.polyval(self.piota,T)

    def lambUB(self,T):
        return self.lamb(T) + self.confidence_interval('lamb',T)

    def lambLB(self,T):
        return self.lamb(T) - self.confidence_interval('lamb',T)

    def iotaUB(self,T):
        return self.iota(T) + self.confidence_interval('iota',T)

    def iotaLB(self,T):
        return self.iota(T) - self.confidence_interval('iota',T)

    def betaUB(self,T):
        return self.beta(T) + self.confidence_interval('beta',T)

    def betaLB(self,T):
        beta = self.beta(T) - self.confidence_interval('beta',T)
        if np.isscalar(beta):
            if beta<0:
                beta = 0
        else:
            beta[beta<0]=0
        return beta
    


    def confidence_interval(self,variable,T):
        n = self.rawT.size                                           # number of observations
        m = self.plamb.size                                                 # number of parameters
        dof = n - m                                                # degrees of freedom
        tconf = 1-(1-self.confidence)/2
        t = stats.t.ppf(tconf, n - m)                              # used for CI and PI bands

        # Estimates of Error in Data/Model
        if variable=='iota':
            p = self.piota
            y_model = self.iota(self.rawT)
            y = self.rawiota
        elif variable=='beta':
            p = self.pbeta
            y_model = self.beta(self.rawT)
            y = self.rawbeta
        elif variable=='lamb':
            p = self.plamb
            y_model = self.lamb(self.rawT)
            y = self.rawlamb


        resid = y - y_model                           
        chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
        chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
        s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

        ci = t * s_err * np.sqrt(1/n + (T - np.mean(self.rawT))**2 / np.sum((self.rawT - np.mean(self.rawT))**2))
        return ci
  