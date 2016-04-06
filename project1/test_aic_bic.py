import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Generate artificial data (2 regressors + constant)
nobs = 100 
X = np.random.random((nobs, 2)) 
X = sm.add_constant(X)
beta = [1, .1, .5] 
e = np.random.random(nobs)
y = np.dot(X, beta) + e 

# Fit regression model
sm_results = sm.OLS(y, X).fit()
regr = LinearRegression(fit_intercept=False)
regr.fit(X, y)

# SSR calculation
print 'sm SSR: {0}'.format(sm_results.ssr)
y_pred = regr.predict(X)
SSR = sum([(y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y)])
print 'sk SSR: {0}'.format(SSR)

# Log-likelihood calculation
N = nobs
s2 = SSR / N
L = (1.0 / np.sqrt(2 * np.pi * s2)) ** N * np.exp(-SSR / (s2 * 2.0))
# better to calculate in log form
logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
print 'log-likelihood =', np.log(L)
print 'log-likelihood (log space) =', logL

# AIC and BIC
k = X.shape[1]
print 'sm AIC: {0}'.format(sm_results.aic)
print 'sk AIC: {0}'.format(2 * k - 2 * np.log(L))

print 'sm BIC: {0}'.format(sm_results.bic)
print 'sk BIC: {0}'.format(k * np.log(N) - 2 * np.log(L))

# Success!


# Here's the code for calculating within the loop
    y_pred = regr.predict(X_cnn_this_t)
    # Calculate SSR, log-likelihood, and AIC / BIC for each response var.
    for a in xrange(0, y_pred.shape[1]):
        SSR = sum([(y_p - y_t) ** 2
                   for y_p, y_t in zip(y_pred[:, a], y_true[:, a])])
        N = y_pred.shape[0]
        s2 = SSR / N
        logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
        k = X.shape[1]
        aic = 2 * k - 2 * logL
        bic = k * np.log(N) - 2 * logL