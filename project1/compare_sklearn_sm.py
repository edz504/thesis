import time
import hickle as hkl
import statsmodels.api as sm
from sklearn import linear_model

y_scaled = hkl.load(
  '/home/smile/edzhou/Thesis/data/train_y_scaled_gzip.hkl')
print 'loaded y'
X_cnn = hkl.load(
    '/home/smile/edzhou/Thesis/data/train_cnn_500_gzip.hkl')
print 'loaded X'

# The below produces the same 2 results, with statsmodels taking 117s
# and sklearn taking 27.
start = time.time() 
model = sm.OLS(y_scaled[:, 0], X_cnn)
sm_result = model.fit()
end = time.time()   
print 'statsmodels took {0}'.format(end - start)

start = time.time()
regr = linear_model.LinearRegression(fit_intercept=True, n_jobs = -1)
regr.fit(X_cnn, y_scaled[:, 0])
end = time.time()
print 'sklearn took {0}'.format(end - start)

print 'sm coef == sklearn coef?'
print np.array_equal(sm_result.params, regr.coef_)


print 'statsmodel AIC: {0}'.format(sm_result.aic)
print 'statsmodel BIC: {0}'.format(sm_result.bic)

y_pred = regr.predict(X_cnn)
SSR = sum([(y_p - y_t) ** 2 for y_p, y_t in zip(y_pred, y_scaled[:, 0])])
N = y_pred.shape[0]
s2 = SSR / N
logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
k = X_cnn.shape[1]
print 'sklearn AIC: {0}'.format(2 * k - 2 * logL)
print 'sklearn BIC: {0}'.format(np.log(N) * k - 2 * logL)


185.990777252
-7813855.92608
AIC for a0, t-0 = 15628619.8522
BIC for a0, t-0 = 15635362.9883