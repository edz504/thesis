import time
import numpy as np
import pandas as pd
import hickle as hkl
import statsmodels.api as sm
import cPickle as pickle

y_scaled = hkl.load(
  '/home/smile/edzhou/Thesis/data/train_y_scaled_gzip.hkl')

# Save the AIC, BIC, F-statistic, and p-value of F-statistic for
# each model (for each time inclusion + each affordance indicator).
results_df = pd.DataFrame(np.full((84, 6), np.nan),
                          columns=['time_inclusion',
                                   'affordance_indicator',
                                   'AIC',
                                   'BIC',
                                   'f_statistic',
                                   'f_pvalue'])

# Fit models for t - 1 , ... t - 6.
model_count = 0
for t in xrange(1, 7):
  # Fit one model for each affordance indicator.
  X_cnn_this_t = hkl.load(
    '/home/smile/edzhou/Thesis/data/train_cnn_500_t{0}_gzip.hkl'.format(
      t))
  print 'Loaded X for t-{0}'.format(t)
  for a in xrange(0, y_scaled.shape[1]): # 14 affordance indicators
      start = time.time() 
      # Start with "t"-th, not 0th data point due to recurrence.
      model = sm.OLS(y_scaled[t:, a:(a + 1)], X_cnn_this_t)
      results = model.fit()
      end = time.time()
      print 'Affordance indicator {0} took {1} to fit with t-{2}'.format(
        a, end - start, t)
      recorded_results = ['t-{0}'.format(t),
                          'a{0}'.format(a + 1),
                          results.aic,
                          results.bic,
                          results.fvalue,
                          results.f_pvalue]
      print recorded_results
      results_df.iloc[model_count, :] = recorded_results
      model_count += 1

results_df.to_csv('aic_bic_f_t-1_through_t-6.csv', index=False)
