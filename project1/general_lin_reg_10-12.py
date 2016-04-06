# NOTE (11/4): killed after t - 10, unclear why.  Memory?  Commenting out
# t - 10 stuff, trying t - 11 and t - 12 again.  Also renamed the .csv
# to just t - 10.

import time
import numpy as np
import pandas as pd
import hickle as hkl
import statsmodels.api as sm
import cPickle as pickle

y_scaled = hkl.load(
  '/home/smile/edzhou/Thesis/data/train_y_scaled_gzip.hkl')
num_models = y_scaled.shape[1]
num_to_name = dict(zip(xrange(0, num_models + 1),
                       ['0' + str(i) if i < 10 else i for
                       i in xrange(0, num_models + 1)]))

# Save the AIC, BIC, F-statistic, and p-value of F-statistic for
# each model (for each time inclusion + each affordance indicator).
results_df = pd.DataFrame(np.full((14 * 2, 6), np.nan),
                          columns=['time_inclusion',
                                   'affordance_indicator',
                                   'AIC',
                                   'BIC',
                                   'f_statistic',
                                   'f_pvalue'])

# Fit models for t - 10, t - 11, t - 12.
# ^ Just t - 11, t - 12.
model_count = 0
for t in xrange(11, 13):
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
      # Save to csv at each step in case something breaks.
      results_df.to_csv('aic_bic_f_t-11_through_t-12.csv', index=False)
