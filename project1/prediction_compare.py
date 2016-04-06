import time
import numpy as np
import pandas as pd
import hickle as hkl
import statsmodels.api as sm
import cPickle as pickle


num_models = y_scaled.shape[1]
num_to_name = dict(zip(xrange(0, num_models + 1),
                       ['0' + str(i) if i < 10 else i for
                       i in xrange(0, num_models + 1)]))

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
