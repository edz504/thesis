import cPickle as pickle
import pandas as pd
import numpy as np
import os

MODEL_DIR = '/home/smile/edzhou/Thesis/data/models'
all_files = os.listdir(MODEL_DIR)
all_t0 = sorted(all_files)[0:14]

# We want ['t', 'a{x}', AIC, BIC, fvalue, f_pvalue]

results_df = pd.DataFrame(np.full((14, 6), np.nan),
                          columns=['time_inclusion',
                                   'affordance_indicator',
                                   'AIC',
                                   'BIC',
                                   'f_statistic',
                                   'f_pvalue'])

for i, t_pkl in enumerate(all_t0):
    with open(os.path.join(MODEL_DIR, t_pkl), 'rb') as f:
        these_results = pickle.load(f)
        recorded_results = ['t',
                            'a{0}'.format(i + 1),
                            these_results.aic,
                            these_results.bic,
                            these_results.fvalue,
                            these_results.f_pvalue]
    results_df.iloc[i, :] = recorded_results
    print 'Finished affordance_indicator {0}'.format(i)

print results_df
results_df.to_csv('aic_bic_f_t_only.csv', index=False)