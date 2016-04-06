import os
import numpy as np
import pandas as pd
import cPickle as pickle

MODEL_DIR = '/home/smile/edzhou/Thesis/data/models'
all_files = os.listdir(MODEL_DIR)
all_linear = sorted(all_files)[0:(len(all_files) - 1)]

# p-values for the y = beta * x_t regression
t_mat = np.full((500, 14), np.nan)

for i, t_pkl in enumerate([f for f in all_linear if 't1' not in f]):
    with open(os.path.join(MODEL_DIR, t_pkl), 'rb') as f:
        these_results = pickle.load(f)
    t_mat[:, i] = these_results.pvalues

# p-values for the y = beta1 * x_t + beta2 * x_(t-1) regression
t1_mat = np.full((1000, 14), np.nan)

for i, t1_pkl in enumerate([f for f in all_linear if 't1' in f]):
    with open(os.path.join(MODEL_DIR, t1_pkl), 'rb') as f:
        these_pvalues = pickle.load(f)
    t1_mat[:, i] = these_pvalues

DATA_DIR = '/home/smile/edzhou/Thesis/data'
t_df = pd.DataFrame(t_mat)
t1_df = pd.DataFrame(t1_mat)
t_df.to_csv(os.path.join(DATA_DIR, 't_pvalues.csv'), index=False)
t1_df.to_csv(os.path.join(DATA_DIR, 't-1_pvalues.csv'), index=False)