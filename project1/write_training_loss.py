import os
import numpy as np
import pandas as pd
from nolearn.lasagne import NeuralNet
import cPickle as pickle

# Load the model
with open('/home/smile/edzhou/Thesis/data/models/model1.pkl', 'rb') as f:
    convnet = pickle.load(f)

train_loss = np.array([i["train_loss"] for i in convnet.train_history_])
valid_loss = np.array([i["valid_loss"] for i in convnet.train_history_])

loss_df = pd.DataFrame({'train_loss': train_loss,
                        'valid_loss': valid_loss})
loss_df.to_csv('loss.csv', index=False)