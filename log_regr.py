import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sys import stdin
import pickle
import os

def split_sequences(sequence, n_steps):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix < len(sequence):
            seq_x = sequence[i:end_ix]
            if sequence[i+n_steps] > 0:
                seq_y = 1.0
            else:
                seq_y = 0.0
            X.append(seq_x)
            y.append(seq_y)
    return np.array(X), np.array(y)

if os.path.isfile('logregrmodel.pkl'):
    with open('logregrmodel.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        for line in stdin:
            if line != "":
                line_input = [float(x) for x in line.split(',')]
                x = line_input[-2:]
                print(model.predict([x])[0])
            else:
                break
else:
    data = pd.read_csv('LatencyTraining.csv')
    log_returns = data['LogReturns'].to_numpy()
    n_steps = 2
    train_X, train_y = split_sequences(log_returns, n_steps)

    logregr = LogisticRegression()
    scores = cross_val_score(logregr, train_X, train_y, cv=10)
    logregr = LogisticRegression().fit(train_X, train_y)
    with open('logregrmodel.pkl', 'wb') as model_file:
        pickle.dump(logregr, model_file)