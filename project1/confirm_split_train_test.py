import numpy as np

def wrangle_x(X, num_prev):
        nrows = X.shape[0] - num_prev
        ncols = X.shape[1] * (num_prev + 1)
        X_prev = np.full((nrows, ncols), np.nan)
        for i in xrange(num_prev, nrows + num_prev):
                total = X[i]
                for j in xrange(1, num_prev + 1):
                        total = np.append(total, X[i - j])
                X_prev[i - num_prev] = total
        return X_prev

N = 15
X0 = np.array(xrange(1, N + 1)).reshape(N, 1)
y0 = np.array([10 * i for i in xrange(1, N + 1)])
X_vec = [X0] + [wrangle_x(X0, t) for t in xrange(1, 8)]
y_vec = [y0] + [y0[i:] for i in xrange(1, 8)]

train_proportion = 0.8
all_ind = np.array(xrange(0, N))
T = 7
train_ind = np.array(xrange(0, T))
train_ind = np.concatenate(
        (train_ind,
         np.random.choice(all_ind[7:], int(train_proportion * N) - T, replace=False)))
test_ind = np.array(list(set(all_ind) - set(train_ind)))

for t, (X, y) in enumerate(zip(X_vec, y_vec)):
    if t > 0:
        train_ind = [ind - 1 for ind in train_ind]
        train_ind.remove(-1)
        train_ind = np.array(train_ind)
        test_ind = [ind - 1 for ind in test_ind]
    print '======'
    print 'X_train:'
    print X[train_ind, :]
    print 'y_train:'
    print y[train_ind]
    print 'X_test:'
    print X[test_ind, :]
    print 'y_test:'
    print y[test_ind]


