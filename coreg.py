from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from time import time
import pandas as pd
import numpy as np
import math

class Coreg():
    def __init__(self, k1=3, k2=3, p1=2, p2=5, max_iters=100, pool_size=543):
        self.k1, self.k2 = k1, k2 # number of neighbors
        self.p1, self.p2 = p1, p2 # distance metrics
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2 = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.h1_temp = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2_temp = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        
    def _add_data(self, labeled_X, labeled_y, unlabeled_X, unlabeled_y, test_x, test_y):
        """
        Add data
        """
        self.X_labeled = labeled_X
        self.y_labeled = labeled_y
        self.X_test = test_x
        self.y_test = test_y
        self.X_unlabeled = unlabeled_X
        self.y_unlabeled = unlabeled_y
        
        self.L1_X = self.X_labeled[:]
        self.L1_y = self.y_labeled[:]
        self.L2_X = self.X_labeled[:]
        self.L2_y = self.y_labeled[:]
        self.U_X = self.X_unlabeled[:]
        self.U_y = self.y_unlabeled[:]
        
    def _fit_and_evaluate(self, verbose):
        """
        Fits h1 and h2 then evaluate metrics
        """
        self.h1.fit(self.L1_X, self.L1_y)
        self.h2.fit(self.L2_X, self.L2_y)
        self._evaluate_metrics(verbose)
        
    def _evaluate_metrics(self, verbose):
        """
        Evaluates KNN regressors on training and test data.
        """
        train1_hat = self.h1.predict(self.X_labeled)
        train2_hat = self.h2.predict(self.X_labeled)
        train_hat = 0.5 * (train1_hat + train2_hat)
        test1_hat = self.h1.predict(self.X_test)
        test2_hat = self.h2.predict(self.X_test)
        test_hat = 0.5 * (test1_hat + test2_hat)
        self.rmse1_train = math.sqrt(mean_squared_error(train1_hat, self.y_labeled))
        self.rmse1_test = math.sqrt(mean_squared_error(test1_hat, self.y_test))
        self.rmse2_train = math.sqrt(mean_squared_error(train2_hat, self.y_labeled))
        self.rmse2_test = math.sqrt(mean_squared_error(test2_hat, self.y_test))
        self.rmse_train = math.sqrt(mean_squared_error(train_hat, self.y_labeled))
        self.rmse_test = math.sqrt(mean_squared_error(test_hat, self.y_test))
        if verbose:
            print('RMSEs:')
            print('  KNN1:')
            print('    Train: {:0.4f}'.format(self.rmse1_train))
            print('    Test : {:0.4f}'.format(self.rmse1_test))
            print('  KNN2:')
            print('    Train: {:0.4f}'.format(self.rmse2_train))
            print('    Test : {:0.4f}'.format(self.rmse2_test))
            print('  Combined:')
            print('    Train: {:0.4f}'.format(self.rmse_train))
            print('    Test : {:0.4f}\n'.format(self.rmse_test))
            
    def _get_pool(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U_X_pool, self.U_y_pool, self.U_idx_pool = shuffle(self.U_X, self.U_y, range(self.U_y.size))
        self.U_X_pool = self.U_X_pool[:self.pool_size]
        self.U_y_pool = self.U_y_pool[:self.pool_size]
        self.U_idx_pool = self.U_idx_pool[:self.pool_size]
        
    def _find_points_to_add(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                        'x2': None, 'y2': None, 'idx2': None}
        # Keep track of added idxs
        added_idxs = []
        for idx_h in [1, 2]:
            if idx_h == 1:
                h = self.h1
                h_temp = self.h1_temp
                L_X, L_y = self.L1_X, self.L1_y
            elif idx_h == 2:
                h = self.h2
                h_temp = self.h2_temp
                L_X, L_y = self.L2_X, self.L2_y
            deltas = self._compute_deltas(L_X, L_y, h, h_temp)
            # Add largest delta (improvement)
            sort_idxs = np.argsort(deltas)[::-1] # max to min
            max_idx = sort_idxs[0]
            if max_idx in added_idxs: max_idx = sort_idxs[1]
            if deltas[max_idx] > 0:
                added_idxs.append(max_idx)
                x_u = self.U_X_pool.iloc[max_idx].values.reshape(1, -1)
                y_u_hat = h.predict(x_u)
                self.to_add['x' + str(idx_h)] = x_u
                self.to_add['y' + str(idx_h)] = y_u_hat
                self.to_add['idx' + str(idx_h)] = self.U_idx_pool[max_idx]
                
    def _compute_delta(self, omega, L_X, L_y, h, h_temp):
        """
        Computes the improvement in MSE among the neighbors of the point being
        evaluated.
        """
        delta = 0
        for idx_o in omega:
            delta += (L_y[idx_o].reshape(1, -1) -
                      h.predict(L_X.iloc[idx_o].values.reshape(1, -1))) ** 2
            delta -= (L_y[idx_o].reshape(1, -1) -
                      h_temp.predict(L_X.iloc[idx_o].values.reshape(1, -1))) ** 2
        return delta

    def _compute_deltas(self, L_X, L_y, h, h_temp):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        deltas = np.zeros((self.U_X_pool.shape[0],))
        for idx_u in (self.U_X_pool.index):
            # Make prediction
            x_u = self.U_X_pool
            y_u_hat = h.predict(x_u)
            # Compute neighbors
            omega = h.kneighbors(x_u, return_distance=False)[0]
            # Retrain regressor after adding unlabeled point
            X_temp = pd.concat([L_X, x_u])
            y_temp = pd.concat([L_y, pd.Series(y_u_hat)]) # use predicted y_u_hat
            h_temp.fit(X_temp, y_temp)
            delta = self._compute_delta(omega, L_X, L_y, h, h_temp)
            deltas[idx_u] = delta
        return deltas
    
    def _run_iteration(self, t, t0, verbose=False, store_results=False):
        """
        Run t-th iteration of co-training, returns stop_training=True if
        no more unlabeled points are added to label sets.
        """
        stop_training = False
        if verbose: print('Started iteration {}: {:0.2f}s'.format(t, time()-t0))
        self._find_points_to_add()
        added = self._add_points()
        if added:
            self._fit_and_evaluate(verbose)
            if store_results:
                self._store_results(t)
            self._remove_from_unlabeled()
            self._get_pool()
        else:
            stop_training = True
        return stop_training
    
    def _remove_from_unlabeled(self):
        # Remove added examples from unlabeled
        to_remove = []
        if self.to_add['idx1'] is not None:
            to_remove.append(self.to_add['idx1'])
        if self.to_add['idx2'] is not None:
            to_remove.append(self.to_add['idx2'])
        self.U_X = np.delete(self.U_X, to_remove, axis=0)
        self.U_y = np.delete(self.U_y, to_remove, axis=0)
        
    def _store_results(self, iteration):
        """
        Stores current MSEs.
        """
        self.mses1_train[self.trial,iteration] = self.mse1_train
        self.mses1_test[self.trial,iteration] = self.mse1_test
        self.mses2_train[self.trial,iteration] = self.mse2_train
        self.mses2_test[self.trial,iteration] = self.mse2_test
        self.mses_train[self.trial,iteration] = self.mse_train
        self.mses_test[self.trial,iteration] = self.mse_test
        
    def _add_points(self):
        """
        Adds new examples to training sets.
        """
        added = False
        if self.to_add['x1'] is not None:
            self.L2_X = pd.concat([self.L2_X, pd.Series(self.to_add['x1'])])
            self.L2_y = pd.concat([self.L2_y, pd.Series(self.to_add['y1'])])
            added = True
        if self.to_add['x2'] is not None:
            self.L1_X = pd.concat([self.L1_X, self.to_add['x2']])
            self.L1_y = pd.concat([self.L1_y, self.to_add['y2']])
            added = True
        return added