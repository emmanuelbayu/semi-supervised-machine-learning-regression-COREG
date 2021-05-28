import pandas as pd
from coreg import Coreg

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#SPLITTING LABELED AND UNLABELED DATA
train_labeled = train[train['price'].notnull()]
train_unlabeled = train[train['price'].isnull()]
train_labeled_x = train_labeled.drop('price', axis=1).reset_index(drop=True)
train_labeled_y = train_labeled['price'].reset_index(drop=True)
train_unlabeled_x = train_unlabeled.drop('price', axis=1).reset_index(drop=True)
train_unlabeled_y = train_unlabeled['price'].reset_index(drop=True)
test_x = test.drop('price', axis=1)
test_y = test['price']

#Define variables for coreg class
k1 = 3
k2 = 3
p1 = 2
p2 = 5
max_iters = 100 
pool_size = 2997 #pool_size according to the total of rows of unlabeled data
verbose = True
random_state = -1
num_labeled = 100
num_test = 1000

cr = Coreg(k1, k2, p1, p2, max_iters, pool_size) #Create coreg class
cr._add_data(train_labeled_x, train_labeled_y, train_unlabeled_x, train_unlabeled_y, test_x, test_y) #Adding data to coreg class
cr._fit_and_evaluate #Train and evaluate(RMSE)
cr._get_pool() #Getting unlabeled data
cr._run_iteration() #Running iteration to train the unlabeled data
