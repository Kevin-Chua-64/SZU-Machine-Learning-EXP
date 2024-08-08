from sklearn.model_selection import train_test_split
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class LinerRegression():
    def __init__(self, data):
        self.data = data
        # See if there are any missing values i the data
        print('\nCheck missing values', self.data.isna().sum(), sep='\n')
        print(self.data.head())

    # Divide data and target
    def divide(self):
        X = self.data.iloc[:, : -1].values
        y = self.data.iloc[:, -1].values
        return X, y

    # linear regression
    def linear(self, X, y, sequence=0):
        # use sequence number to record the matrix if it is singular
        # w_ = (w, b).T
        # X_ = (X, 1)
        # y = X_ * w_

        # w_ = (X_.T * X_).I * X_.T * y
        try:
            X_ = np.column_stack((X, np.ones(X.shape[0])))
            w_ = np.dot(np.dot(np.linalg.inv(np.dot(X_.T, X_)), X_.T), y)
            return w_
        # (X_.T * X_) might be singular matrix
        except:
            print(f'The matrix sequence={sequence} is a singular matrix.')
            return None

    # ridge regression
    def ridge(self, X, y, lamda):
        # w_ = (w, b).T
        # X_ = X - X.mean(axis=0)
        # y_ = y - y.mean(axis=0)
        # y = X_ * w + b

        # w = (X.T * X + lamda*I).I * X.T * y
        # b = y.mean(axis=0) - X.mean(axis=0) * w
        X_ = X - X.mean(axis=0)
        y_ = y - y.mean(axis=0)
        w = np.dot(np.dot(np.linalg.inv(np.dot(X_.T, X_) + np.diag([lamda] * X.shape[1])), X_.T), y_)
        b = y.mean(axis=0) - np.dot(X.mean(axis=0), w)
        w_ = np.append(w, b)
        return w_

    # split into training and testing sets and ten sets for cross validation
    def split(self, X, y):
        X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.1, random_state=64)
        rows = len(X_tr)
        row = math.ceil(rows/10)
        X_val = [X_tr[i:i+row] for i in range(0, len(X_tr), row)]
        y_val = [y_tr[i:i+row] for i in range(0, len(y_tr), row)]
        X_train = list([X_tr]*10)
        y_train = list([y_tr]*10)
        for i, j in zip(range(10), range(0, len(X_tr), row)):
            try:
                X_train[i] = np.delete(X_train[i], obj=slice(j, j+row), axis=0)
                y_train[i] = np.delete(y_train[i], obj=slice(j, j+row), axis=0)
            # the last set index is out of range
            except:
                X_train[i] = np.delete(X_train[i], obj=slice(j, -1), axis=0)
                y_train[i] = np.delete(y_train[i], obj=slice(j, -1), axis=0)
        return X_train, y_train, X_val, y_val, X_test, y_test

    # 10-fold cross validation
    def cross_validation(self, X_train, y_train, method='ridge', lamda=[0.01, 0.1, 0.5, 1, 2]):
        if method == 'linear':
            w_ = copy.deepcopy(y_train)
            for i in range(10):
                w_[i] = self.linear(X_train[i], y_train[i], sequence=i)
        else:
            w_ = [[self.ridge(X_train[i], y_train[i], x) for i in range(10)] for x in lamda]
        return w_

    # prediction
    def predict(self, X, w_, type='10-fold'):
        # w_ = (w, b).T
        # X_ = (X, 1)
        # y^ = X_ * w_
        # only one set of testing values
        if type == 'single':
            X_ = np.column_stack((X, np.ones(X.shape[0])))
            try:
                y_pred = np.dot(X_, w_)
                return y_pred
            # a singular matrix made w_ was 'NoneType'
            except TypeError:
                return None
        else:
            y_pred = copy.deepcopy(w_)
            for i in range(len(X)):
                X_ = np.column_stack((X[i], np.ones(X[i].shape[0])))
                try:
                    y_pred[i] = np.dot(X_, w_[i])
                # a singular matrix made w_[i] was 'NoneType'
                except TypeError:
                    y_pred[i] = None
                    continue
            return y_pred

    # mean square error
    def MSE(self, y_pred, y_true, type='10-fold'):
        # only one set of predicted values
        if type == 'single':
            try:
                num = len(y_pred)
                mse = sum((y_pred - y_true) ** 2) / num
                return mse
            # a singular matrix made y_pred was 'NoneType'
            except TypeError:
                return None
        else:
            mse = copy.deepcopy(y_pred)
            for i in range(len(y_pred)):
                try:
                    num = len(y_pred[i])
                    mse[i] = sum((y_pred[i] - y_true[i]) ** 2) / num
                # a singular matrix made y_pred[i] was 'NoneType'
                except TypeError:
                    mse[i] = None
                    continue
            return mse

    # R2 score
    def R2(self, y_pred, y_true, type='10-fold'):
        # only one set of predicted values
        if type == 'single':
            try:
                len(y_pred)
                r2 = 1 - (sum((y_pred - y_true) ** 2) / sum((y_true - np.mean(y_true)) ** 2))
                return r2
            # a singular matrix made y_pred was 'NoneType'
            except TypeError:
                return None
        else:
            r2 = copy.deepcopy(y_pred)
            for i in range(len(y_pred)):
                try:
                    len(y_pred[i])
                    r2[i] = 1 - (sum((y_pred[i] - y_true[i]) ** 2) / sum((y_true[i] - np.mean(y_true[i])) ** 2))
                # a singular matrix made y_pred[i] was 'NoneType'
                except TypeError:
                    r2[i] = None
                    continue
            return r2
    
    # determine the best hyper-parameter among lamda
    def hyper_parameter(self, X_train, y_train, X_val, y_val, lamda=[0.01, 0.1, 0.5, 1, 2]):
        w_ = self.cross_validation(X_train, y_train, lamda=lamda)
        y_pred = [self.predict(X_val, w_[i]) for i in range(len(lamda))]
        mse = [self.MSE(y_pred[i], y_val) for i in range(len(lamda))]
        mses = [np.mean(mse[i]) for i in range(len(lamda))]
        plt.figure(figsize=[8, 6])
        plt.plot(lamda, mses, 'x')
        plt.xlabel('lamda')
        plt.ylabel('mean aquare error (MSE)')
        plt.title('MSE with different lamda in ridge regression')
        plt.show(block=False)
        plt.pause(0.01)
        # find lamda with the minimum MSE
        best = (min(mses) == mses[:])
        loc = np.where(best)
        best_lamda = lamda[loc]
        min_mse = min(mses)
        print(f'\nThe best hyper-parameter in range [{lamda[0]}, {lamda[-1]}] in ridge regression is:')
        print(f'lamda={best_lamda} with MSE={min_mse}')
        return best_lamda, min_mse

    # the distribution of the target variable
    def y_dist(self, y):
        plt.figure(figsize=[8, 6])
        plt.hist(y, bins=20)
        plt.xlabel('MEDV')
        plt.title('Histogrm of the target variable distribution')
        plt.show(block=False)
        plt.pause(0.01)

    # the correlation matrix and plot the heatmap and dimensional correlation diagram
    def correlation(self):
        corr = self.data.corr()
        print('\nThe correlation matrix', corr, sep='\n')
        # heatmap
        plt.subplots(figsize=(6, 6))
        sns.heatmap(corr, vmin = -1, vmax = 1, annot=True, fmt='.2f', cmap = 'Oranges')
        plt.title('The correlation of variables and target in a heatmap')
        plt.show(block=False)
        plt.pause(0.01)
        # dimensional correlation diagram
        strong = ['RM', 'LSTAT', 'MEDV']
        sns.pairplot(self.data[strong], size=2
        ).fig.suptitle('The dimensional correlation diagram for highly correlated attributes')
        plt.show(block=False)
        plt.pause(0.01)

    # show and evaluate the model with a sheet
    def sheet(self, w_, train_mse, test_mse, train_r2, test_r2, name):
        columns = ['Model', 'Train MSE', 'Test MSE', 'Train R2', 'Test R2']
        df = [[], [], [], [], [], [], [], [], [], []]
        for i in range(10):
            model = '%.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f*%s + %.2f' %(w_[i][0],self.data.columns[0],
            w_[i][1],self.data.columns[1], w_[i][2],self.data.columns[2], w_[i][3],self.data.columns[3], w_[i][4],self.data.columns[4], w_[i][5],self.data.columns[5], 
            w_[i][6],self.data.columns[6], w_[i][7],self.data.columns[7], w_[i][8],self.data.columns[8], w_[i][9],self.data.columns[9], w_[i][10],self.data.columns[10], 
            w_[i][11],self.data.columns[11], w_[i][12],self.data.columns[12], w_[i][13])
            values = [model, train_mse[i], test_mse[i], train_r2[i], test_r2[i]]
            df[i] = pd.DataFrame([values], columns=columns, index=[name+''+str(i+1)])
        sheet = pd.concat([df[j] for j in range(10)])
        # determine the mean and add to the sheet
        mean_train_mse = np.mean(train_mse)
        mean_test_mse = np.mean(test_mse)
        mean_train_r2 = np.mean(train_r2)
        mean_test_r2 = np.mean(test_r2)
        mean_ = pd.DataFrame([('', mean_train_mse, mean_test_mse, mean_train_r2, mean_test_r2)], columns=columns, index=['mean'])
        sheet = pd.concat([sheet, mean_])
        return sheet

    # find the analytic target
    def max_min(self, mse):
        maximum = (max(mse) == mse[:])
        max_loc = np.where(maximum)
        minimum = (min(mse) == mse[:])
        min_loc = np.where(minimum)
        # convert to integer
        max_loc = list(max_loc[0])[0].tolist()
        min_loc = list(min_loc[0])[0].tolist()
        return max_loc, min_loc

    # plot the graph betweem predicted and true values
    def scatter(self, y_pred, y_true, mse, title):
        # plot the scatter of predicted and true values
        num = len(y_true)
        plt.figure(figsize=[12, 5])
        plt.scatter(np.arange(num), y_pred, marker='x', color='blue', label='y_pred')
        plt.scatter(np.arange(num), y_true, color='red', label='y_true')
        plt.legend(loc=0)
        plt.title('Scatter of predicted and true values' + title)
        plt.show(block=False)
        plt.pause(0.01)
        # plot the true values line and predicted point
        plt.figure(figsize=[8, 6])
        plt.scatter(y_true, y_pred, label='predict')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=3, label='true')
        plt.text(y_true.max()*0.8, y_pred.min(), 'MSE=%.2f' % mse, fontsize=15)
        plt.legend(loc=0)
        plt.title('Predicted point with the true line' + title)
        plt.show(block=False)
        plt.pause(0.01)


# load and generate the data
boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target

# instantiation
model = LinerRegression(df)
# normalization
X, y = model.divide()
# plot the target distribution
model.y_dist(y)
# correlation between features and target
model.correlation()
# split into training and testing sets and ten training and validation sets
X_train, y_train, X_val, y_val, X_test, y_test = model.split(X, y)

# linear regression
w_linear = model.cross_validation(X_train, y_train, method='linear')
# determine the best hyper-parameter
best_lamda, _ = model.hyper_parameter(X_train, y_train, X_val, y_val, lamda=np.linspace(0, 0.08, 161))
# ridge regression
w_ridge = model.cross_validation(X_train, y_train, lamda=best_lamda)

# predition
linear_pred_test = model.predict([X_test]*10, w_linear)
linear_pred_train = model.predict(X_train, w_linear)
ridge_pred_test = model.predict([X_test]*10, w_ridge[0])
ridge_pred_train = model.predict(X_train, w_ridge[0])
# evaluation
linear_test_mse = model.MSE(linear_pred_test, [y_test]*10)
linear_train_mse = model.MSE(linear_pred_train, y_train)
ridge_test_mse = model.MSE(ridge_pred_test, [y_test]*10)
ridge_train_mse = model.MSE(ridge_pred_train, y_train)
linear_test_r2 = model.R2(linear_pred_test, [y_test]*10)
linear_train_r2 = model.R2(linear_pred_train, y_train)
ridge_test_r2 = model.R2(ridge_pred_test, [y_test]*10)
ridge_train_r2 = model.R2(ridge_pred_train, y_train)
# show the result in a sheet
linear_sheet = model.sheet(w_linear, linear_train_mse, linear_test_mse, linear_train_r2, linear_test_r2, name='linear')
ridge_sheet = model.sheet(w_ridge[0], ridge_train_mse, ridge_test_mse, ridge_train_r2, ridge_test_r2, name='ridge')
# # store in excel to access a better view
# with pd.ExcelWriter('./model.xlsx') as writer:
#     linear_sheet.to_excel(writer, sheet_name='linear')
#     ridge_sheet.to_excel(writer, sheet_name='ridge')

# select some results to analysis, choose the sets with the maximum and minimum MSE
linear_test_max, linear_test_min = model.max_min(linear_test_mse)
linear_train_max, linear_train_min = model.max_min(linear_train_mse)
ridge_test_max, ridge_test_min = model.max_min(ridge_test_mse)
ridge_train_max, ridge_train_min = model.max_min(ridge_train_mse)

# select the minimum MSE on the testing set and predict on the whole set
w_linear_min = w_linear[linear_test_min]
w_ridge_min = w_ridge[0][ridge_test_min]
linear_pred_whole = model.predict(X, w_linear_min, type='single')
ridge_pred_whole = model.predict(X, w_ridge_min, type='single')
linear_whole_mse = model.MSE(linear_pred_whole, y, type='single')
ridge_whole_mse = model.MSE(ridge_pred_whole, y, type='single')

# plot the scatter and add titles
# linear test max,min
model.scatter(linear_pred_test[linear_test_max], y_test, linear_test_mse[linear_test_max],
title=' (linear model on testing set with maximum MSE)')
model.scatter(linear_pred_test[linear_test_min], y_test, linear_test_mse[linear_test_min],
title=' (linear model on testing set with minimum MSE)')
# linear train max,min
model.scatter(linear_pred_train[linear_train_max], y_train[linear_train_max], linear_train_mse[linear_train_max],
title=' (linear model on training set with maximum MSE)')
model.scatter(linear_pred_train[linear_train_min], y_train[linear_train_min], linear_train_mse[linear_train_min],
title=' (linear model on training set with minimum MSE)')
# ridge test max,min
model.scatter(ridge_pred_test[ridge_test_max], y_test, ridge_test_mse[ridge_test_max],
title=' (ridge model on testing set with maximum MSE)')
model.scatter(ridge_pred_test[ridge_test_min], y_test, ridge_test_mse[ridge_test_min],
title=' (ridge model on testing set with minimum MSE)')
# ridge train max,min
model.scatter(ridge_pred_train[ridge_train_max], y_train[ridge_train_max], ridge_train_mse[ridge_train_max],
title=' (ridge model on training set with maximum MSE)')
model.scatter(ridge_pred_train[ridge_train_min], y_train[ridge_train_min], ridge_train_mse[ridge_train_min],
title=' (ridge model on training set with minimum MSE)')
# linear whole
model.scatter(linear_pred_whole, y, linear_whole_mse, title=' (best linear model on the whole set)')
# ridge whole
model.scatter(ridge_pred_whole, y, ridge_whole_mse, title=' (best ridge model on the whole set)')

# A diagram of the difference between the two regression
plt.figure(figsize=[14, 8])
num = len(y)
plt.plot(np.linspace(0, num-1, num), linear_pred_whole, linewidth=3, label='linear')
plt.plot(np.linspace(0, num-1, num), ridge_pred_whole, ':', linewidth=3, label='ridge')
plt.legend(loc=0)
plt.xlabel('target set')
plt.title('The difference between two models')
plt.pause(0.01)
plt.show() 