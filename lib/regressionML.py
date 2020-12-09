from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from lightgbm_sklearn import LightGBM
# from deep_learning_regressor import DeepLearningRegressor
# from gpflow_regressor import GPFLOWRegressor
from ml import ML
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import datetime


class RegressionML(ML):
    def __init__(self, ml_algs=['LR', 'GPR', 'MLP', 'DL', 'SVR', 'RFR', 'DTR', 'GBR']):
        super().__init__()
        self.regressors = []

        for alg in ml_algs:
            # if alg == 'DL':
            #     self.regressors.append(DeepLearningRegressor(type='custom'))
            if alg == 'BRR':
                self.regressors.append(linear_model.BayesianRidge())
            elif alg == 'RFR':
                self.regressors.append(RandomForestRegressor(n_estimators=100))
            elif alg == 'DTR':
                self.regressors.append(DecisionTreeRegressor())
            elif alg == 'GBR':
                self.regressors.append(GradientBoostingRegressor())
            elif alg == 'LR':
                self.regressors.append(LinearRegression())
            elif alg == 'GPR':
                self.regressors.append(GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0))
            elif alg == 'SVR':
                self.regressors.append(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
            elif alg == 'MLP':
                self.regressors.append(MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

    def train_ML(self, ml_name):
        for clf in self.regressors:
            name = type(clf).__name__
            if ml_name == name:
                ml = clf.fit(self.X_train, self.y_train)
                return ml

    def perform_ML(self, file_save=None):
        print(f'#==============================================================#')
        print(f'Training size[{len(self.y_train)}], Test size[{len(self.y_test)}]')
        print(f'#==============================================================#')
        print('Score:')

        # iterate over classifiers
        results = {}
        for clf in self.regressors:
            name = type(clf).__name__
            info = ''
            if name == 'SVC':
                info = clf.kernel
            if name == 'DeepLearningRegressor':
                info = clf.type

            clf.fit(self.X_train, self.y_train)
            # score = clf.score(self.X_test, self.y_test)

            predicted = clf.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, predicted)

            print(f'{name} {info}\t', mae)
            results[name] = mae

            if file_save is not None:
                df = pd.DataFrame({'Actual': self.y_test, 'Predicted': predicted.ravel()})
                df.to_csv(f'{file_save}_{name}_{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv')

        return results

    def get_short_name(self, name):
        if name == 'SVR':
            return 'SVR'
        elif name == 'DeepLearningRegressor':
            return 'DLR'
        elif name == 'GradientBoostingRegressor':
            return 'GBR'
        elif name == 'DecisionTreeRegressor':
            return 'DTR'
        elif name == 'LinearRegression':
            return 'LR'
        elif name == 'GaussianProcessRegressor':
            return 'GPR'
        elif name == 'MLPRegressor':
            return 'MLP'
        elif name == 'BayesianRidge':
            return 'BRR'
        elif name == 'RandomForestRegressor':
            return 'RFR'

    def perform_ML_predicted_results(self, file_save=None):
        print(f'#==============================================================#')
        print(f'Training size[{len(self.y_train)}], Test size[{len(self.y_test)}]')
        print(f'#==============================================================#')
        print('Score:')

        # iterate over classifiers
        results = {}
        results_predicted = {}
        for clf in self.regressors:
            name = self.get_short_name(type(clf).__name__)

            clf.fit(self.X_train, self.y_train)
            # score = clf.score(self.X_test, self.y_test)

            predicted = clf.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, predicted)

            results[name] = mae
            results_predicted[name] = predicted

            if file_save is not None:
                df = pd.DataFrame({'Actual': self.y_test, 'Predicted': predicted.ravel()})
                df.to_csv(f'{file_save}_{name}_{datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.csv')

        return results, results_predicted

