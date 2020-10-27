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
from deep_learning_regressor import DeepLearningRegressor
# from gpflow_regressor import GPFLOWRegressor
from ml import ML
from multiprocessing import Process, Queue
import pandas as pd
import numpy as np
import datetime


class RegressionML(ML):
    def __init__(self):
        super().__init__()
        self.regressors = [
            DeepLearningRegressor(type='custom'),
            linear_model.BayesianRidge(),
            RandomForestRegressor(n_estimators=100),
            DecisionTreeRegressor(),
            GradientBoostingRegressor(),
            # MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.001, batch_size='auto',
            #     learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            #     random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            #     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
            LinearRegression(),
            GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(), random_state=0),
            SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        ]

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

