from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import warnings
from ml import ML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

class ClassificationML(ML):
    def __init__(self):
        super().__init__()
        self.classifiers = [
            KNeighborsClassifier(),
            GaussianProcessClassifier(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
        ]

    def train_ML(self, ml_name):
        for clf in self.classifiers:
            name = type(clf).__name__
            if ml_name == name:
                ml = clf.fit(self.X_train, self.y_train)
                return ml

    def perform_ML(self):
        print(f'#==============================================================#')
        print(f'Training size[{len(self.y_train)}], Test size[{len(self.y_test)}]')
        print(f'#==============================================================#')
        print('Mean Accuracy Score:')

        # iterate over classifiers
        results = {}
        for clf in self.classifiers:
            name = type(clf).__name__
            info = ''
            if name == 'SVC':
                info = clf.kernel

            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)

            predicted = clf.predict(self.X_test)
            f1 = f1_score(self.y_test, predicted, average='micro')
            cm = confusion_matrix(self.y_test, predicted)
            cr = classification_report(self.y_test, predicted)
            print(f'{name} {info}\t')
            print(cr)

            results[name] = score

        return results