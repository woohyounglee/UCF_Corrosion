
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from regressionML import RegressionML
from statistics import stdev
from statistics import mean
from saveResults import SaveResults
from datetime import datetime

import winsound
import os

import numpy as np
import matplotlib.pyplot as plt

# penetration_free chlorine
class PenetrationPrediction():
    def __init__(self):
        # Sets configuration
        self.cf = {}
        self.cf['sheet'] = 'TestData'
        self.cf['selected_Xs'] = ['Z (µm)', 'Time']
        self.cf['distance'] = 'Z (µm)'
        self.cf['target'] = 'Concentration'

        self.size_experiments = 2
        # self.size_experiments = 5

        # Define excel data for storing all data to Excel
        self.excel = {}
        self.excel['Data'] = []
        self.excel['Target'] = []
        self.excel['Normalization'] = []
        self.excel['Train Size'] = []
        self.excel['Test Size'] = []

        ml_algs = ['RFR', 'DTR', 'GBR']
        self.ml = RegressionML(ml_algs)

        for clf in self.ml.regressors:
            name = self.ml.get_short_name(type(clf).__name__)
            self.excel[name + '-MAE'] = []
        for clf in self.ml.regressors:
            name = self.ml.get_short_name(type(clf).__name__)
            self.excel[name + '-MAE-STD'] = []

        self.excel['X Variables'] = []
        self.excel['train_clusters']= []
        self.excel['test_clusters']= []

        # set model info
        for v in self.cf['selected_Xs']:
            self.excel['X Variables'].append(v)


    def append_excel_column(self):
        self.excel['Data'].append('Data')
        self.excel['Target'].append('Target')
        self.excel['Normalization'].append('Normalization')
        self.excel['Train Size'].append('Train Size')
        self.excel['Test Size'].append('Test Size')
        self.excel['X Variables'].append('X Variables')
        self.excel['train_clusters'].append('train_clusters')
        self.excel['test_clusters'].append('test_clusters')

        for clf in self.ml.regressors:
            name = self.ml.get_short_name(type(clf).__name__)
            self.excel[name + '-MAE'].append(name + '-MAE')
        for clf in self.ml.regressors:
            name = self.ml.get_short_name(type(clf).__name__)
            self.excel[name + '-MAE-STD'].append(name + '-MAE-STD')


    def save_excel_file(self):
        experiment_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        # excel_file = f"../data_result/[Result]{self.cf['base_name']}"
        excel_file = f"../data_result/[Result]{experiment_time}.xlsx"
        excel_experiment = SaveResults(excel_file)
        for k, l in self.excel.items():
            excel_experiment.insert(k, l)
        excel_experiment.save()

    def run(self, f):
        base_name = os.path.basename(f)
        print(base_name)

        self.cf['input_file'] = f
        self.cf['base_name'] = base_name

        # 1. Loads data from xlsx
        excel_data = pd.read_excel(self.cf['input_file'], self.cf['sheet'])

        # 1. Add previous variables
        # excel_data['Z (µm)-1'] = excel_data['Z (µm)'].shift()
        # self.cf['selected_Xs'].append('Z (µm)-1')
        # excel_data['Z (µm)-2'] = excel_data['Z (µm)-1'].shift()
        # self.cf['selected_Xs'].append('Z (µm)-2')

        # 2. Selects variables
        excel_data = excel_data[self.cf['selected_Xs'] + [self.cf['target']]]

        excel_data = excel_data.dropna()

        # 3. Removes all duplicated
        excel_data = excel_data.drop_duplicates()

        # 5. Normalization is performed
        # scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(excel_data)
        # excel_data = pd.DataFrame(scaled_df, index=excel_data.index, columns=excel_data.columns)

        # 5. Split into training and test part according to cases
        time_clusters = excel_data['Time'].unique().tolist()

        for time_cluster in time_clusters:
            train_clusters = time_clusters.copy()
            train_clusters.remove(time_cluster)
            test_clusters = []
            test_clusters.append(time_cluster)

            # df_train = excel_data
            df_train = excel_data[excel_data['Time'].isin(train_clusters)]
            df_test = excel_data[excel_data['Time'].isin(test_clusters)]

            # 6. Separate Xs and y
            X_train, y_train = df_train[self.cf['selected_Xs']], df_train[self.cf['target']]
            X_test, y_test = df_test[self.cf['selected_Xs']], df_test[self.cf['target']]

            # 9. Perform several ML experiments
            sum_results = None
            all_results = {}
            for i in range(self.size_experiments):

                # 13. Set data X and y for ML
                self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                # 14. Perform ML
                results, results_predicted = self.ml.perform_ML_predicted_results()

                if len(all_results) == 0:
                    all_results = {x: [v] for x, v in results.items()}
                else:
                    for x, v in all_results.items():
                        for x2, v2 in results.items():
                            if x2 == x:
                                v.append(v2)

                if sum_results is None:
                    sum_results = results
                else:
                    sum_results = {x: v + v2 for x, v in sum_results.items() for x2, v2 in results.items() if x2 == x}

            # 15. Set all results for the excel output
            for clf in self.ml.regressors:
                name = self.ml.get_short_name(type(clf).__name__)
                # self.excel[name + '-MAE'].append(avg_results[name])
                self.excel[name + '-MAE'].append(round(mean(all_results[name]), 4))
                self.excel[name + '-MAE-STD'].append(round(stdev(all_results[name]), 4))

            self.excel['Data'].append(self.cf['base_name'])
            self.excel['Normalization'].append('True')
            self.excel['Train Size'].append(len(X_train))
            self.excel['Test Size'].append(len(X_test))
            self.excel['train_clusters'].append(', '.join(map(str, train_clusters)))
            self.excel['test_clusters'].append(', '.join(map(str, test_clusters)))

            # Draw plots for the actual cases
            r1 = X_test.iloc[0,:]['Z (µm)']
            r2 = X_test.iloc[-1,:]['Z (µm)']
            x = np.linspace(r1, r2, len(X_test['Z (µm)']))
            y = y_test

            fig, ax = plt.subplots()

            # Using set_dashes() to modify dashing of an existing line
            ax.plot(x, y, label='Actual Data')
            # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

            # Draw plots for the predicted cases

            index = 1
            for k, predicted in results_predicted.items():
                # Using plot(..., dashes=...) to set the dashing when creating a line
                score = self.excel[k + '-MAE'][-1]
                ax.plot(x, predicted, dashes=[6, index], label=f'{k}({score})')
                index += 2

            # Include the plot information
            plt.title(f'{time_cluster} hours')
            ax.legend()

            # plt.show()
            plt.savefig(f"../data_saved_image/predicted_penetration_{time_cluster}.png")
            plt.close()
            print("Done!")


file = '../data/penetration_free chlorine.xlsx'

ex = PenetrationPrediction()
ex.run(file)
ex.append_excel_column()


ex.save_excel_file()

winsound.Beep(1000, 440)

