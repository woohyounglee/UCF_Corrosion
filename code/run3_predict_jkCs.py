
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


class JKCsPrediction():
    def __init__(self):
        # Sets configuration
        self.cf = {}
        self.cf['sheet'] = 'Sheet1'
        # self.cf['selected_Xs'] = ['DIC', 'pH', 'Phosphate']
        self.cf['selected_Xs'] = ['Depth (µm)', 'Concentration', 'DIC', 'pH', 'Phosphate']
        self.cf['distance'] = 'Depth (µm)'
        # Concentration

        # self.size_experiments = 2
        self.size_experiments = 5

        # Define excel data for storing all data to Excel
        self.excel = {}
        self.excel['Data'] = []
        self.excel['Target'] = []
        self.excel['Normalization'] = []
        self.excel['Train Size'] = []
        self.excel['Test Size'] = []

        self.ml = RegressionML()
        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE'] = []
        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE-STD'] = []

        self.excel['X Variables'] = []

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

        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE'].append(name + '-MAE')
        for clf in self.ml.regressors:
            name = type(clf).__name__
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

        if 'pH' in base_name:
            self.cf['targets'] = ['pH_output']
        else:
            self.cf['targets'] = ['Cs', 'J', 'K']

        for target in self.cf['targets']:
            self.cf['target'] = target

            # 1. Loads data from xlsx
            excel_data = pd.read_excel(self.cf['input_file'], self.cf['sheet'])

            # 2. Selects variables
            excel_data = excel_data[self.cf['selected_Xs'] + [self.cf['target']]]

            # 3. Removes all duplicated
            excel_data = excel_data.drop_duplicates()

            # 5. Normalization is performed
            scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(excel_data)
            excel_data = pd.DataFrame(scaled_df, index=excel_data.index, columns=excel_data.columns)

            # 6. Separate Xs and y
            df_X, df_y = excel_data[self.cf['selected_Xs']], excel_data[self.cf['target']]

            # 9. Perform several ML experiments
            sum_results = None
            all_results = {}
            for i in range(self.size_experiments):
                # 9. Split into training and test part
                X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2)

                # 13. Set data X and y for ML
                self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                # 14. Perform ML
                results = self.ml.perform_ML()

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
                name = type(clf).__name__
                # self.excel[name + '-MAE'].append(avg_results[name])
                self.excel[name + '-MAE'].append(round(mean(all_results[name]), 4))
                self.excel[name + '-MAE-STD'].append(round(stdev(all_results[name]), 4))

            self.excel['Data'].append(self.cf['base_name'])
            self.excel['Normalization'].append('True')
            self.excel['Train Size'].append(len(X_train))
            self.excel['Test Size'].append(len(X_test))
            self.excel['Target'].append(target)


files = ['../data_jkCs/data_free_chlorine_jkCs_[DO-Copper].xlsx',
         '../data_jkCs/data_free_chlorine_jkCs_[DO-Ductile].xlsx',
         '../data_jkCs/data_free_chlorine_jkCs_[Free Chlorine-Copper].xlsx',
         '../data_jkCs/data_free_chlorine_jkCs_[Free Chlorine-Ductile].xlsx',
         '../data_jkCs/data_free_chlorine_jkCs_[pH-Copper].xlsx',
         '../data_jkCs/data_free_chlorine_jkCs_[pH-Ductile].xlsx',
         '../data_jkCs/data_monochloramine_jkCs_[DO-Copper].xlsx',
         '../data_jkCs/data_monochloramine_jkCs_[DO-Ductile].xlsx',
         '../data_jkCs/data_monochloramine_jkCs_[Monochloramine-Copper].xlsx',
         '../data_jkCs/data_monochloramine_jkCs_[Monochloramine-Ductile].xlsx',
         '../data_jkCs/data_monochloramine_jkCs_[pH-Copper].xlsx',
         '../data_jkCs/data_monochloramine_jkCs_[pH-Ductile].xlsx',
        ]

ex = JKCsPrediction()

for f in files:
    ex.run(f)
    ex.append_excel_column()

ex.save_excel_file()

winsound.Beep(1000, 440)

