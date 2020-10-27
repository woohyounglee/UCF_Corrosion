
import pandas as pd
from datetime import datetime
import winsound
import os
from regressionML import RegressionML
from statistics import stdev
from statistics import mean
from saveResults import SaveResults


class JKCsPredictionUnknownX():
    def __init__(self, file):
        self.time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        self.file = file
        self.ml = RegressionML()
        self.size_experiments = 5
        self.cf = {}
        self.cf['selected_Xs'] = ['DIC', 'pH', 'Phosphate']
        self.cf['targets'] = ['Cs', 'J', 'K']

        self.excel = {}
        self.excel['File'] = []
        self.excel['Experiment'] = []
        self.excel['Target'] = []
        self.excel['Train Size'] = []
        self.excel['Test Size'] = []

        self.ml = RegressionML()

        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE-AVG'] = []

        for clf in self.ml.regressors:
            name = type(clf).__name__
            self.excel[name + '-MAE-STD'] = []

    def save_excel_file(self):
        excel_file = f"../data_result/[predicted_from_unknown_X_{self.unknow_x}]{self.time}.xlsx"
        excel_experiment = SaveResults(excel_file)
        for k, l in self.excel.items():
            excel_experiment.insert(k, l)
        excel_experiment.save()

    def run(self, unknow_x=True):
        self.unknow_x = unknow_x

        path = "../data_output"
        os.chdir(path)

        df_list = []
        print(self.file)

        # 1. Loads data from xlsx
        self.excel_data = pd.read_excel(path+'/'+self.file, 'Sheet1')

        # Takes a experiment list
        categories = self.excel_data['output_file'].unique()

        # For each experiment
        for c in categories:
            # Takes an experiment data
            df_cur_ex = self.excel_data[(self.excel_data['output_file'] == c)]

            # Takes a experiment list
            sub_experiments = df_cur_ex['sheet'].unique()

            for ex in sub_experiments:
                # Takes an experiment data
                df_sub_ex = df_cur_ex[(df_cur_ex['sheet'] == ex)]
                print(c, ":", ex)

                df_train, df_test = None, None

                # Takes a list
                groups = df_sub_ex['Type'].unique()

                # For each experiment
                for target in self.cf['targets']:
                    self.cf['target'] = target

                    sum_results = None
                    all_results = {}

                    # unknow_x True means that this one of eight groups is used for testing and remaining groups are used for training
                    # unknow_x False means that all data are shuffled and randomly selected for test and training
                    if unknow_x is True:
                        for g in groups:
                            df_test = df_sub_ex[(df_sub_ex['Type'] == g)]
                            df_train = df_sub_ex[(df_sub_ex['Type'] != g)]

                            # 5. Normalization is performed
                            # scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_sub_ex)
                            # df_sub_ex = pd.DataFrame(scaled_df, index=df_sub_ex.index, columns=df_sub_ex.columns)

                            # 6. Separate Xs and y
                            X_train, y_train = df_train[self.cf['selected_Xs']], df_train[self.cf['target']]
                            X_test, y_test = df_test[self.cf['selected_Xs']], df_test[self.cf['target']]

                            # 13. Set data X and y for ML
                            self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                            # 14. Perform ML
                            results = self.ml.perform_ML()
                            print(results)

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
                    elif unknow_x is False:
                        df_X, df_y = df_sub_ex[self.cf['selected_Xs']], df_sub_ex[self.cf['target']]

                        for i in range(self.size_experiments):
                            # 9. Split into training and test part
                            # X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2)
                            X_train, X_test, y_train, y_test = df_X, df_X, df_y, df_y

                            # 13. Set data X and y for ML
                            self.ml.set_train_test_data(X_train, X_test, y_train, y_test)

                            # 14. Perform ML
                            results = self.ml.perform_ML()
                            print(results)

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
                        self.excel[name + '-MAE-AVG'].append(round(mean(all_results[name]), 14))
                        self.excel[name + '-MAE-STD'].append(round(stdev(all_results[name]), 14))

                    self.excel['File'].append(c)
                    self.excel['Experiment'].append(ex)
                    self.excel['Train Size'].append(len(X_train))
                    self.excel['Test Size'].append(len(X_test))
                    self.excel['Target'].append(target)


file = 'summary_10_10_2020-11_55_48.xlsx'

ex = JKCsPredictionUnknownX(file)
# ex.run(unknow_x=True)
ex.run(unknow_x=False)
ex.save_excel_file()

winsound.Beep(1000, 440)

