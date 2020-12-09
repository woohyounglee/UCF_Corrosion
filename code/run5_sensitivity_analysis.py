from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from saveResults import SaveResults
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import winsound
import os


class FeatureImportance():
    def __init__(self):
        # Sets configuration
        self.cf = {}
        self.cf['sheet'] = 'Sheet1'
        # self.cf['selected_Xs'] = ['DIC', 'pH', 'Phosphate']
        self.cf['selected_Xs'] = ['Depth (µm)', 'Concentration', 'DIC', 'pH', 'Phosphate']
        self.cf['distance'] = 'Depth (µm)'
        self.cf['removal_targets'] = ['DIC', 'pH', 'Phosphate']
        # Concentration
        self.ml = RandomForestRegressor(n_estimators=100)

        # self.size_experiments = 2
        self.size_experiments = 5

        # Define excel data for storing all data to Excel
        self.excel = {}
        self.excel['Data'] = []
        self.excel['Target'] = []
        self.excel['Train Size'] = []
        self.excel['Test Size'] = []

        for x in self.cf['selected_Xs']:
            self.excel[x] = []

        self.excel['X Variables'] = []

        # set model info
        for v in self.cf['selected_Xs']:
            self.excel['X Variables'].append(v)

    def append_excel_column(self):
        self.excel['Data'].append('Data')
        self.excel['Target'].append('Target')

        self.excel['Train Size'].append('Train Size')
        self.excel['Test Size'].append('Test Size')
        self.excel['X Variables'].append('X Variables')

        for x in self.cf['selected_Xs']:
            self.excel[x].append(x)


    def save_excel_file(self):
        experiment_time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        # excel_file = f"../data_result/[Result]{self.cf['base_name']}"
        excel_file = f"../data_sensitivity_analysis/[Importance_Feature_Result]{experiment_time}.xlsx"
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

            selected_Xs = self.cf['selected_Xs'].copy()

            # 2. Selects variables
            excel_data = excel_data[selected_Xs + [self.cf['target']]]

            # 3. Removes all duplicated
            excel_data = excel_data.drop_duplicates()

            # 5. Normalization is performed
            scaled_df = MinMaxScaler(feature_range=(0, 1)).fit_transform(excel_data)
            excel_data = pd.DataFrame(scaled_df, index=excel_data.index, columns=excel_data.columns)

            # 6. Separate Xs and y
            df_X, df_y = excel_data[selected_Xs], excel_data[self.cf['target']]


            # 9. Split into training and test part
            X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=.2)

            # Train model
            model = self.ml.fit(X_train, y_train)

            # Calculate feature importances
            importance = self.ml.feature_importances_
            importance = pd.DataFrame(importance, index=selected_Xs, columns=["Importance"])

            print(importance)

            # 15. Set all results for the excel output
            for x in self.cf['selected_Xs']:
                self.excel[x].append(round(importance.iloc[:, x][0], 4))

            self.excel['Data'].append(self.cf['base_name'])
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

ex = FeatureImportance()

for f in files:
    ex.run(f)
    ex.append_excel_column()

ex.save_excel_file()

winsound.Beep(1000, 440)

