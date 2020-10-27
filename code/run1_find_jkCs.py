import pandas as pd
import math
import winsound
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime


class JKCsFinder():
    def __init__(self, input_file, output_file, sheet):
        self.time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        # Sets configuration
        self.cf = {}
        self.cf['input_file'] = input_file
        self.cf['output_file'] = output_file
        self.cf['sheet'] = sheet

        self.cf['targets'] = ['OV', 'Oil_separation(%)']
        self.cf['selected_Xs'] = ['Depth (µm)']
        self.cf['distance'] = 'Depth (µm)'

        if 'Free Chlorine' in self.cf['sheet']:
            self.cf['difussion_coefficient'] = 0.0000144
        elif 'Monochloramine' in self.cf['sheet']:
            self.cf['difussion_coefficient'] = 0.0000166
        elif 'DO' in self.cf['sheet']:
            self.cf['difussion_coefficient'] = 0.0000117

    def perform_ML(self, r1, r2):
        print(r1, r2)

        # find an outside range
        # [Reason] In some cases, only two data are selected, because of a small range
        # e.g.,)
        # In: [-550, -400, -350, -300, -250, -200, -150], [-380, -330]
        # Out: [-400, -350, -300]
        # Process:
        #  x1,  x2      x       pre_x     r1      r2
        # -380, -330    -550
        # -380, -330    -400    -550
        # -380, -330    -350
        # -380, -330    -300
        # -380, -330    -250
        x1, x2 = r1, r2
        pre_x = -10000
        for x in self.df_cur_ex['Depth (µm)'].to_list():
            # if x1 == x:
            #     r1 = x1
            if pre_x < r1 and r1 < x:
                x1 = pre_x

            # if x2 == x:
            #     r2 = x2
            if pre_x < r2 and r2 < x:
                x2 = x

            pre_x = x

        print('Selected', x1, x2)
        df_train = self.df_cur_ex[self.df_cur_ex['Depth (µm)'].between(int(x1), int(x2))]

        X_train = df_train['Depth (µm)'].to_numpy().reshape(-1, 1)
        y_train = df_train['Concentration'].to_numpy().reshape(-1, 1)

        if len(y_train) == 0:
            print(f'!!! Data size [{x1, x2}]is zero !!!')
            return None, None

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        predicted = lr.predict(X_train)
        r_square = r2_score(y_train, predicted)
        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]

        return r_square, slope, intercept, x1, x2

    def find_slope(self, min, max, window):
        df_ranges = pd.DataFrame(columns=['r1', 'r2', 'x1', 'x2', 'r_square', 'slope', 'intercept'])

        for s in range((max-min)):
            r1 = min + s
            r2 = r1 + window
            if r2 > max:
                break

            r_square, slope, intercept, x1, x2 = self.perform_ML(r1, r2)
            # print(f'[{r1, r2}]: {r_square, slope, intercept}')

            df_ranges = df_ranges.append({'r1': r1, 'r2': r2, 'x1': x1, 'x2': x2, 'r_square': r_square, 'slope': math.fabs(slope), 'intercept':intercept}, ignore_index=True)

        # changed 10/9/2020
        # # Find the highest r square row
        # df_highest_r2 = df_ranges[df_ranges['r_square'] == df_ranges['r_square'].max()][0:1]
        #
        # return df_highest_r2 # 0.95 > and

        # new approach to find the highest range 10/9/2020
        # Find 0.95 > r square and highest slope
        # df_highest_r2 = df_ranges[df_ranges['r_square'] > 0.95]
        df_highest_r2 = df_ranges[df_ranges['r_square'] > 0.88]
        df_highest_r2 = df_highest_r2[df_highest_r2['slope'] == df_highest_r2['slope'].max()][0:1]

        return df_highest_r2

    def run_for_FC_MN_DO(self):

        # Takes a experiment list
        experiments = self.excel_data['Experiment'].unique()

        # For each experiment
        df_list = []
        for ex in experiments:
            # Check Error
            if self.cf['output_file'] == 'data_free_chlorine_jkCs' and self.cf['sheet'] == 'Free Chlorine-Ductile' and ex == 9:
                # winsound.Beep(1000, 440)
                print('error')

            print(self.cf['output_file'], self.cf['sheet'], ex)

            # Takes an experiment data
            self.df_cur_ex = self.excel_data[(self.excel_data['Experiment'] == ex)]

            # Finds a slope of a high-scored R2 and slope
            df_highest_r2= self.find_slope(-400, 0, 200) # 10/10/2020
            # df_highest_r2 = self.find_slope(-400, 0, 140) # 10/7/2020

            r1 = df_highest_r2['r1'].values[0]
            r2 = df_highest_r2['r2'].values[0]
            x1 = df_highest_r2['x1'].values[0]
            x2 = df_highest_r2['x2'].values[0]
            r_square = df_highest_r2['r_square'].values[0]
            slope = df_highest_r2['slope'].values[0]
            intercept = df_highest_r2['intercept'].values[0]

            print(slope, intercept)

            # Finds a Cs value
            Cs = intercept

            # Cs calculation rules
            if 'Free Chlorine' in self.cf['sheet'] and Cs < 0.05:
                Cs = 0.05
            elif 'Monochloramine' in self.cf['sheet'] and Cs < 0.05:
                Cs = 0.05
            elif 'DO' in self.cf['sheet'] and Cs < 0.06:
                Cs = 0.06

            # Finds a J value
            J = self.cf['difussion_coefficient'] * slope * 10

            # Finds a K value
            K = (J / Cs) * 1000

            self.df_cur_ex['Cs'] = Cs
            self.df_cur_ex['J'] = J
            self.df_cur_ex['K'] = K

            self.df_cur_ex['r1'] = r1
            self.df_cur_ex['r2'] = r2
            self.df_cur_ex['x1'] = x1
            self.df_cur_ex['x2'] = x2
            self.df_cur_ex['r_square'] = r_square
            self.df_cur_ex['slope'] = slope
            self.df_cur_ex['intercept'] = intercept

            # Adds file info
            self.df_cur_ex['output_file'] = self.cf['output_file']
            self.df_cur_ex['sheet'] = self.cf['sheet']

            df_list.append(self.df_cur_ex)

        # Stacks all results
        df_vertical_stack = pd.concat(df_list, axis=0)

        # Saves all results as one file
        df_vertical_stack.to_excel(f"../data_jkCs/{self.cf['output_file']}_[{self.cf['sheet']}].xlsx")

    def run_for_pH(self):

        # Takes a experiment list
        experiments = self.excel_data['Experiment'].unique()

        # For each experiment
        df_list = []
        for ex in experiments:
            # Takes an experiment data
            self.df_cur_ex = self.excel_data[(self.excel_data['Experiment'] == ex)]

            # Finds df in 'Depth (µm)' <= -30
            df_filtered_ex = self.df_cur_ex[self.df_cur_ex['Depth (µm)'] <= -30]

            # Adds the output_pH
            self.df_cur_ex['pH_output'] = df_filtered_ex['Concentration'].tail(1).values[0]

            # Adds file info
            self.df_cur_ex['output_file'] = self.cf['output_file']
            self.df_cur_ex['sheet'] = self.cf['sheet']

            df_list.append(self.df_cur_ex)

        # Stacks all results
        df_vertical_stack = pd.concat(df_list, axis=0)

        # Saves all results as one file
        df_vertical_stack.to_excel(f"../data_jkCs/{self.cf['output_file']}_[{self.cf['sheet']}].xlsx")


    def run(self):
        sh = self.cf['sheet']

        # 1. Loads data from xlsx
        self.excel_data = pd.read_excel(self.cf['input_file'], sh)

        # Cs calculation rules
        if ('Free Chlorine' in sh) or ('Monochloramine' in sh) or ('DO' in sh):
            self.run_for_FC_MN_DO()
        elif ('pH' in sh):
            self.run_for_pH()

configures = [
                 # {'input_file': '../data/data_error.xlsx',
                 # 'output_file': 'data_error_jkCs',
                 # 'sheets':['Free Chlorine-Copper']
                 # },
                 {'input_file': '../data/data_free_chlorine.xlsx',
                  'output_file': 'data_free_chlorine_jkCs',
                  'sheets':['Free Chlorine-Ductile', 'Free Chlorine-Copper', 'DO-Copper', 'DO-Ductile', 'pH-Ductile', 'pH-Copper' ]
                 },
                 {'input_file': '../data/data_monochloramine.xlsx',
                  'output_file': 'data_monochloramine_jkCs',
                  'sheets':['Monochloramine-Ductile', 'Monochloramine-Copper', 'DO-Copper', 'DO-Ductile', 'pH-Ductile', 'pH-Copper' ]
                  },
             ]

for conf in configures:
    for sheet in conf['sheets']:
        print('***', conf['input_file'], conf['output_file'], sheet)
        JKCsFinder(conf['input_file'], conf['output_file'], sheet).run()

winsound.Beep(1000, 440)
