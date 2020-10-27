
import pandas as pd
from datetime import datetime
import winsound
import os

class MakeDiscrete():
    def __init__(self, file):
        self.time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        self.file = file

    def run(self):
        path = "../data_output"
        os.chdir(path)

        df_list = []
        print(self.file)

        # 1. Loads data from xlsx
        self.excel_data = pd.read_excel(path+'/'+self.file, 'Sheet1')

        # Discretize in intervals for J, K, Cs
        num_bin = 10
        self.excel_data['dJ'] = pd.cut(self.excel_data['J'], num_bin)
        self.excel_data['dK'] = pd.cut(self.excel_data['K'], num_bin)
        self.excel_data['dCs'] = pd.cut(self.excel_data['Cs'], num_bin)

        # (1.4e-07, 5.42e-07]	(-4.92e-05, 0.00688]	(6.959, 7.726]
        def replace_data(x):
            x = str(x)
            x = x.replace('(', '')
            x = x.replace(')', '')
            x = x.replace(']', '')
            x = x.replace('.', '_')
            x = x.replace('-', '_')
            x = x.replace(',', '-')
            return x

        self.excel_data['dJ'] = self.excel_data['dJ'].apply(lambda x: replace_data(x))
        self.excel_data['dK'] = self.excel_data['dK'].apply(lambda x: replace_data(x))
        self.excel_data['dCs'] = self.excel_data['dCs'].apply(lambda x: replace_data(x))

        # Selects variables
        # selected_data = self.excel_data[['Phosphate', 'DIC', 'pH', 'dJ', 'dK', 'dCs']]
        selected_data = self.excel_data[['Phosphate', 'DIC', 'pH', 'dJ']]

        # Saves all results as one file
        selected_data.to_csv(f"../data_output/Discretized_{self.time}.csv")

file = 'summary_10_10_2020-11_55_48.xlsx'

MakeDiscrete(file).run()
winsound.Beep(1000, 440)
