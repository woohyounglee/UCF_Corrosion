import pandas as pd
from datetime import datetime
import winsound
import glob, os


class JKCsSummary():
    def __init__(self):
        self.time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")

    def run(self):
        path = "../data_jkCs"
        os.chdir(path)

        df_list = []
        for file in glob.glob("*.xlsx"):
            print(file)

            if 'pH' in file:
                continue

            # Loads data from xlsx
            self.excel_data = pd.read_excel(path + '/' + file, 'Sheet1')

            # Takes a experiment list
            experiments = self.excel_data['Experiment'].unique()

            # For each experiment
            for ex in experiments:
                # Takes an experiment data
                self.df_cur_ex = self.excel_data[(self.excel_data['Experiment'] == ex)]
                df_list.append(self.df_cur_ex.tail(1))

        # Stacks all results
        df_vertical_stack = pd.concat(df_list, axis=0)

        # Saves all results as one file
        df_vertical_stack.to_excel(f"../data_output/summary_{self.time}.xlsx")

JKCsSummary().run()

winsound.Beep(1000, 440)
