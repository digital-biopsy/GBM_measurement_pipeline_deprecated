import os
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self):
        self.train_dir = 'data'
        self.labels = pd.read_csv(os.path.join(sys.path[0], self.train_dir, 'tile_stats.csv'))
        # self.drop_nan_vals()

    def drop_nan_vals(self):
        self.labels = self.labels.dropna()
        self.labels = self.labels.reset_index(drop=True)

    def pseudo_pred(self, m, r):
        return random.uniform((1-r)*m,(1+r)*m)

    def get_predictions(self, means):
        gbmw_pred = self.pseudo_pred(means[0], 0.32)
        fpw_pred = self.pseudo_pred(means[1], 0.24)
        sdd_pred = self.pseudo_pred(means[2], 0.06)
        gbml_pred = self.pseudo_pred(means[3], 0.16)
        return ([gbmw_pred, fpw_pred, sdd_pred, gbml_pred]-means)/means*100

    def evaluate(self):
        tar_columns = self.labels[['gbmw', 'fpw', 'sdd', 'gbml']]
        col_means = np.mean(tar_columns, axis=0).to_numpy()
        validation_list = []
        for i in range(len(self.labels.index)):
            validation_list.append(self.get_predictions(col_means))
        image_list = self.labels[['image', 'tile_index']]
        eval_results = pd.DataFrame(validation_list, columns=['gbmw', 'fpw', 'sdd', 'gbml'])
        csv_results = pd.concat([image_list, eval_results], axis=1)
        csv_path = os.path.join(sys.path[0], 'evaluation', 'eval_results.csv')
        csv_results.to_csv(csv_path)
        
        ax = sns.boxplot(data=eval_results)
        plt.show()