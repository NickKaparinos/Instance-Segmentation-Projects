"""
Instance-Segmentation-Projects
Nick Kaparinos
2021
"""

import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from utilities import *

if __name__ == "__main__":
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    dpi = 200

    # Read csv
    path = '/'
    file_name = 'wandb_export_2021-11-25T11_17_35.439+02_00.csv'
    data = pd.read_csv(path + file_name)
    plot_type = 'learning_curve'

    if plot_type == 'validation_metrics':
        # Drop unnecessary columns
        columns = data.columns.values
        columns_to_drop = [col for col in columns if 'MIN' in col or 'MAX' in col or 'step' in col]
        data.drop(columns=columns_to_drop, inplace=True)

        # Rename columns
        new_columns = ['Epoch']
        for column in data.columns.values[1:]:
            if 'segm' in column:
                new_column_name = 'Segmentation '
            else:
                new_column_name = 'BBox '
            if 'mAP' in column:
                new_column_name += 'mAP'
            else:
                new_column_name += 'mAR'
            new_columns.append(new_column_name)
        data.columns = new_columns

        # Plot
        fg = plt.figure(1, dpi=dpi)
        sns.set()
        for i in range(1, data.shape[1]):
            ax = sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, i], label=data.columns[i])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Validation metrics per training epoch', fontsize=16)
        plt.ylabel('Score')
        plt.savefig(path + 'validation_metrics.jpg', dpi=dpi)
    elif plot_type == 'learning_curve':
        # Drop unnecessary columns
        columns = data.columns.values
        columns_to_drop = [col for col in columns if 'MIN' in col or 'MAX' in col or 'step' in col]
        data.drop(columns=columns_to_drop, inplace=True)

        # Rename columns, keep the part from '-' onward
        new_columns = ['Epoch']
        for column in data.columns.values[1:]:
            new_column_name = column[column.find('-') + 2:]
            new_columns.append(new_column_name)
        data.columns = new_columns

        # Plot
        fg = plt.figure(1, dpi=dpi)
        sns.set()
        for i in range(1, data.shape[1]):
            ax = sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, i], label=data.columns[i])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('Learning Curve', fontsize=16)
        plt.ylabel('Loss')
        plt.savefig(path + 'learning_curve.jpg', dpi=dpi)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
