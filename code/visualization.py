import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def visualization_1(csv_path, alpha_list, fig_path='./result/figures/result_alpha.png', scale=1, cut=False, sharey=True, sharex=False, shareylabel=True, sharexlabel=True, sharextitle=False, column=3):
    
    try:
        for path in csv_path:
            try:
                new = pd.read_csv(path)
                df = pd.concat([df, new], axis=0)
            except:
                df = pd.read_csv(path)
    except:
        df = pd.read_csv(csv_path)

    df = df.fillna(100)

    df1 = df.iloc[:,2:]

    x_ticks = np.concatenate([(np.arange(9) + 2) * 10 ** i for i in range(math.ceil(math.log10(df1.shape[1])))])
    x_ticks = np.concatenate([np.array([1]), x_ticks])

    x_ticks = x_ticks[x_ticks <= df1.shape[1]]

    l = len(alpha_list)
    r = (l - 1) // column + 1
    
    fig, axes = plt.subplots(r, column, figsize=(12, 12 / column * r * scale), sharey=sharey, sharex=sharex)

    if r == 1:
        axes = axes[None, :]
    
    if column == 1:
        axes = axes[:, None]
    
    for i in range(r):
        for j in range(column):
            if i * column + j < l:
                acc1 = np.array(df1.iloc[4 * (i * column + j), :])
                acc2 = np.array(df1.iloc[4 * (i * column + j) + 1, :])
                if cut:
                    m1 = np.where(acc1 > 99.5)[0]
                    x1, x2 = df1.shape[1], df1.shape[1]
                    if m1.shape[0] > 0:
                        x1 = 2 * m1.min()
                    m2 = np.where(acc2 > 99.5)[0]
                    if m2.shape[0] > 0:
                        x2 = 2 * m2.min()
                    x_max = min(max(x1, x2), df1.shape[1])
                    acc1 = acc1[:x_max]
                    acc2 = acc2[:x_max]
                    if not sharex:
                        x_ticks = x_ticks[x_ticks <= x_max]
                axes[i, j].plot(acc1)
                axes[i, j].plot(acc2)
                axes[i, j].set_xscale('log')
                axes[i, j].set_xticks(x_ticks)
                if shareylabel:
                    if j == 0:
                        axes[i, j].set_ylabel('Accuracy(%)')
                else:
                    axes[i, j].set_ylabel('Accuracy(%)')
                if sharexlabel:
                    if i == r-1:
                        axes[i, j].set_xlabel('epoch')
                else:
                    axes[i, j].set_xlabel('epoch')
                if sharextitle:
                    if i == 0:
                        axes[i, j].set_title('alpha = {}'.format(alpha_list[i * 3 + j]))
                else:
                    axes[i, j].set_title('alpha = {}'.format(alpha_list[i * 3 + j]))

    plt.tight_layout()

    fig.savefig(fig_path)

    plt.close()

def visualization_2(csv_path, n_list, fig_path='./result/figures/result_alpha.png', scale=1, cut=False, sharey=True, sharex=False, shareylabel=True, sharexlabel=True, sharextitle=False, column=3):
    
    try:
        for path in csv_path:
            try:
                new = pd.read_csv(path)
                df = pd.concat([df, new], axis=0)
            except:
                df = pd.read_csv(path)
    except:
        df = pd.read_csv(csv_path)

    df = df.fillna(100)

    df1 = df.iloc[:,2:]

    x_ticks = np.concatenate([(np.arange(9) + 2) * 10 ** i for i in range(math.ceil(math.log10(df1.shape[1])))])
    x_ticks = np.concatenate([np.array([1]), x_ticks])

    x_ticks = x_ticks[x_ticks <= df1.shape[1]]

    l = len(n_list)
    r = (l - 1) // column + 1
    
    fig, axes = plt.subplots(r, column, figsize=(12, 12 / column * r * scale), sharey=sharey, sharex=sharex)

    if r == 1:
        axes = axes[None, :]
    
    if column == 1:
        axes = axes[:, None]

    for i in range(r):
        for j in range(column):
            if i * column + j < l:
                acc1 = np.array(df1.iloc[4 * (i * column + j), :])
                acc2 = np.array(df1.iloc[4 * (i * column + j) + 1, :])
                if cut:
                    m1 = np.where(acc1 > 99.5)[0]
                    x1, x2 = df1.shape[1], df1.shape[1]
                    if m1.shape[0] > 0:
                        x1 = 2 * m1.min()
                    m2 = np.where(acc2 > 99.5)[0]
                    if m2.shape[0] > 0:
                        x2 = 2 * m2.min()
                    x_max = min(max(x1, x2), df1.shape[1])
                    acc1 = acc1[:x_max]
                    acc2 = acc2[:x_max]
                    if not sharex:
                        x_ticks = x_ticks[x_ticks <= x_max]
                axes[i, j].plot(acc1)
                axes[i, j].plot(acc2)
                axes[i, j].set_xscale('log')
                axes[i, j].set_xticks(x_ticks)
                if shareylabel:
                    if j == 0:
                        axes[i, j].set_ylabel('Accuracy(%)')
                else:
                    axes[i, j].set_ylabel('Accuracy(%)')
                if sharexlabel:
                    if i == r-1:
                        axes[i, j].set_xlabel('epoch')
                else:
                    axes[i, j].set_xlabel('epoch')
                if sharextitle:
                    if i == 0:
                        axes[i, j].set_title('K = {}'.format(n_list[i * column + j]))
                else:
                    axes[i, j].set_title('K = {}'.format(n_list[i * column + j]))

    plt.tight_layout()

    fig.savefig(fig_path)

    plt.close()

def visualization_3(csv_path, optimizer_names, alpha_list, optimizer_indexes=None, fig_path='./result/figures/result_3.png', scale=0.75, column=3):

    try:
        for path in csv_path:
            try:
                new = pd.read_csv(path)
                df = pd.concat([df, new], axis=0)
            except:
                df = pd.read_csv(path)
    except:
        df = pd.read_csv(csv_path)

    df = df.fillna(100)

    df1 = df.iloc[:,3:-1]
    df2 = df.loc[:,'average']

    if optimizer_indexes == None:
        optimizer_indexes = range(len(optimizer_names))

    l2 = len(alpha_list)
    l1 = len(optimizer_indexes)
    l3 = df1.shape[1]
    r = (l1 - 1) // column + 1
    
    fig, axes = plt.subplots(r, column, figsize=(12, 12 / column * r * scale), sharex=True, sharey=True)

    if r == 1:
        axes = axes[None, :]
    
    if column == 1:
        axes = axes[:, None]
    
    for i in range(r):
        for j in range(column):
            if i * column + j < l1:
                average = df2.iloc[(optimizer_indexes[i*column + j]) * l2: (optimizer_indexes[i*column + j] + 1) * l2]
                axes[i, j].plot(alpha_list, average)

                for k in range(l2):
                    points = df1.iloc[(optimizer_indexes[i*column + j]) * l2 + k, :]
                    x = alpha_list[k] * np.ones((l3,))
                    axes[i, j].scatter(x, points, alpha=0.4, color='blue')
                    axes[i, j].set_ylim(0, 100)
                    axes[i, j].set_xlim(min(alpha_list), max(alpha_list))
                    if j == 0:
                        axes[i, j].set_ylabel('Accuracy(%)')
                    if i == r-1:
                        axes[i, j].set_xlabel('alpha')

                axes[i, j].grid(True)
                axes[i, j].set_title(optimizer_names[optimizer_indexes[i*3 + j]])

    plt.tight_layout()

    fig.savefig(fig_path)

    plt.close()