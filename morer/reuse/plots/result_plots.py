import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Assuming df is your input DataFrame
# Sample structure:
# STATISTICAL_TEST, FEATURE_CASE, COMMUNITY_DETECTION_ALGORITHM, EXTENSION, BUDGET, GT, TPS, FPS, FNS, P, R, F, F_std, runtime, SELECTION, data_set, al, method

# Load your data
# df = pd.read_csv('your_data.csv')  # Uncomment this line if you want to load data from a CSV

# Example of creating a grouped bar plot
def plot_grouped_bar(df):
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 30
    # sns.set_context("paper", rc={"figure.figsize": (8, 6)})

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    rc_dict = {
        'font.size': MEDIUM_SIZE,
        'axes.labelsize': BIGGER_SIZE,
        'axes.titlesize': BIGGER_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': MEDIUM_SIZE,
        'figure.titlesize': BIGGER_SIZE
    }
    sns.set_context(rc=rc_dict)
    # Extracting relevant columns: 'BUDGET', 'F', 'method'
    df['method'] = df[['method', 'al']].apply(lambda x: '+'.join(x.dropna()), axis=1)
    df['BUDGET'] = df['BUDGET'].map({-1: 'all', 1000: '1K',1500:'1.5K', 2000:'2K', 50:'50%', 0: '0'})
    plot_data = df[['data_set', 'BUDGET', 'F', 'method']]
    print(plot_data)
    # Set plot style for better visualization
    sns.set_style(style='whitegrid')
    # Initialize the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Define the color palette
    palette = ['#008837', '#a6dba0', '#7b3294', '#c2a5cf', '#0081cf', '#926c00']

    # Create the grouped bar plot using seaborn's catplot
    g = sns.catplot(
        data=plot_data,
        col='data_set',
        col_order=['Dexter', 'WDC-computer', 'Music'],
        x='BUDGET',
        y='F',
        aspect=0.9,
        #width=2,
        height=5,
        hue='method',
        order=['0', '1K', '1.5K', '2K','50%', 'all'],
        margin_titles=True,
        native_scale=False,
        palette=palette,
        kind='bar',
        sharey=True,
        legend=True,  # Enable the legend
        legend_out=True,
        col_wrap=3,
        dodge=True,
        hue_order=['MoRe+Bootstrap', 'MoRe+Almser', 'Almser', 'TransER', 'Ditto', 'MultiEM']
    )
    g.legend.set_title(None)
    # Set titles for each subplot
    g.set_titles(col_template="{col_name}")

    # Set axis labels
    g.set_xlabels('Budget')
    g.set_ylabels('F1')
    sns.move_legend(g, loc='center right', bbox_to_anchor=(1.25, 0.5), frameon=True)
    plt.ylim(0.5, 1)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join('results/plots', 'comparison_new.pdf'), bbox_inches='tight')

def plot_grouped_bar_retrain(df):
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 30
    # sns.set_context("paper", rc={"figure.figsize": (8, 6)})

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    rc_dict = {
        'font.size': MEDIUM_SIZE,
        'axes.labelsize': BIGGER_SIZE,
        'axes.titlesize': BIGGER_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': MEDIUM_SIZE,
        'figure.titlesize': BIGGER_SIZE
    }
    sns.set_context(rc=rc_dict)

    def combine_columns(row):
        return '-'.join(str(x) for x in row if pd.notna(x))

    df['rt_budget'] = df['rt_budget'].astype('Int64')
    print(df['rt_budget'])
    # Extracting relevant columns: 'BUDGET', 'F', 'method'
    df['method'] = df[['num_rt', 'rt_budget']].apply(combine_columns, axis=1)

    df['BUDGET'] = df['BUDGET'].map({-1: 'all', 1000: '1K',1500:'1.5K', 2000:'2K', 50:'50%', 0: '0'})
    plot_data = df[['data_set','BUDGET', 'F', 'method']]
    # Set plot style for better visualization
    sns.set_style(style='whitegrid')
    # Initialize the matplotlib figure
    plt.figure(figsize=(8, 6))
    print(df)
    # Define the color palette
    palette = ['#696969', '#008837', '#a6dba0']

    # Create the grouped bar plot using seaborn's catplot
    g = sns.catplot(
        data=plot_data,
        col_order=['Dexter', 'WDC-computer', 'Music'],
        col='data_set',
        x='BUDGET',
        y='F',
        aspect=0.8,
        #width=2,
        height=5,
        hue='method',
        order=['1K', '1.5K', '2K'],
        margin_titles=True,
        native_scale=False,
        palette=palette,
        kind='bar',
        sharey=True,
        legend=True,  # Enable the legend
        legend_out=True,
        dodge=True,
        hue_order=['w/o', '2-250', '5-100']
    )
    g.legend.set_title(None)
    # Set titles for each subplot
    g.set_titles(col_template="{col_name}")
    for ax in g.axes.flatten():
        ax.set_yticks(np.arange(0.7, 1, 0.05))  # Set specific y ticks
    # Set axis labels
    g.set_xlabels('Data set')
    g.set_ylabels('F1')
    #sns.move_legend(g, loc='center left', bbox_to_anchor=(0.3, -0.1), ncol=3, frameon=True)
    sns.move_legend(g, bbox_to_anchor=(1, 0.8), fontsize=MEDIUM_SIZE, loc='upper left', frameon=True)
                #    title='budget-#retrain-b_retrain')
    plt.ylim(0.7, 1)

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join('results/plots', 'retrain.pdf'), bbox_inches='tight')

def plot_stats_grouped_bar(df):
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16
    # sns.set_context("paper", rc={"figure.figsize": (8, 6)})
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    rc_dict = {
        'font.size': MEDIUM_SIZE,
        'axes.labelsize': BIGGER_SIZE,
        'axes.titlesize': BIGGER_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': MEDIUM_SIZE,
        'figure.titlesize': BIGGER_SIZE
    }

    # Extracting relevant columns: 'BUDGET', 'F', 'method'
    new_df = df[df['BUDGET'] == 1500]
    new_df["method"] = new_df['method'] + "-" + new_df['al']
    plot_data = new_df[['data_set', 'STATISTICAL_TEST', 'F', 'al']]

    # Set plot style for better visualization
    sns.set_style(style='whitegrid')
    # sns.set(rc={'legend.labelspacing': 0.2, 'legend.handletextpad': 0.1, 'legend.columnspacing': 0.1})
    sns.set_context(rc=rc_dict)
    # Initialize the matplotlib figure
    plt.figure(figsize=(6, 6))
    g = sns.FacetGrid(plot_data, col='data_set', legend_out=True, despine=False,
                      row_order=['bs', 'Almser'],
                      col_order=['Dexter', 'WDC-computer', 'Music'])
    # Create the grouped bar plot using seaborn's barplot
    g.set_titles(col_template="{col_name}")
    palette=['#a6cee3','#1f78b4','#b2df8a']
    g.map(sns.barplot,
          'al',
          'F',
          'STATISTICAL_TEST',
          palette=palette, hue_order=['KS', 'WD', 'PSI']
          )

    ax1, ax2, ax3 = g.axes[0]

    # ax1.axvline(3, ls='--')
    # ax2.axhline(3, ls='--')
    # ax3.axhline(3, ls='--')
    #g.set_xticklabels(rotation=45)
    g.set_xlabels('')
    g.set_ylabels('F1')
    g.set(yticks=np.arange(0.7, 1, 0.05))
    plt.legend(bbox_to_anchor=(1, 1), fontsize=MEDIUM_SIZE, loc='upper left', title='statistical test')
    plt.ylim(0.7, 1)
    # Setting the title and labels
    # ax.title(title)

    # value_tick = range(1000, 2500, 500)
    # ax.set_xticks(ticks=value_tick)  # set new labels
    # ax.set_xticklabels(labels=['1k', '1.5k', '2k'])

    # Rotate x-axis labels for better readability
    #plt.xticks(rotation=45)

    # Adding legend and setting the position
    # Show the plot
    #plt.tight_layout()
    plt.savefig(os.path.join('results/plots', 'stat_comparison.pdf'), bbox_inches='tight')
    plt.show()



def plot_grouped_bar_runtime(df):
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 30
    # sns.set_context("paper", rc={"figure.figsize": (8, 6)})

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    rc_dict = {
        'font.size': MEDIUM_SIZE,
        'axes.labelsize': BIGGER_SIZE,
        'axes.titlesize': BIGGER_SIZE,
        'xtick.labelsize': MEDIUM_SIZE,
        'ytick.labelsize': MEDIUM_SIZE,
        'legend.fontsize': MEDIUM_SIZE,
        'figure.titlesize': BIGGER_SIZE
    }
    sns.set_context(rc=rc_dict)
    # Extracting relevant columns: 'BUDGET', 'F', 'method'
    df['method'] = df[['method', 'al']].apply(lambda x: '+'.join(x.dropna()), axis=1)
    df['BUDGET'] = df['BUDGET'].map({-1: 'all', 1000: '1K', 1500: '1.5K', 2000: '2K', 50: '50%', 0: '0'})
    plot_data = df[['data_set', 'BUDGET', 'runtime', 'method']]
    print(plot_data)
    # Set plot style for better visualization
    sns.set_style(style='whitegrid')
    # Initialize the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Define the color palette
    palette = ['#008837', '#a6dba0', '#7b3294', '#c2a5cf', '#0081cf', '#926c00']

    # Create the grouped bar plot using seaborn's catplot
    g = sns.catplot(
        data=plot_data,
        col='data_set',
        col_order=['Dexter', 'WDC-computer', 'Music'],
        x='BUDGET',
        y='runtime',
        aspect=0.9,
        # width=2,
        height=5,
        hue='method',
        order=['0', '1K', '1.5K', '2K', '50%', 'all'],
        margin_titles=True,
        native_scale=False,
        palette=palette,
        kind='bar',
        sharey=True,
        legend=True,  # Enable the legend
        legend_out=True,
        col_wrap=3,
        dodge=True,
        hue_order=['MoRe+Bootstrap', 'MoRe+Almser', 'Almser', 'TransER', 'Ditto', 'MultiEM']
    )
    g.legend.set_title(None)
    # Set titles for each subplot
    g.set_titles(col_template="{col_name}")

    # Set axis labels
    g.set_xlabels('Budget')
    plt.yscale("log")
    g.set_ylabels('Runtime(s)')
    #sns.move_legend(g, loc='center left', bbox_to_anchor=(0.1, -0.1), ncol=3, frameon=True)
    sns.move_legend(g, loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1, frameon=True)
    plt.ylim(1, 30000)

    plt.tight_layout()


    # value_tick = range(1000, 2500, 500)
    # ax.set_xticks(ticks=value_tick)  # set new labels
    # ax.set_xticklabels(labels=['1k', '1.5k', '2k'])

    # Rotate x-axis labels for better readability
    #plt.xticks(rotation=45)

    # Adding legend and setting the position
    # Show the plot
    #plt.tight_layout()
    plt.savefig(os.path.join('results/plots', "runtime_new.pdf"), bbox_inches='tight')
    plt.show()


# Assuming the DataFrame is named df
# Call the function to generate the plot
# plot_grouped_bar(df)  # Uncomment this line to generate the plot after loading your DataFrame
if __name__ == '__main__':
    df_ret = pd.read_csv('results/trans_er_incremental_results_retrain.csv')
    plot_grouped_bar_retrain(df_ret)
    df = pd.read_csv('results/comparison_result_new.csv')
    stat_df = pd.read_csv('results/comparison_stats.csv')
    plot_grouped_bar(df)
    plot_stats_grouped_bar(stat_df)
    df = pd.read_csv('results/comparison_result_new.csv')
    plot_grouped_bar_runtime(df)
