# Import Packages
from pandas import read_excel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from pandas.plotting import autocorrelation_plot, scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error 
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss

def ReadData():
    dt = read_excel('Dataset.xls', sheet_name='Data1', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  
    dt2 = read_excel('Dataset.xls', sheet_name='Data2', header=0, 
              index_col=0, parse_dates=True, squeeze=True) 
    dt3 = read_excel('Dataset.xls', sheet_name='Data3', header=0, 
              index_col=0, parse_dates=True, squeeze=True) 
    return dt,dt2,dt3


def TimePlot(dt):
    names = dt.columns.tolist()
    index = dt.index.name
    startingpoint=[1998,1998,1983,1998,1997]
    for i in range(len(dt.columns)):
        series = dt[[names[i]]].copy()
        # Drop Rows with Empty Cells
        series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        
        # Imports new index with numbers
        series.reset_index(inplace=True)
        
        # Import data
        x = series[index].values
        y = series[names[i]].values
        
        plt.figure(figsize=(10,5), dpi=80)
        plt.plot(x, y, color='steelblue')
        plt.gca().set(title='Unemployment rates in {0} from {1} to 2020'.format(names[i],startingpoint[i]), 
                      xlabel='Dates', ylabel='Unemployment percentage\n of active population')
        plt.show()
    return

def TwoSidesPlot(dt):
    names = dt.columns.tolist()
    index = dt.index.name
    startingpoint=[1998,1998,1983,1998,1997]

    for i in range(len(dt.columns)):
        series = dt[[names[i]]].copy()
        
        # Drop Rows with Empty Cells
        series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        series.reset_index(inplace=True)

        # Import data
        x = series[index].values
        y1 = series[names[i]].values
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi= 80)
        plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='steelblue')
        plt.ylim(-(series[names[i]].max() + 2), series[names[i]].max() + 2)
        plt.title('Unemployment of {0} from {1} to 2000 (Two Side View)'.format(names[i],startingpoint[i]), fontsize=12)
        plt.xlabel("Dates")
        plt.ylabel("Unemployment percentage\n of active population")
        plt.hlines(y=0, xmin=np.min(series.iloc[:,0]), xmax=np.max(series.iloc[:,0]), linewidth=.5)
        plt.show()
    return


def SeasonalPlot(dt):
    names = dt.columns.tolist()
    startingpoint=[1998,1998,1983,1998,1997]

    for j in range(len(dt.columns)):
        # Import Data
        series = dt[[names[j]]].copy()
        series.reset_index(inplace=True)
        series['year'] = [d.year for d in series.Dates]
        series['month'] = [d.strftime('%b') for d in series.Dates]
        years = series['year'].unique()

        # Preparation of Colours
        np.random.seed(100)
        mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)
        
        # Draw Plot
        plt.figure(figsize=(10,5), dpi= 80)
        for i, y in enumerate(years):
            if i > 0:        
                plt.plot('month', names[j], data=series.loc[series.year==y, :], color=mycolors[i], label=y)
                plt.text(series.loc[series.year==y, :].shape[0]-.9, series.loc[series.year==y, names[j]][-1:].values[0], y, fontsize=12, color=mycolors[i])
        
        # Decoration
        plt.gca().set(xlim=(-0.1, 11), ylim=(0, series[names[j]].max() + 2), ylabel='Unemployment rates\n of active population$', xlabel='$Month$')
        plt.yticks(fontsize=12, alpha=.7)
        plt.title("Seasonal plot of unemployment rates in {0} from {1} to 2020".format(names[j],startingpoint[j]), fontsize=12)
        plt.show()     
    return

def Boxplot(dt):                    

    names = dt.columns.tolist()

    for j in range(len(dt.columns)):
        # Import Data
        series = dt[[names[j]]].copy()
        series.reset_index(inplace=True)
        series['year'] = [d.year for d in series.Dates]
        series['month'] = [d.strftime('%b') for d in series.Dates]
        years = series['year'].unique()
        
        # Draw Plot
        fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 100)
        sns.boxplot(x='year', y= names[j], data=series, ax=axes[0])
        sns.boxplot(x='month', y= names[j], data=series.loc[~series.year.isin([1999, 2020]), :])

        # Set Title
        axes[0].set_title('Year-wise Box Plot of {0} \n(The Trend)'.format(names[j]), fontsize=12); 
        axes[1].set_title('Month-wise Box Plot of {0} \n(The Seasonality)'.format(names[j]), fontsize=12)
        plt.show()
    return


def Decomposition(dt):
    names = dt.columns.tolist()
    for j in range(len(dt.columns)):
        # Import Data
        series = dt[[names[j]]].copy()
        
        # Drop Rows with Empty Cells
        series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
        # Multiplicative Decomposition 
        result_mul = seasonal_decompose(series[names[j]], model='multiplicative', extrapolate_trend='freq')
        
        # Additive Decomposition
        result_add = seasonal_decompose(series[names[j]], model='additive', extrapolate_trend='freq')
        
        # Plot
        plt.rcParams.update({'figure.figsize': (10,10)})
        result_mul.plot()
        result_add.plot()
        plt.show()
    return


def ACF(dt):
    names = dt.columns.tolist()
    
    for j in range(len(dt.columns)):
        # Import Data
        series = dt[[names[j]]].copy()
        # Drop Rows with Empty Cells
        series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        
        #autocorrelation_plot(series)
        plot_acf(series, title='ACF Plot unemployment in {0} - Histogram format'.format(names[j]), lags=100)
        plt.show()
        
        # PACF plot on 50 time lags
        plot_pacf(series, title='PACF of unemployment in {0} - Histogram format'.format(names[j]), lags=50)
        plt.show()
    return
