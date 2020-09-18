# Import Packages
from __future__ import print_function
from pandas import read_excel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from pandas.plotting import autocorrelation_plot, scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from statsmodels.formula.api import ols,wls
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error 
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
import  math, datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn import linear_model
from sklearn.svm import SVR



def SummaryTimePlot(dt):
    names = dt.columns.tolist()
    indx = dt.index.name
    #dt.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    dt.reset_index(inplace=True)

    # create a color palette
    palette = plt.get_cmap('tab10')
    # multiple line plot
    plt.figure(figsize=(16,5), dpi= 80)
    num=0

    for column in dt.drop(indx, axis=1):
        num+=1
        plt.plot(dt[indx], dt[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
         
        # Add legend
        plt.legend(loc=2, ncol=2)
         
        # Add titles
        plt.title("Spaghetti plot of Unemployment in Countries ", loc='center', fontsize=12, alpha=.7)
        plt.xlabel("Years")
        plt.ylabel("Unemployment rates of \nactive population")
    return 


def Correlation(dt):
    names = dt.columns.tolist()


    # Compute Pairwise Correlation
    CorrelationMatrix = dt.corr()
    print('\n\nCompute Pairwise Correlation of countries:\n{0}\n'.format(CorrelationMatrix))
    
    # Sorted Correlations Based on a Specific Dataset
    print('\n\nSorted Correlations Based on Dataset of {1}:\n{0}\n'.format(CorrelationMatrix[names[0]].sort_values(ascending=False),names[0]))
    
    # Shows Pairwise Correlation Plot and Kernel density estimation (KDE)
    scatter_matrix(dt, figsize=(16,12))
    plt.suptitle('Scatter Plot')
    plt.show()
    return


def DescriptiveStatistics(series):
    # Generates Descriptive Statistics
    print('\n\nDescriptive Statistics of all countries:\n{0}\n'.format(series.describe()))   
    return


def ANOVA(dt):

    # Inserts a new index    
    dt.reset_index(inplace=True)
    
    # Reading the basic variables
    Greece = dt.Greece
    Portugal = dt.Portugal
    Netherlands = dt.Netherlands
    Latvia = dt.Latvia
    Poland = dt.Poland

    # Monthly Dummy variables
    D1 = dt.D1
    D2 = dt.D2
    D3 = dt.D3
    D4 = dt.D4
    D5 = dt.D5
    D6 = dt.D6
    D7 = dt.D7
    D8 = dt.D8
    D9 = dt.D9
    D10 = dt.D10
    D11 = dt.D11
    
    
    # Dummy variable for summer months
    Dum = dt.Dum


    # Reading the Time Variables
    time1 = dt.time
    time2 = dt.timepowerof2
    time3 = dt.timepowerof3


    # Regression model(s)
    formula1 = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D2+D3+D4+D5+D6+D7+D8+D9+D10+D11+Dum+time+timepowerof2+timepowerof3' 
    formula2 = 'Greece~Portugal+Netherlands+Latvia+Poland+D10+Dum+time+timepowerof2+timepowerof3' 
    formula3 = 'Greece~Portugal+Netherlands+Latvia+Poland+D11+Dum+time+timepowerof2+timepowerof3' # optimal
    formula4 = 'Greece~Portugal+Netherlands+Latvia+Poland+Dum+time+timepowerof2+timepowerof3'
    formula5 = 'Greece~Portugal+Netherlands+Latvia+Poland+D10+D11+time+timepowerof2+timepowerof3'  
    formula6 = 'Greece~Portugal+Netherlands+Latvia+Poland+time+timepowerof2+timepowerof3'  
    
    # Ordinary Least Squares (OLS)
    results1 = ols(formula1, data=dt).fit()
    results2 = ols(formula2, data=dt).fit()
    results3 = ols(formula3, data=dt).fit()
    results4 = ols(formula4, data=dt).fit()
    results5 = ols(formula5, data=dt).fit()
    results6 = ols(formula6, data=dt).fit()
    
    print(results1.summary())
    print(results2.summary())
    print(results3.summary())
    print(results4.summary())
    print(results5.summary())
    print(results6.summary())
    return



def Regression(dt, fitted_values):
    # Inserts a new index    
    #dt.reset_index(inplace=True)
    
    # Reading the basic variables
    Greece = dt.Greece
    Portugal = dt.Portugal
    Netherlands = dt.Netherlands
    Latvia = dt.Latvia
    Poland = dt.Poland

    
    # Create dummy variables 

    
    # Reading the indicator variables
    D1 = dt.D1
    D2 = dt.D2
    D3 = dt.D3
    D4 = dt.D4
    D5 = dt.D5
    D6 = dt.D6
    D7 = dt.D7
    D8 = dt.D8
    D9 = dt.D9
    D10 = dt.D10
    D11 = dt.D11

    
    # Dummy variable for summer months
    Dum = dt.Dum

    # Reading the Time Variables
    time1 = dt.time
    time2 = dt.timepowerof2
    time3 = dt.timepowerof3
    
    # Merges the 3 dataframes in 1 
    # series = pd.concat([dt, dummies, time], axis=1)

    # Regression model(s)
    formula = 'Greece~Portugal+Netherlands+Latvia+Poland+D11+Dum+time+timepowerof2+timepowerof3'  # Optimal
    # Ols generate statistics and the parameters b0, b1, etc., of the model
    results = ols(formula, data=dt).fit()
    print(results.summary())
    results.summary()

    
    b0 = results.params.Intercept
    b1 = results.params.Portugal
    b2 = results.params.Netherlands
    b3 = results.params.Latvia
    b4 = results.params.Poland
    b5 = results.params.D11
    b6 = results.params.Dum
    b7 = results.params.time
    b8 = results.params.timepowerof2
    b9 = results.params.timepowerof3


    fitted_values = fitted_values[-266:-1]

    a1 = np.array(fitted_values.Portugal)
    a2 = np.array(fitted_values.Netherlands)
    a3 = np.array(fitted_values.Latvia)
    a4 = np.array(fitted_values.Poland)
    a5 = np.array(dt.D11)
    a6 = np.array(dt.Dum)
    a7 = np.array(dt.time)
    a8 = np.array(dt.timepowerof2)
    a9 = np.array(dt.timepowerof3)
    
    F=a1
    for i in range(265):
        F[i] = b0 + a1[i]*b1 + a2[i]*b2 + a3[i]*b3  + a4[i]*b4 + a5[i]*b5 + a6[i]*b6 + a7[i]*b7 + a8[i]*b8 + a9[i]*b9

    # print(Dum.tail(20))
    
    forec = read_excel('Dataset.xls', sheet_name ='HW.Regression1',header=0)
    predictions = results.predict(forec)

        
    v1=np.array(forec.Portugal)
    v2=np.array(forec.Netherlands)
    v3=np.array(forec.Latvia)
    v4=np.array(forec.Poland)
    v5=np.array(forec.D11)
    v6=np.array(forec.Dum)
    v7=np.array(forec.time)
    v8=np.array(forec.timepowerof2)
    v9=np.array(forec.timepowerof3)
   
    
    
    # Building the 8 values of the forecast ahead
    E=v1
    print(E)
    for i in range(8):
        E[i] = b0 + v1[i]*b1  + v2[i]*b2 + v3[i]*b3  + v4[i]*b4 + v5[i]*b5 + v6[i]*b6 + v7[i]*b7 + v8[i]*b8 + v9[i]*b9 
        
    # Joining the fitted values of the forecast and the points ahead
    K=np.append(F, E)
    
    
    print('Forecasts')
    print(E)


    val = read_excel('Dataset.xls', sheet_name='HW.Reg1', header=0, 
                      squeeze=True, dtype=float)

    # Evaluating the MSE to generate the confidence interval
    values=val.Greece[:265]
    Error = values - F
    MSE=sum(Error**2)*1.0/len(F)
    
    # Lower and upper bounds of forecasts for z=1.960 (95%); see equation (2.2) in Chap 2.
    LowerE = E - 1.960*MSE
    UpperE = E + 1.960*MSE
    
    
    print('UpperE')
    print(UpperE)
    print("LowerE")
    print(LowerE)

    LowerE = val.LowerE
    UpperE = val.UpperE
    
    plt.figure(figsize=(12,8),dpi= 100)
    line1, = plt.plot(K, color='red', label='Forecast values')
    line2, = plt.plot(values, color='steelblue', label='Original data')
    line3, = plt.plot(LowerE, color='darkslategray', label='95% Confidence Intervals')
    line4, = plt.plot(UpperE, color='darkslategray')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.title('Ordinary least squares regression forecasts with confidence interval for Greece`s total unemployment rates')
    plt.show()
    
    MSE1=mean_squared_error(F, values)
    print('\nThe MSE value of the regression forecasting is:\n{0}'.format(MSE1))
    return


def Weighted_ANOVA(dt):

    # # Inserts a new index    
    # dt.reset_index(inplace=True)

    # Reading the basic variables
    Greece = dt.Greece
    Portugal = dt.Portugal
    Netherlands = dt.Netherlands
    Latvia = dt.Latvia
    Poland = dt.Poland

    # Monthly Dummy variables
    D1 = dt.D1
    D2 = dt.D2
    D3 = dt.D3
    D4 = dt.D4
    D5 = dt.D5
    D6 = dt.D6
    D7 = dt.D7
    D8 = dt.D8
    D9 = dt.D9
    D10 = dt.D10
    D11 = dt.D11
    
    # Dummy variable for summer months
    Dum = dt.Dum
    
    # Reading the Time Variables
    time1 = dt.time
    time2 = dt.timepowerof2
    time3 = dt.timepowerof3
    
    # Weights
    w = []
    w1 = np.repeat(0.33,47)
    w2 = np.repeat(0.66,78)
    w3 = np.repeat(1.,141)
    w.extend(w1)
    w.extend(w2)
    w.extend(w3)
    print(len(w))
    print(len(dt))
    print(dt)

    # Regression model(s)
    formula1 = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D2+D3+D4+D5+D6+D7+D8+D9+D10+D11+Dum+time+timepowerof2+timepowerof3' 
    formula2 = 'Greece~Portugal+Netherlands+Latvia+Poland+D9+D10+D11+Dum+time+timepowerof2+timepowerof3' 
    formula3 = 'Greece~Portugal+Netherlands+Latvia+Poland+D9+D10+Dum+time+timepowerof2+timepowerof3' 
    formula4 = 'Greece~Portugal+Netherlands+Latvia+Poland+D10+D11+Dum+time+timepowerof2+timepowerof3'
    formula5 = 'Greece~Portugal+Netherlands+Latvia+Poland+D10+Dum+time+timepowerof2+timepowerof3'  # optimal
    formula6 = 'Greece~Portugal+Netherlands+Latvia+Poland+time+timepowerof2+timepowerof3'  
    
    # Ordinary Least Squares (OLS)
    results1 = wls(formula1, data=dt, weights = w).fit()
    results2 = wls(formula2, data=dt, weights = w).fit()
    results3 = wls(formula3, data=dt, weights = w).fit()
    results4 = wls(formula4, data=dt, weights = w).fit()
    results5 = wls(formula5, data=dt, weights = w).fit()
    results6 = wls(formula6, data=dt, weights = w).fit()
    
    print(results1.summary())
    print(results2.summary())
    print(results3.summary())
    print(results4.summary())
    print(results5.summary())
    print(results6.summary())
    return


def Weighted_Regression(dt, fitted_values):
    # Inserts a new index    
    dt.reset_index(inplace=True)

    # Reading the basic variables
    Greece = dt.Greece
    Portugal = dt.Portugal
    Netherlands = dt.Netherlands
    Latvia = dt.Latvia
    Poland = dt.Poland

    # Monthly Dummy variables
    D1 = dt.D1
    D2 = dt.D2
    D3 = dt.D3
    D4 = dt.D4
    D5 = dt.D5
    D6 = dt.D6
    D7 = dt.D7
    D8 = dt.D8
    D9 = dt.D9
    D10 = dt.D10
    D11 = dt.D11
    
    
    # Dummy variable for summer months
    Dum = dt.Dum


    # Reading the Time Variables
    time1 = dt.time
    time2 = dt.timepowerof2
    time3 = dt.timepowerof3
    
    
    # Regression model
    formula ='Greece~Portugal+Netherlands+Latvia+Poland+D11+Dum+time+timepowerof2+timepowerof3'  # Optimal
    
    a1 = 0.01
    a2 = 0.01
    i = 0.01
    j = 0.01
    AIC = 9999999
    while i<=1.00:
        j = 0.01
        while j<=1.00:
            # Weights
            w = []
            w1 = np.repeat(i,47)
            w2 = np.repeat(j,78)
            w3 = np.repeat(1,141)
            w.extend(w1)
            w.extend(w2)
            w.extend(w3)
            results =  wls(formula, data=dt, weights = w).fit()
            print(results.aic)
            if (results.aic < AIC):
                a1 = i
                a2 = j
                AIC = results.aic
            j = round(j + 0.01,2)
        i = round(i + 0.01,2)

    print(a1)
    print(a2)



    print('The weights of with the smallest AIC value:\n{0}\n{1}'.format(a1,a2))
    w = []
    w1 = np.repeat(a1,47)
    w2 = np.repeat(a2,78)
    w3 = np.repeat(1,141)

    w.extend(w1)
    w.extend(w2)
    w.extend(w3)   

    # wls generate statistics and the parameters b0, b1, etc., of the model
    results =  wls(formula, data=dt, weights = w).fit()

    
    
    print(results.summary())
    
    results.summary()

    
    b0 = results.params.Intercept
    b1 = results.params.Portugal
    b2 = results.params.Netherlands
    b3 = results.params.Latvia
    b4 = results.params.Poland
    b5 = results.params.D11
    b6 = results.params.Dum
    b7 = results.params.time
    b8 = results.params.timepowerof2
    b9 = results.params.timepowerof3


    fitted_values = fitted_values[-266:-1]

    a1 = np.array(fitted_values.Portugal)
    a2 = np.array(fitted_values.Netherlands)
    a3 = np.array(fitted_values.Latvia)
    a4 = np.array(fitted_values.Poland)
    a5 = np.array(dt.D11)
    a6 = np.array(dt.Dum)
    a7 = np.array(dt.time)
    a8 = np.array(dt.timepowerof2)
    a9 = np.array(dt.timepowerof3)
    
    F=a1
    for i in range(265):
        F[i] = b0 + a1[i]*b1 + a2[i]*b2 + a3[i]*b3  + a4[i]*b4 + a5[i]*b5 + a6[i]*b6 + a7[i]*b7 + a8[i]*b8 + a9[i]*b9

    # print(Dum.tail(20))
    forec = read_excel('Dataset.xls', sheet_name ='HW.Regression2',header=0)
    predictions = results.predict(forec)

        
    v1=np.array(forec.Portugal)
    v2=np.array(forec.Netherlands)
    v3=np.array(forec.Latvia)
    v4=np.array(forec.Poland)
    v5=np.array(forec.D11)
    v6=np.array(forec.Dum)
    v7=np.array(forec.time)
    v8=np.array(forec.timepowerof2)
    v9=np.array(forec.timepowerof3)
   
    
    # print(D8.tail(20))
    
    # Building the 8 values of the forecast ahead
    E=v1
    for i in range(8):
        E[i] = b0 + v1[i]*b1  + v2[i]*b2 + v3[i]*b3  + v4[i]*b4 + v5[i]*b5 + v6[i]*b6 + v7[i]*b7 + v8[i]*b8 + v9[i]*b9 
        
    # Joining the fitted values of the forecast and the points ahead
    K=np.append(F, E)
    print('Forecasts')
    print(E)

    val = read_excel('Dataset.xls', sheet_name='HW.Reg2', header=0, 
                      squeeze=True, dtype=float)
    ##########################
    # Evaluating the MSE to generate the confidence interval
    values=val.Greece[:265]
    Error = values - F
    MSE=sum(Error**2)*1.0/len(F)
    
    # Lower and upper bounds of forecasts for z=1.960 (95%); see equation (2.2) in Chap 2.
    LowerE = E - 1.960*MSE
    UpperE = E + 1.960*MSE
    
    
    print('UpperE')
    print(UpperE)
    print("LowerE")
    print(LowerE)

    LowerE = val.LowerE
    UpperE = val.UpperE
    
    plt.figure(figsize=(12,8),dpi= 100)
    line1, = plt.plot(K, color='red', label='Forecast values')
    line2, = plt.plot(values, color='steelblue', label='Original data')
    line3, = plt.plot(LowerE, color='darkslategray', label='95% Confidence Intervals')
    line4, = plt.plot(UpperE, color='darkslategray')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.title('Weighted least squares regression forecasts with confidence interval for Greece`s total unemployment rates')
    plt.show()
    
    MSE1=mean_squared_error(F, values)
    print('\nThe MSE value of the regression forecasting is:\n{0}'.format(MSE1))
    return





