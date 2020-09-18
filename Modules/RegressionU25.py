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
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style



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
        plt.ylabel("Unemployment - percentage of active population")
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

    
    # Create dummy variables 
    height = len(dt)
    width = 11
    dummies = pd.DataFrame(0, index=range(height), columns=range(width))
    dummies.columns = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11']
    for j in range(len(dummies.columns)):
        for i in range(j,len(dummies),12):
            dummies.iloc[[i],[j]] =1
            
    # Create dummy variables 
    height = len(dt)
    width = 1
    dum = pd.DataFrame(0, index=range(height), columns=range(width))
    dum.columns = ['Dum']

    for i in range(3,len(dum)-5,12):
        dum.iloc[[i],[0]] =1
        dum.iloc[[i+1],[0]] =1
        dum.iloc[[i+2],[0]] =1
        dum.iloc[[i+3],[0]] =1
        dum.iloc[[i+4],[0]] =1
        dum.iloc[[i+5],[0]] =1
              
    height = len(dt)
    width = 3
    time = pd.DataFrame(0, index=range(height), columns=range(width))        
    time.columns = ['time', 'timepowerof2', 'timepowerof3']
    
    for j in range(len(time.columns)):
        for i in range(0,len(time),1):
            time.iloc[[i],[j]] = (i+1)**(j+1)
    
    # Reading the indicator variables
    D1 = dummies.D1
    D2 = dummies.D2
    D3 = dummies.D3
    D4 = dummies.D4
    D5 = dummies.D5
    D6 = dummies.D6
    D7 = dummies.D7
    D8 = dummies.D8
    D9 = dummies.D9
    D10 = dummies.D10
    D11 = dummies.D11
    
    # Dummy variable for summer months
    Dum = dum.Dum


    # Reading the Time Variables
    time1 = time.time
    time2 = time.timepowerof2
    time3 = time.timepowerof3

    # Merges the 3 dataframes in 1 
    series = pd.concat([dt, dummies, time], axis=1)

    # Regression model(s)
    formula1 = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D2+D3+D4+D5+D6+D7+D8+D9+D10+D11+Dum+time+timepowerof2+timepowerof3'
    formula2 = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D7+D8+D10+Dum+time+timepowerof2+timepowerof3' 
    formula3 = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D8+D10+Dum+time+timepowerof2+timepowerof3' # optimal
    formula4 = 'Greece~Portugal+Netherlands+Latvia+Poland+D8+D10+Dum+time+timepowerof2+timepowerof3' 
    formula5 = 'Greece~Portugal+Netherlands+Latvia+Poland+D8+Dum+time+timepowerof2+timepowerof3'  

    # Ordinary Least Squares (OLS)
    results1 = ols(formula1, data=series).fit()
    results2 = ols(formula2, data=series).fit()
    results3 = ols(formula3, data=series).fit()
    results4 = ols(formula4, data=series).fit()
    results5 = ols(formula5, data=series).fit()

    print(results1.summary())
    print(results2.summary())
    print(results3.summary())
    print(results4.summary())
    print(results5.summary())
    return



def Regression(dt, fitted_values):

    
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
    
    # # Merges the 3 dataframes in 1 


    # Regression model(s)
    formula = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D8+D10+Dum+time+timepowerof2+timepowerof3'  # Optimal
    # Ols generate statistics and the parameters b0, b1, etc., of the model
    results = ols(formula, data=dt).fit()
    print(results.summary())
    results.summary()

    
    b0 = results.params.Intercept
    b1 = results.params.Portugal
    b2 = results.params.Netherlands
    b3 = results.params.Latvia
    b4 = results.params.Poland
    b5 = results.params.D1
    b6 = results.params.D8
    b7 = results.params.D10
    b8 = results.params.Dum
    b9 = results.params.time
    b10 = results.params.timepowerof2
    b11 = results.params.timepowerof3


    fitted_values = fitted_values[-266:-1]

 
    a1 = np.array(fitted_values.Portugal)
    a2 = np.array(fitted_values.Netherlands)
    a3 = np.array(fitted_values.Latvia)
    a4 = np.array(fitted_values.Poland)
    a5 = np.array(dt.D1)
    a6 = np.array(dt.D8)
    a7 = np.array(dt.D10)
    a8 = np.array(dt.Dum)
    a9 = np.array(dt.time)
    a10 = np.array(dt.timepowerof2)
    a11 = np.array(dt.timepowerof3)
    
    F=a1


    

    for i in range(265):
        F[i] = b0 + a1[i]*b1 + a2[i]*b2  + a3[i]*b3 + a4[i]*b4 + a5[i]*b5 + a6[i]*b6 + a7[i]*b7 + a8[i]*b8 + a9[i]*b9 + a10[i]*b10 + a11[i]*b11

        
    forec = read_excel('DatasetU25.xls', sheet_name ='HW.Regression1',header=0)
    predictions = results.predict(forec)

        
    v1=np.array(forec.Portugal)
    v2=np.array(forec.Netherlands)
    v3=np.array(forec.Latvia)
    v4=np.array(forec.Poland)
    v5=np.array(forec.D1)
    v6=np.array(forec.D8)
    v7=np.array(forec.D10)
    v8=np.array(forec.Dum)
    v9=np.array(forec.time)
    v10=np.array(forec.timepowerof2)
    v11=np.array(forec.timepowerof3)
   
    
    # Building the 8 values of the forecast ahead
    E=v1
    for i in range(8):
        E[i] = b0 + v1[i]*b1 + v2[i]*b2  + v3[i]*b3 + v4[i]*b4 + v5[i]*b5 + v6[i]*b6 + v7[i]*b7 + v8[i]*b8 + v9[i]*b9 + v10[i]*b10 + v11[i]*b11
        
    
    # Joining the fitted values of the forecast and the points ahead
    K=np.append(F, E)
    
    
    print('Forecasts')
    print(E)


    val = read_excel('DatasetU25.xls', sheet_name='HW.Reg1', header=0, 
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
    plt.title('Ordinary least squares regression forecasts with confidence interval for Greece`s U25 unemployment rates')
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


    # Regression model(s)
    formula1 = 'Greece~Portugal+Netherlands+Latvia+Poland+D1+D2+D3+D4+D5+D6+D7+D8+D9+D10+D11+Dum+time+timepowerof2+timepowerof3' 
    formula2 = 'Greece~Portugal+Netherlands+Latvia+Poland+D2+D3+D5+D6+D8+D10+D11+Dum+time+timepowerof2+timepowerof3' 
    formula3 = 'Greece~Portugal+Netherlands+Latvia+Poland+D2+D3+D5+D6+D10+time+timepowerof2+timepowerof3' 
    formula4 = 'Greece~Portugal+Netherlands+Latvia+Poland+D2+D3+D5+D6+D10+time+timepowerof2+timepowerof3'
    formula5 = 'Greece~Portugal+Netherlands+Latvia+Poland+D2+D3+D5+D6+D10+time+timepowerof2+timepowerof3'  # optimal
    formula6 = 'Greece~Portugal+Netherlands+Latvia+Poland+D2+D3+D5+D6+D10+time+timepowerof2+timepowerof3'  
    
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
    formula ='Greece~Portugal+Netherlands+Latvia+Poland+D1+D8+D10+Dum+time+timepowerof2+timepowerof3'  # Optimal
    
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
    


    a1 = 1
    a2 = 1

    print('The weights of with the smallest AIC value:\n{0}\n{1}'.format(a1,a2))
    w = []
    w1 = np.repeat(a1,47)
    w2 = np.repeat(a2,78)
    w3 = np.repeat(1,141)

    w.extend(w1)
    w.extend(w2)
    w.extend(w3)   
    print(len(w))

    # wls generate statistics and the parameters b0, b1, etc., of the model
    results =  wls(formula, data=dt, weights = w).fit()

    
    
    print(results.summary())
    
    results.summary()

    
    b0 = results.params.Intercept
    b1 = results.params.Portugal
    b2 = results.params.Netherlands
    b3 = results.params.Latvia
    b4 = results.params.Poland
    b5 = results.params.D1
    b6 = results.params.D8
    b7 = results.params.D10
    b8 = results.params.Dum
    b9 = results.params.time
    b10 = results.params.timepowerof2
    b11 = results.params.timepowerof3


    fitted_values = fitted_values[-266:-1]

    a1 = np.array(fitted_values.Portugal)
    a2 = np.array(fitted_values.Netherlands)
    a3 = np.array(fitted_values.Latvia)
    a4 = np.array(fitted_values.Poland)
    a5 = np.array(dt.D1)
    a6 = np.array(dt.D8)
    a7 = np.array(dt.D10)
    a8 = np.array(dt.Dum)
    a9 = np.array(dt.time)
    a10 = np.array(dt.timepowerof2)
    a11 = np.array(dt.timepowerof3)
    
    F=a1
    for i in range(265):
        F[i] = b0 + a1[i]*b1 + a2[i]*b2 + a3[i]*b3  + a4[i]*b4 + a5[i]*b5 + a6[i]*b6 + a7[i]*b7 + a8[i]*b8 + a9[i]*b9 + a10[i]*b10 + a11[i]*b11 


    forec = read_excel('DatasetU25.xls', sheet_name ='HW.Regression2',header=0)
    predictions = results.predict(forec)
    

        
    v1=np.array(forec.Portugal)
    v2=np.array(forec.Netherlands)
    v3=np.array(forec.Latvia)
    v4=np.array(forec.Poland)
    v5=np.array(forec.D1)
    v6=np.array(forec.D8)
    v7=np.array(forec.D10)
    v8=np.array(forec.Dum)
    v9=np.array(forec.time)
    v10=np.array(forec.timepowerof2)
    v11=np.array(forec.timepowerof3)
   
    

    
    # Building the 8 values of the forecast ahead
    E=v1
    for i in range(8):
        E[i] = b0 + v1[i]*b1  + v2[i]*b2 + v3[i]*b3  + v4[i]*b4 + v5[i]*b5 + v6[i]*b6 + v7[i]*b7 + v8[i]*b8 + v9[i]*b9 + v10[i]*b10 + v11[i]*b11 
    # Joining the fitted values of the forecast and the points ahead
    K=np.append(F, E)
    print('Forecasts')
    print(E)

    val = read_excel('DatasetU25.xls', sheet_name='HW.Reg2', header=0, 
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
    line3, = plt.plot(LowerE, color='blue', label='Lower forecast')
    line4, = plt.plot(UpperE, color='orange', label='Upper forecast')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.title('Weighted least squares regression forecasts with confidence interval for Greece`s U25 unemployment rates')
    plt.show()
    
    MSE1=mean_squared_error(F, values)
    print('\nThe MSE value of the regression forecasting is:\n{0}'.format(MSE1))
    return





def MachineLearning(dt):
    style.use('ggplot')
    forecast_out = 8
    series = dt[['Greece']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    dt = dt.iloc[:-1]
    
    y = series['Greece']
    y_true = y[-27:]

    y = np.array(y)
    df = dt
    X = dt.drop(columns=['Greece'])
    X = np.array(X)
    
    

    
    
    X_lately = read_excel('Dataset.xls', sheet_name='ML', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

    print(X_train)
    print(len(X_train))
    print('\n')
    print(X_test)
    print(len(X_test))
    print('\n')
    print(y_train)
    print(len(y_train))
    print('\n')
    print(y_test)
    print(len(X_test))

    clf1 = LinearRegression(n_jobs=-1)
    clf1.fit(X_train, y_train)
    confidence1 = clf1.score(X_test, y_test)
    forecast_set1 =  clf1.predict(X_lately)
    a = mean_squared_error(y_true,y_test)
    
    print('\nOrdinary least squares\n{0}\n'.format(forecast_set1))
    print('\nR^2\n{0}\n'.format(confidence1))
    print('\nnmean_squared_error\n{0}\n'.format(a))
    
    df['Forecast'] = np.nan
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day
    for i in forecast_set1:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
    
    plt.figure(figsize=(10,5), dpi=80)
    print(forecast_set1)
    print(df['Forecast'])
    df['Greece'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.title('Ordinary least squares Linear Regression for the total unemployment rates of active population in Greece', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    return