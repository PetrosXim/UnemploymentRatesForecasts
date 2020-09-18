from pandas import read_excel
import matplotlib.pyplot as plt
import statsmodels.api as sm  
import pandas as pd
import warnings
import itertools


def ARIMA(dt):

    # Greece
    series = dt[['Greece']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fit_values1 = OptimalParameters('Greece',series,8,'2000-04-01')
    fit_values1 = pd.DataFrame(fit_values1)
    fit_values1.columns = ['Greece']
    
    # Portugal
    series = dt[['Portugal']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fit_values2 = OptimalParameters('Portugal',series,7,'2000-02-01')
    fit_values2 = pd.DataFrame(fit_values2)
    fit_values2.columns = ['Portugal']
    fit_values = pd.concat([fit_values1, fit_values2], axis=1)
    
    # Netherlands
    series = dt[['Netherlands']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fit_values1 = OptimalParameters('Netherlands',series,7,'1985-01-01')
    fit_values1 = pd.DataFrame(fit_values1)
    fit_values1.columns = ['Netherlands']
    fit_values = pd.concat([fit_values, fit_values1], axis=1)
    
    # Latvia
    series = dt[['Latvia']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fit_values1 = OptimalParameters('Latvia',series,7,'2000-04-01')
    fit_values1 = pd.DataFrame(fit_values1)
    fit_values1.columns = ['Latvia']
    fit_values = pd.concat([fit_values, fit_values1], axis=1)
    
    # Poland
    series = dt[['Poland']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fit_values1 = OptimalParameters('Poland',series,7,'1999-01-01')
    fit_values1 = pd.DataFrame(fit_values1)
    fit_values1.columns = ['Poland']
    fit_values = pd.concat([fit_values, fit_values1], axis=1)
    return fit_values


def OptimalParameters(nm,series,tp,available_data):

    # Drop Rows with Empty Cells
    plt.style.use('fivethirtyeight')
  
    #===================================================
    #Identifying the parameters with smallest AIC
    #===================================================
    
    #Define the p, d and q parameters to take any value between 0 and 1
    p = d = q = range(0, 3)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore") # specify to ignore warning messages
        
    collectAICs=[]
    AIC = 10000
    par = (0,0,0)
    seasonal_par =(0,0,0)
    for param in pdq:
            
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(series,
                                                order=param,
                                                seasonal_order=param_seasonal,                                        
                                                enforce_invertibility=False)
                results = mod.fit()
                if results.aic<AIC:
                    par = param
                    seasonal_par = param_seasonal
                    AIC = results.aic
                    print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print('ARIMA{}x{} - AIC:{}'.format(par, seasonal_par, AIC))     
    fit_values = SARIMA(series, par, seasonal_par, nm, tp, available_data)
    return fit_values


def SARIMA(series, par, seasonal_par, nm, tp, available_data):
    plt.style.use('fivethirtyeight')
    
    #==================================================================
    #Fitting the ARIMA model and printing related statistics
    #this one is based on MA1 model template
    mod = sm.tsa.statespace.SARIMAX(series[nm], order=par, seasonal_order=seasonal_par)
    results = mod.fit(disp=False)
    print(results.summary())
    #==================================================================
    
    #GRAPH BLOCK1======================================================
    #Printing the graphical statistics of model (correlogram = ACF plot)
    results.plot_diagnostics(figsize=(15, 12))
    plt.show() 
    #==================================================================
    
    #GRAPH BLOCK2======================================================
    # printing the part of forecasts fitted to original data (for accuracy evaluation)
    # the start date has to be provided; hence should be a time within the original time series;
    # in this case, it is to start on 01 January 2000
    pred = results.get_prediction(start=pd.to_datetime(available_data), dynamic=False)
    pred_ci = pred.conf_int()
    
    # printing one-step ahead forecasts together with the original data set;
    # hence, the starting point (year) of the data set is required 
    # in order to build the plot of original series
    plt.subplots(figsize=(12,5), dpi= 100)
    ax = series[nm].plot(label='Original data')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7) 
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.legend()
    plt.show()
    #===================================================================   
    #GRAPH BLOCK3=======================================================
    # Get forecast 20 steps ahead in future
    pred_uc = results.get_forecast(steps=tp)
    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
    # plotting forecasts ahead
    plt.subplots(figsize=(12,5), dpi= 100)
    ax = series[nm].plot(label='Original data')
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    plt.legend()
    plt.show()
    print(pred_uc.predicted_mean)
    #====================================================================
    
    #====================================================================
    # MSE evaluation
    y_forecasted = pred.predicted_mean
    y_truth = series[nm]
    # Compute the mean square error
    MSE = ((y_forecasted - y_truth) ** 2).mean()
    print('MSE of the forecasts is {}'.format(round(MSE, 2)))
    #====================================================================
    return y_forecasted