import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error 
from statsmodels.graphics.tsaplots import plot_acf

def TransformPlots(dt):
    names = dt.columns.tolist()
    for j in range(len(dt.columns)):
        # Import Data
        series = dt[[names[j]]].copy()
        
        # Drop Rows with Empty Cells
        series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            
        series['log'] = np.log(series[names[j]])
        series['sqrt'] = np.sqrt(series[names[j]])


        fig, axs = plt.subplots(2, 3, figsize=(20,7), dpi= 80)
        axs[0, 0].plot(series[names[j]])
        axs[0, 0].set_title('Original Data')
        axs[0, 1].plot(series[names[j]])
        axs[0, 1].set_title('Log Tranformed')
        axs[0, 2].plot(series['log'])
        axs[0, 2].set_title('Sqrt Tranformed')
        axs[1, 0].hist(series[names[j]])
        axs[1, 1].hist(series['log'])
        axs[1, 2].hist(series['sqrt'])
        num = 0
        for ax in axs.flat:
            if (num < 3):
                ax.set(xlabel='Year', ylabel='Unemployment Rate')
            else:
                ax.set(xlabel='Observations', ylabel='Unemployment Rate')
            num = num + 1
    return


def TransformSummary(dt):
    names = dt.columns.tolist()
    for j in range(len(dt.columns)):
        # Import Data
        series = dt[[names[j]]].copy()
        
        # Drop Rows with Empty Cells
        series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        
        #===========================
        # Scenario #1: Original Data
        #===========================
        fit1 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add').fit()   
        fit2 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='mul').fit()   
        
        MSE1=mean_squared_error(fit1.fittedvalues, series)
        MSE2=mean_squared_error(fit2.fittedvalues, series)
        
        results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
        params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
        results["HW-Additive"] = [fit1.params[p] for p in params] + [MSE1]
        results["HW-Multiplicative"] = [fit2.params[p] for p in params] + [MSE2]
        
        print("Parameters of Holt-Winters Method for {0} Original Data".format(names[j]))
        print(results)
        
        if names[j]=='Italy':
            continue
        #=================================
        # Scenario #2: Log Transform Data
        #=================================

        series2=np.log(series)    # Transform 
        fit3 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='add').fit()   # Exponential Smoothing
        fit4 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='mul').fit()   # Exponential Smoothing
        fit3.fittedvalues=np.exp(abs(fit3.fittedvalues))     #Back Transform
        fit4.fittedvalues=np.exp(abs(fit4.fittedvalues))     #Back Transform
    
        MSE3=mean_squared_error(fit3.fittedvalues, series)
        MSE4=mean_squared_error(fit4.fittedvalues, series)
        
        results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
        params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
        results["HW-Additive"] = [fit3.params[p] for p in params] + [MSE3]
        results["HW-Multiplicative"] = [fit4.params[p] for p in params] + [MSE4]
        
        print("Parameters of Holt-Winters Method for {0} Log Transformed Data".format(names[j]))
        print(results)
        
        #=================================
        # Scenario #3: Sqrt Transform Data
        #=================================
        series3=np.sqrt(series)     # Transform 
        
        fit5 = ExponentialSmoothing(series3, seasonal_periods=12, trend='add', seasonal='add').fit()  # Exponential Smoothing
        fit6 = ExponentialSmoothing(series3, seasonal_periods=12, trend='add', seasonal='mul').fit()  # Exponential Smoothing
        
        fit5.fittedvalues=np.power(fit5.fittedvalues,2)     #Back Transform
        fit6.fittedvalues=np.power(fit6.fittedvalues,2)     #Back Transform      
        
        MSE5=mean_squared_error(fit5.fittedvalues, series)
        MSE6=mean_squared_error(fit6.fittedvalues, series)
        
        results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
        params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
        results["HW-Additive"] = [fit5.params[p] for p in params] + [MSE5]
        results["HW-Multiplicative"] = [fit6.params[p] for p in params] + [MSE6]
        
        print("Parameters of Holt-Winters Method for {0} Sqrt Transformed Data".format(names[j]))
        print(results)
        # if names[j]=='Cyprus':
        #     break
    return


def HW_Forecasting(names,series,transform,tp,timepoints):
    
    if transform=='original':
        if tp=='add':
            # ===================================
            # Model: Holt-Winter method with additive trend and seasonality (Original time series) 
            # Here, the parameters alpha, beta, and gamma are optimized
            # ===================================
            plt.figure(figsize=(12,8),dpi= 100)
            series2=series
            fit1 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='add').fit()
            fit1.fittedvalues.rename('Opt HW-Additive seasonality').plot(color='red', legend=True)
            #========================
            # Time and forecast plots 
            #========================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            new=fit1.forecast(timepoints)
            LowerE = new - 1.960*MSE1
            UpperE = new + 1.960*MSE1            
            new=fit1.forecast(timepoints)            
            series[names].rename('Time plot of original series').plot(color='steelblue', legend=True)       
            LowerE.rename('95% Confidence Intervals').plot(color='darkslategray', legend=True)
            UpperE.plot(color='darkslategray')
            new.plot(color='red')
            plt.xlabel('Dates')
            plt.ylabel('Unemployment percentage\n of active population')
            plt.title("Forecasting of unemployment rates in {0} with Holt-Winters method".format(names))
            plt.show()
            print('The forecasted unemployment rates of {0} with the H-W`s method until December of 2020 are:\n\n{1}'.format(names,new))
            #======================
            # Evaluating the errors 
            #======================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            A = fit1.fittedvalues[24:]
            B = series[names].values[24:]
            MSE2=round(mean_squared_error(A, B),2)
            print('MSE1 value: {0}'.format(MSE1))
            print('MSE2 value: {0}'.format(MSE2))
            #====================================
            # Printing the parameters and errors 
            #====================================
            results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
            params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
            results["HW-additive of {0}".format(names)] = [fit1.params[p] for p in params] + [MSE1]
            print(results)
            #============================================
            # Evaluating and plotting the residual series
            #============================================
            plt.figure(figsize=(12,8),dpi= 100)
            residuals1= fit1.fittedvalues - series[names].values
            residuals1.rename('HW-additive - Original time series').plot(color='red', legend=True)
            plt.title('Residual plots of the model for {0}'.format(names))
            plt.show()
            #=================================
            # ACF plots of the residual series 
            #=================================
            plot_acf(residuals1, title='Residual ACF for HW-additive - Original time series', lags=50)
            plt.show()

        else:
            # ===================================
            # Model: Holt-Winter method with additive trend and multiplicative seasonality (Original time series)
            # Here, the parameters alpha, beta, and gamma are optimized
            # ===================================
            plt.figure(figsize=(12,8),dpi= 100)
            series2=series
            fit1 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='mul').fit()
            fit1.fittedvalues.rename('Opt HW-Multiplicative seasonality').plot(color='red', legend=True)
            #=====================================
            # Time and forecast plots 
            #=====================================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            new=fit1.forecast(timepoints)
            LowerE = new - 1.960*MSE1
            UpperE = new + 1.960*MSE1            
            new=fit1.forecast(timepoints)            
            series[names].rename('Time plot of original series').plot(color='steelblue', legend=True)       
            LowerE.rename('95% Confidence Intervals').plot(color='darkslategray', legend=True)
            UpperE.plot(color='darkslategray')
            new.plot(color='red')
            plt.xlabel('Dates')
            plt.ylabel('Unemployment percentage\n of active population')
            plt.title("Forecasting of unemployment rates in {0} with Holt-Winters method".format(names))
            plt.show()
            print('The forecasted unemployment rates of {0} with the H-W`s method until December of 2020 are:\n\n{1}'.format(names,new))
            #======================
            # Evaluating the errors 
            #======================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            A = fit1.fittedvalues[24:]
            B = series[names].values[24:]
            MSE2=round(mean_squared_error(A, B),2)
            print('MSE1 value: {0}'.format(MSE1))
            print('MSE2 value: {0}'.format(MSE2))
            #==================================
            # Printing the paramters and errors
            #==================================
            results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
            params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
            results["HW-multiplicative of {0}".format(names)] = [fit1.params[p] for p in params] + [MSE1]
            print(results)
            #============================================
            # Evaluating and plotting the residual series 
            #============================================
            plt.figure(figsize=(12,8),dpi= 100)
            residuals1= fit1.fittedvalues - series[names].values
            residuals1.rename('HW-multiplicative - Original time series').plot(color='red', legend=True)
            plt.title('Residual plots of the model for {0}'.format(names))
            plt.show()
            #=================================
            # ACF plots of the residual series 
            #=================================
            plot_acf(residuals1, title='Residual ACF for HW-multiplicative - Original time series', lags=50)
            plt.show()
    if transform=='log':
        if tp=='add':
            # ===================================
            # Model: Holt-Winter method with additive trend and seasonality (Log transformed time series) 
            # Here, the parameters alpha, beta, and gamma are optimized
            # ===================================
            plt.figure(figsize=(12,8),dpi= 100)
            series2=np.log(series)
            fit1 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='add').fit()
            fit1.fittedvalues=np.exp(fit1.fittedvalues) 
            fit1.fittedvalues.rename('Opt HW-Additive seasonality').plot(color='red', legend=True)
            #========================
            # Time and forecast plots 
            #========================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            new=np.exp(fit1.forecast(timepoints))
            LowerE = new - 1.960*MSE1
            UpperE = new + 1.960*MSE1            
            new=np.exp(fit1.forecast(timepoints))      
            series[names].rename('Time plot of original series').plot(color='steelblue', legend=True)       
            LowerE.rename('95% Confidence Intervals').plot(color='darkslategray', legend=True)
            UpperE.plot(color='darkslategray')
            new.plot(color='red')
            plt.xlabel('Dates')
            plt.ylabel('Unemployment percentage\n of active population')
            plt.title("Forecasting of unemployment rates in {0} with Holt-Winters method".format(names))
            plt.show()
            print('The forecasted unemployment rates of {0} with the H-W`s method until December of 2020 are:\n\n{1}'.format(names,new))
            #======================
            # Evaluating the errors 
            #======================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            A = fit1.fittedvalues[24:]
            B = series[names].values[24:]
            MSE2=round(mean_squared_error(A, B),2)
            print('MSE1 value: {0}'.format(MSE1))
            print('MSE2 value: {0}'.format(MSE2))
            #==================================
            # Printing the paramters and errors 
            #==================================
            results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
            params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
            results["HW-additive of {0}".format(names)] = [fit1.params[p] for p in params] + [MSE1]
            print(results)
            #============================================
            # Evaluating and plotting the residual series 
            #============================================
            residuals1= fit1.fittedvalues - series[names].values
            residuals1.rename('HW-additive - Log transformed time series').plot(color='red', legend=True)
            plt.title('Residual plots of the model for {0}'.format(names))
            plt.show()
            #=================================
            # ACF plots of the residual series 
            #=================================
            plot_acf(residuals1, title='Residual ACF for HW-additive - Log transfomed time series', lags=50)
            plt.show()
        else:
            # ===================================
            # Model: Holt-Winter method with additive trend and multiplicative seasonality (Log transformed time series)
            # Here, the parameters alpha, beta, and gamma are optimized
            # ===================================
            plt.figure(figsize=(12,8),dpi= 100)
            series2=np.log(series)
            fit1 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='mul').fit()
            fit1.fittedvalues=np.exp(fit1.fittedvalues)
            fit1.fittedvalues.rename('Opt HW-Multiplicative seasonality').plot(color='red', legend=True)
            #=====================================
            # Time and forecast plots 
            #=====================================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            new=np.exp(fit1.forecast(timepoints))
            LowerE = new - 1.960*MSE1
            UpperE = new + 1.960*MSE1            
            new=np.exp(fit1.forecast(timepoints))      
            series[names].rename('Time plot of original series').plot(color='steelblue', legend=True)       
            LowerE.rename('95% Confidence Intervals').plot(color='darkslategray', legend=True)
            UpperE.plot(color='darkslategray')
            new.plot(color='red')
            plt.xlabel('Dates')
            plt.ylabel('Unemployment percentage\n of active population')
            plt.title("Forecasting of unemployment rates in {0} with Holt-Winters method".format(names))
            plt.show()
            print('The forecasted unemployment rates of {0} with the H-W`s method until December of 2020 are:\n\n{1}'.format(names,new))
            #======================
            # Evaluating the errors 
            #======================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            A = fit1.fittedvalues[24:]
            B = series[names].values[24:]
            MSE2=round(mean_squared_error(A, B),2)
            print('MSE1 value: {0}'.format(MSE1))
            print('MSE2 value: {0}'.format(MSE2))
            #==================================
            # Printing the paramters and errors 
            #==================================
            results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
            params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
            results["HW-multiplicative of {0}".format(names)] = [fit1.params[p] for p in params] + [MSE1]
            print(results)
            #============================================
            # Evaluating and plotting the residual series 
            #============================================
            plt.figure(figsize=(12,8),dpi= 100)
            residuals1= fit1.fittedvalues - series[names].values
            residuals1.rename('HW-multiplicative - Log transformed time series').plot(color='red', legend=True)
            plt.title('Residual plots of the model for {0}'.format(names))
            plt.show()
            #=================================
            # ACF plots of the residual series 
            #=================================
            plot_acf(residuals1, title='Residual ACF for HW-multiplicative - Log transformed time series', lags=50)
            plt.show()
            
    if  transform=='sqrt':
        if tp=='add':
            # ===================================
            # Model: Holt-Winter method with additive trend and seasonality (Sqrt transformed time series) 
            # Here, the parameters alpha, beta, and gamma are optimized
            # ===================================
            plt.figure(figsize=(12,8),dpi= 100)
            series2=np.sqrt(series)
            fit1 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='add').fit()
            fit1.fittedvalues=np.power(fit1.fittedvalues,2)
            fit1.fittedvalues.rename('Opt HW-Additive seasonality').plot(color='red', legend=True)
            #========================
            # Time and forecast plots 
            #=========================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            new=np.power(fit1.forecast(timepoints),2)
            LowerE = new - 1.960*MSE1
            UpperE = new + 1.960*MSE1            
            new=np.power(fit1.forecast(timepoints),2)
            series[names].rename('Time plot of original series').plot(color='steelblue', legend=True)
            LowerE.rename('95% Confidence Intervals').plot(color='darkslategray', legend=True)
            UpperE.plot(color='darkslategray')
            new.plot(color='red')
            plt.xlabel('Dates')
            plt.ylabel('Unemployment percentage\n of active population')
            plt.title("Forecasting of unemployment rates in {0} with Holt-Winters method".format(names))
            plt.show()
            print('The forecasted unemployment rates of {0} with the H-W`s method until December of 2020 are:\n\n{1}'.format(names,new))
            #======================
            # Evaluating the errors 
            #======================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            A = fit1.fittedvalues[24:]
            B = series[names].values[24:]
            MSE2=round(mean_squared_error(A, B),2)
            print('MSE1 value: {0}'.format(MSE1))
            print('MSE2 value: {0}'.format(MSE2))
            #==================================
            # Printing the paramters and errors
            #==================================
            results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
            params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
            results["HW-additive of {0}".format(names)] = [fit1.params[p] for p in params] + [MSE1]
            print(results)
            #============================================
            # Evaluating and plotting the residual series 
            #============================================
            plt.figure(figsize=(12,8),dpi= 100)
            residuals1= fit1.fittedvalues - series[names].values
            residuals1.rename('HW-additive - Sqrt transformed time series').plot(color='red', legend=True)
            plt.title('Residual plots of the model for {0}'.format(names))
            plt.show()
            #=================================
            # ACF plots of the residual series 
            #=================================
            plot_acf(residuals1, title='Residual ACF for HW-additive - Sqrt transformed time series', lags=50)
            plt.show()
        else:
            # ===================================
            # Model: Holt-Winter method with additive trend and multiplicative seasonality (Sqrt transformed time series)
            # Here, the parameters alpha, beta, and gamma are optimized
            # ===================================
            plt.figure(figsize=(12,8),dpi= 100)
            series2=np.sqrt(series)
            fit1 = ExponentialSmoothing(series2, seasonal_periods=12, trend='add', seasonal='mul').fit()
            fit1.fittedvalues=np.power(fit1.fittedvalues,2) 
            fit1.fittedvalues.rename('Opt HW-Multiplicative seasonality').plot(color='red', legend=True)
            #========================
            # Time and forecast plots 
            #========================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            new=np.power(fit1.forecast(timepoints),2)
            LowerE = new - 1.960*MSE1
            UpperE = new + 1.960*MSE1            
            new=np.power(fit1.forecast(timepoints),2)               
            series[names].rename('Time plot of original series').plot(color='steelblue', legend=True)       
            LowerE.rename('95% Confidence Intervals').plot(color='darkslategray', legend=True)
            UpperE.plot(color='darkslategray')
            new.plot(color='red')
            plt.xlabel('Dates')
            plt.ylabel('Unemployment percentage\n of active population')
            plt.title("Forecasting of unemployment rates in {0} with Holt-Winters method".format(names))
            plt.show()
            print('The forecasted unemployment rates of {0} with the H-W`s method until December of 2020 are:\n\n{1}'.format(names,new))
            #======================
            # Evaluating the errors 
            #======================
            MSE1=round(mean_squared_error(fit1.fittedvalues, series[names].values),2)
            A = fit1.fittedvalues[24:]
            B = series[names].values[24:]
            MSE2=round(mean_squared_error(A, B),2)
            print('MSE1 value: {0}'.format(MSE1))
            print('MSE2 value: {0}'.format(MSE2))
            #==================================
            # Printing the paramters and errors 
            #==================================
            results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
            params = ['smoothing_level', 'smoothing_slope', 'smoothing_seasonal', 'initial_level', 'initial_slope']
            results["HW-multiplicative of {0}".format(names)] = [fit1.params[p] for p in params] + [MSE1]
            print(results)
            #============================================
            # Evaluating and plotting the residual series 
            #============================================
            plt.figure(figsize=(12,8),dpi= 100)
            residuals1= fit1.fittedvalues - series[names].values
            residuals1.rename('HW-multiplicative - Sqrt transformed time series').plot(color='red', legend=True)
            plt.title('Residual plots of the model for {0}'.format(names))
            plt.show()
            #=================================
            # ACF plots of the residual series 
            #=================================
            plot_acf(residuals1, title='Residual ACF for HW-multiplicative - Sqrt transformed time series', lags=50)
            plt.show()

    return fit1.fittedvalues


def HoltWinterMethod(dt):
    # Greece
    series = dt[['Greece']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fitted_values1 = HW_Forecasting('Greece',series,'sqrt','add',8)
    fitted_values1 = pd.DataFrame(fitted_values1)
    fitted_values1.columns = ['Greece']
    HW_Forecasting('Greece',series,'sqrt','mul',8) #
    
    # Portugal
    series = dt[['Portugal']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fitted_values2 = HW_Forecasting('Portugal',series,'sqrt','add',7)
    fitted_values2 = pd.DataFrame(fitted_values2)
    fitted_values2.columns = ['Portugal']  
    fitted_values = pd.concat([fitted_values1, fitted_values2], axis=1)
    HW_Forecasting('Portugal',series,'sqrt','mul',7)
    
    # Netherlands
    series = dt[['Netherlands']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fitted_values1 = HW_Forecasting('Netherlands',series,'sqrt','add',7)
    fitted_values1 = pd.DataFrame(fitted_values1)
    fitted_values1.columns = ['Netherlands']
    fitted_values = pd.concat([fitted_values, fitted_values1], axis=1)
    HW_Forecasting('Netherlands',series,'sqrt','mul',7)
    
    # Latvia
    series = dt[['Latvia']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fitted_values1 = HW_Forecasting('Latvia',series,'log','add',7)  
    fitted_values1 = pd.DataFrame(fitted_values1)
    fitted_values1.columns = ['Latvia']
    fitted_values = pd.concat([fitted_values, fitted_values1], axis=1)
    HW_Forecasting('Latvia',series,'log','mul',7)
    
    # Poland
    series = dt[['Poland']].copy()
    series.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    fitted_values1 = HW_Forecasting('Poland',series,'original','add',7)  
    fitted_values1 = pd.DataFrame(fitted_values1)
    fitted_values1.columns = ['Poland']
    fitted_values = pd.concat([fitted_values, fitted_values1], axis=1)
    HW_Forecasting('Poland',series,'original','mul',7)
    return fitted_values


