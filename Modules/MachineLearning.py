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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from statistics import mean 


def MachineLearning(dt):
    style.use('ggplot')
    forecast_out = 8
    
    dt.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

    series = dt[['Greece']].copy()
    y = series['Greece']
    y_true = y[-27:]


    y = np.array(y)
    y_true = np.array(y_true)
   
    
    df = dt
    X = dt.drop(columns=['Greece'])

    X_true = X[-27:]

    X = np.array(X)
    X_true = np.array(X_true)
    

    X_lately = read_excel('Dataset.xls', sheet_name='ML', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  


    ###############################   1st: Ordinary  Linear Regression     ###################################
    A1 = pd.DataFrame(columns=['AIC'])
    R = pd.DataFrame(columns=['R2'])
    matrix_1 = pd.DataFrame(columns=['Ob1', 'Ob2', 'Ob3','Ob4', 'Ob5', 'Ob6','Ob7', 'Ob8','R_square','mse','msa','SSR'])
    m1 = pd.DataFrame(columns=['O1', 'O2', 'O3','O4', 'O5','O6','O7', 'O8', 'Ο9', 'Ο10','Ο11', 'Ο12', 'O13', 
                              'O14', 'O15', 'O16', 'O17','O18','O19', 'O20','Ο21', 'Ο22', 'Ο23', 'Ο24','O25',
                              'O26', 'O27'])    
    coef1 = pd.DataFrame(columns=['b0', 'b1', 'b2', 'b3','b4'])  
    for i in range(1000):   

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
        model_1 = LinearRegression(fit_intercept=True, normalize=False)
        model_1.fit(X_train, y_train)
        
        forecast_set1 =  model_1.predict(X_lately)
        y_fitted = model_1.predict(X_test)
        y_test, y_fitted
        ## Measurements
        confidence1 = model_1.score(X_test, y_test)
        mse1 = mean_squared_error(y_test, y_fitted)
        mae1 = mean_absolute_error(y_test, y_fitted)
        R2 = r2_score(y_test, y_fitted)
        SSR1 = explained_variance_score(y_test, y_fitted)
        
        ## Coefficients
        coeffi = model_1.coef_
        interc = model_1.intercept_
        resid = y_fitted - y_test
        sse = sum(resid**2)
        k = 20
        a_i_c = 2*k - 2*np.log(sse)
        A1 = A1.append({'AIC': a_i_c}, ignore_index=True)
        R = R.append({'R2': R2}, ignore_index=True)
        matrix_1 = matrix_1.append({'Ob1': forecast_set1[0], 'Ob2': forecast_set1[1], 'Ob3': forecast_set1[2],'Ob4': forecast_set1[3], 'Ob5': forecast_set1[4], 'Ob6': forecast_set1[5],'Ob7': forecast_set1[6], 'Ob8': forecast_set1[7], 'R_square': confidence1, 'mse': mse1, 'mae': mae1, 'SSR': SSR1}, ignore_index=True)
        m1 = m1.append({'O1': y_test[0], 'O2': y_test[1], 'O3': y_test[2],'O4': y_test[3], 'O5': y_test[4], 
                              'O6': y_test[5],'O7': y_test[6], 'O8': y_test[7], 'Ο9': y_test[8], 'Ο10': y_test[9], 
                              'Ο11': y_test[10], 'Ο12': y_test[11], 'O13': y_test[12], 'O14': y_test[13], 'O15': y_test[14],
                              'O16': y_test[15], 'O17': y_test[16], 'O18': y_test[17],'O19': y_test[18], 'O20': y_test[19], 
                              'Ο21': y_test[20], 'Ο22': y_test[21], 'Ο23': y_test[22], 'Ο24': y_test[23],'O25': y_test[24], 
                              'O26': y_test[25], 'O27': y_test[26]}, ignore_index=True)
        coef1 = coef1.append({'b0': interc, 'b1': coeffi[0], 'b2': coeffi[1], 'b3': coeffi[2],'b4': coeffi[3]}, ignore_index=True)
    
    y1_test = []
    for i in range(27):
        n = m1.iloc[:,i].mean()
        y1_test.append(n)
    forecasts1 = []
    forecasts1_var = []
    for i in range(8):
        new = matrix_1.iloc[:,i].mean()
        forecasts1.append(new)
        new1 = matrix_1.iloc[:,i].var()
        forecasts1_var.append(new1)
        
    Coefficients1 = []
    for i in range(5):
        Coeff1 = coef1.iloc[:,i].mean()
        Coefficients1.append(Coeff1)
        
    aic = A1.iloc[:,0].mean()
    aic_v = A1.iloc[:,0].var()
    
    R2 = R.iloc[:,0].mean()
    R2_v = R.iloc[:,0].var()
    
    R_square1 = matrix_1['R_square'].mean()
    R_square1_var = matrix_1['R_square'].var()
    
    mse1 = matrix_1['mse'].mean()
    mse1_var = matrix_1['mse'].var()
    
    mae1 = matrix_1['mae'].mean()
    mae1_var = matrix_1['mae'].var()
    
    SSR1 = matrix_1['SSR'].mean()
    SSR1_var = matrix_1['SSR'].var()
    

    f = {'Forecasts': forecasts1}
    forec = pd.DataFrame(f, columns = ['Forecasts'], index=['2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01'])
    forec.index.name = 'Dates'
    
    result = pd.concat([dt['Greece'], forec['Forecasts']], axis=1, sort=False)
    result['Forecasts'].iloc[-8:] = forecasts1
    plt.figure(figsize=(10,5), dpi=80)
    result['Greece'].rename('Original time series').plot(color='steelblue', legend=True)
    result['Forecasts'].rename('Forecasted Values').plot(color='red', legend=True)
    plt.legend(loc=2)
    plt.title('Linear Regression for the total unemployment rates of active population in Greece', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    
    print('\n-------Linear Regression-------\n\n')
    print('Mean of forecasted values:\n {0}'.format(forecasts1))
    print('\nMean variance of forecasted values:\n {0}'.format(mean(forecasts1_var)))
    print('\nMean of R^2 of the prediction: {0}'.format(R_square1))
    print('Variance of R^2 of the prediction: {0}'.format(R_square1_var))
    print('\nmean_squared_error: {0}'.format(mse1))
    print('Variance of mean_squared_error: {0}'.format(mse1_var))
    print('\nSSR: {0}'.format(SSR1))
    print('Variance of SSR: {0}'.format(SSR1_var))
    print('\nMean_AIC: {0}'.format(aic))
    print('Variance of AIC: {0}\n'.format(aic_v))
    print('\nParameters: {0}\n'.format(model_1.get_params))
    print('\nCOEFFICIENTS: {0}\n'.format(model_1.coef_))
    print('\nINTERCEPT: {0}\n'.format(model_1.intercept_))
    


    ##############################  2nd: Ordinary  Lasso     ###################################
    A = pd.DataFrame(columns=['AIC'])
    R = pd.DataFrame(columns=['R2'])
    it = pd.DataFrame(columns=['Iter'])
    lamd = pd.DataFrame(columns=['L'])
    matrix_3 = pd.DataFrame(columns=['Ob1', 'Ob2', 'Ob3','Ob4', 'Ob5', 'Ob6','Ob7', 'Ob8','R_square','mse','msa','SSR'])
    m3 = pd.DataFrame(columns=['O1', 'O2', 'O3','O4', 'O5','O6','O7', 'O8', 'Ο9', 'Ο10','Ο11', 'Ο12', 'O13', 
                              'O14', 'O15', 'O16', 'O17','O18','O19', 'O20','Ο21', 'Ο22', 'Ο23', 'Ο24','O25',
                              'O26', 'O27'])  
    coef3 = pd.DataFrame(columns=['b0', 'b1', 'b2', 'b3','b4'])  
     
    for i in range(1000):   
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
        model_3 = linear_model.LassoLarsIC(criterion='aic', fit_intercept=True, normalize=False)
        model_3.fit(X_train, y_train)
        
        forecast_set3 =  model_3.predict(X_lately)
        y_fitted3 = model_3.predict(X_test)
        
        # Forecasts
        confidence3 = model_3.score(X_test, y_test)
        forecast_set3 =  model_3.predict(X_lately)
        
        # Measurements 
        mse3 = mean_squared_error(y_test, y_fitted3)
        mae3 = mean_absolute_error(y_test, y_fitted3)
        SSR3 = explained_variance_score(y_test, y_fitted3)
        R2 = r2_score(y_test, y_fitted3)
        lam = model_3.alpha_
        coeffi = model_3.coef_
        interc = model_3.intercept_
        a_i_c = min(model_3.criterion_)
        ite = model_3.n_iter_
        R = R.append({'R2': R2}, ignore_index=True)
        A = A.append({'AIC': a_i_c}, ignore_index=True)
        lamd = lamd.append({'L': lam}, ignore_index=True)
        matrix_3 = matrix_3.append({'Ob1': forecast_set3[0], 'Ob2': forecast_set3[1], 'Ob3': forecast_set3[2],'Ob4': forecast_set3[3], 'Ob5': forecast_set3[4], 'Ob6': forecast_set3[5],'Ob7': forecast_set3[6], 'Ob8': forecast_set3[7], 'R_square': confidence3, 'mse': mse3, 'mae': mae3, 'SSR': SSR3}, ignore_index=True)
        m3 = m3.append({'O1': y_test[0], 'O2': y_test[1], 'O3': y_test[2],'O4': y_test[3], 'O5': y_test[4], 
                              'O6': y_test[5],'O7': y_test[6], 'O8': y_test[7], 'Ο9': y_test[8], 'Ο10': y_test[9], 
                              'Ο11': y_test[10], 'Ο12': y_test[11], 'O13': y_test[12], 'O14': y_test[13], 'O15': y_test[14],
                              'O16': y_test[15], 'O17': y_test[16], 'O18': y_test[17],'O19': y_test[18], 'O20': y_test[19], 
                              'Ο21': y_test[20], 'Ο22': y_test[21], 'Ο23': y_test[22], 'Ο24': y_test[23],'O25': y_test[24], 
                              'O26': y_test[25], 'O27': y_test[26]}, ignore_index=True)
        coef3 = coef3.append({'b0': interc, 'b1': coeffi[0], 'b2': coeffi[1], 'b3': coeffi[2],'b4': coeffi[3],'b5': coeffi[4], 'b6': coeffi[5], 'b7': coeffi[6], 'b8': coeffi[7],'b9': coeffi[8],
                              'b10': coeffi[9], 'b11': coeffi[10], 'b12': coeffi[11], 'b13': coeffi[12],'b14': coeffi[13],'b15': coeffi[14], 'b16': coeffi[15], 'b17': coeffi[16], 'b18': coeffi[17],'b19': coeffi[18]}, ignore_index=True)
        it = it.append({'Iter': ite}, ignore_index=True)
        
        
    y3_test = []
    for i in range(27):
        n = m3.iloc[:,i].mean()
        y3_test.append(n)
    forecasts3 = []
    forecasts3_var = []
    for i in range(8):
        new = matrix_3.iloc[:,i].mean()
        forecasts3.append(new)
        new1 = matrix_3.iloc[:,i].var()
        forecasts3_var.append(new1)
    Coefficients3 = []
    Coefficients3_var = []
    for i in range(20):
        Coeff3 = coef3.iloc[:,i].mean()
        coe3 = coef3.iloc[:,i].var()
        Coefficients3.append(Coeff3)
        Coefficients3_var.append(coe3)
        
        
    L = lamd.iloc[:,0].mean()
    L_v = lamd.iloc[:,0].var()
    
    aic = A.iloc[:,0].mean()
    aic_v = A.iloc[:,0].var()
    
    R3 = R.iloc[:,0].mean()
    R3_v = R.iloc[:,0].var()
    
    iterations_mean = it.iloc[:,0].mean()
    iterations_variance = it.iloc[:,0].var()

    R_square3 = matrix_3['R_square'].mean()
    R_square3_var = matrix_3['R_square'].var()
    mse3 = matrix_3['mse'].mean()
    mse_var3 = matrix_3['mse'].var()
    mae3 = matrix_3['mae'].mean()
    SSR3 = matrix_3['SSR'].mean()
    SSR3_var = matrix_3['SSR'].var()
       
    
    
    f = {'Forecasts': forecasts3}
    forec = pd.DataFrame(f, columns = ['Forecasts'], index=['2020-05-01','2020-06-01','2020-07-01','2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01'])
    forec.index.name = 'Dates'
    
    result = pd.concat([dt['Greece'], forec['Forecasts']], axis=1, sort=False)

    result['Forecasts'].iloc[-8:] = forecasts3
    plt.figure(figsize=(10,5), dpi=80)
    result['Greece'].rename('Original time series').plot(color='steelblue', legend=True)
    result['Forecasts'].rename('Forecasted Values').plot(color='red', legend=True)
    plt.legend(loc=2)
    plt.title('Lasso for the total unemployment rates of active population in Greece', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    
    print('\n-------Lasso-------\n\n')
    print('Mean of forecasted values:\n{0}'.format(forecasts3))
    print('\nVariance of forecasted values:\n{0}'.format(mean(forecasts3_var)))
    print('\nMean of b of the prediction:\n{0}'.format(Coefficients3))
    print('Variance of b of the prediction:\n{0}'.format(Coefficients3_var))
    print('\nMean of R^2 of the prediction: {0}'.format(R_square3))
    print('Variance of R^2 of the prediction: {0}'.format(R_square3_var))
    print('\nMean of iterations of the prediction: {0}'.format(iterations_mean))
    print('Variance of iterations of the prediction: {0}'.format(iterations_variance))
    print('\nmean_squared_error: {0}'.format(mse3))
    print('Variance of mean_squared_error: {0}'.format(mse_var3))
    print('\nSSR: {0}'.format(SSR3))
    print('Variance of SSR: {0}'.format(SSR3_var))
    print('\nMean lambda-value: {0}'.format(L))
    print('Var lambda-value: {0}'.format(L_v))
    print('\nMean_AIC: {0}'.format(aic))
    print('Variance of AIC: {0}\n'.format(aic_v))
    print('\nParameters: {0}'.format(model_3.get_params))
    return
    
    
