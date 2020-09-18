from pandas import read_excel
import matplotlib.pyplot as plt

def FinalPlot():
    Greece = read_excel('Dt.xls', sheet_name='Greece', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  
    Portugal = read_excel('Dt.xls', sheet_name='Portugal', header=0, 
              index_col=0, parse_dates=True, squeeze=True)     
    Netherlands = read_excel('Dt.xls', sheet_name='Netherlands', header=0, 
              index_col=0, parse_dates=True, squeeze=True) 
    Latvia = read_excel('Dt.xls', sheet_name='Latvia', header=0, 
              index_col=0, parse_dates=True, squeeze=True) 
    Poland = read_excel('Dt.xls', sheet_name='Poland', header=0, 
              index_col=0, parse_dates=True, squeeze=True) 
    
    
    plt.figure(figsize=(10,5), dpi=80)
    Greece['Original time series'].rename('Original time series').plot(color='steelblue', legend=True)
    Greece['Holt-Winter`s'].rename('Holt-Winter`s').plot(color='green', legend=True)
    Greece['ARIMA'].rename('ARIMA').plot(color='red', legend=True)
    Greece['Linear Regression'].rename('Linear Regression').plot(color='orange', legend=True)
    Greece['Weighted Linear Regression'].rename('Weighted Linear Regression').plot(color='orchid', legend=True)
    Greece['Linear Regression (ML)'].rename('Linear Regression (ML)').plot(color='grey', legend=True)
    Greece['Lasso'].rename('Lasso').plot(color='brown', legend=True)

    plt.legend(loc=2)
    plt.title('Comparison plot for the forecasts of the total unemployment rates of active population in Greece', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    
    plt.figure(figsize=(10,5), dpi=80)
    Portugal['Original time series'].rename('Original time series').plot(color='steelblue', legend=True)
    Portugal['Holt-Winter`s'].rename('Holt-Winter`s').plot(color='green', legend=True)
    Portugal['ARIMA'].rename('ARIMA').plot(color='red', legend=True)
    plt.legend(loc=2)
    plt.title('Comparison plot for the forecasts of the total unemployment rates of active population in Portugal', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    
    plt.figure(figsize=(10,5), dpi=80)
    Netherlands['Original time series'].rename('Original time series').plot(color='steelblue', legend=True)
    Netherlands['Holt-Winter`s'].rename('Holt-Winter`s').plot(color='green', legend=True)
    Netherlands['ARIMA'].rename('ARIMA').plot(color='red', legend=True)
    plt.legend(loc=2)
    plt.title('Comparison plot for the forecasts of the total unemployment rates of active population in Netherlands', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()

    plt.figure(figsize=(10,5), dpi=80)
    Latvia['Original time series'].rename('Original time series').plot(color='steelblue', legend=True)
    Latvia['Holt-Winter`s'].rename('Holt-Winter`s').plot(color='green', legend=True)
    Latvia['ARIMA'].rename('ARIMA').plot(color='red', legend=True)
    plt.legend(loc=2)
    plt.title('Comparison plot for the forecasts of the total unemployment rates of active population in Latvia', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    
    plt.figure(figsize=(10,5), dpi=80)
    Poland['Original time series'].rename('Original time series').plot(color='steelblue', legend=True)
    Poland['Holt-Winter`s'].rename('Holt-Winter`s').plot(color='green', legend=True)
    Poland['ARIMA'].rename('ARIMA').plot(color='red', legend=True)
    plt.legend(loc=2)
    plt.title('Comparison plot for the forecasts of the total unemployment rates of active population in Poland', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.show()
    return 