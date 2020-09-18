from Modules.Preliminary import *
from Modules.ExponentialSmoothing import *
from Modules.Regression import *
from Modules.ARIMA import *
from Modules.MachineLearning import *
from Modules.FinalPlot import *


###################
### Preliminary ###
###################

### Import Data ###
dt,dt2,dt3 = ReadData()

### Time Plots ###
TimePlot(dt)

### Two Sides Plots ###
TwoSidesPlot(dt)

### Seasonal Plots ###
SeasonalPlot(dt)  #### #problem in returning

### Box Plots ###
Boxplot(dt)

### Autocorrelation Plot ###
ACF(dt)

### Additive & Multiplicative Decomposition ###
Decomposition(dt)



# #############################
# ### Exponential Smoothing ###
# #############################

### Transforms ###
TransformPlots(dt)
TransformSummary(dt)

### Holt-Winter`s Method ###
fitted_values1 = HoltWinterMethod(dt)



####################
### ARIMA Method ###
####################

### ARIMA Method ###
fitted_values2 = ARIMA(dt)


##################
### Regression ###
##################

### Summary Time Plots ###
SummaryTimePlot(dt2)
dt,dt2,dt3 = ReadData()

### Correlation ###
Correlation(dt2)
 
### Descriptive Statistics ###
DescriptiveStatistics(dt2)

### ANOVA Table ###
ANOVA(dt3)
dt,dt2,dt3 = ReadData()

### Regression  Method ###
Regression(dt3, fitted_values1)

### Weighted ANOVA Table ###
Weighted_ANOVA(dt3)
dt,dt2,dt3 = ReadData()

### Weighted Regression Method ###
Weighted_Regression(dt3, fitted_values1)

########################
### Machine Learning ###
########################

### Machine Learning ###
MachineLearning(dt3)

### Final ###
FinalPlot()
