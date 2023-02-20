# covid 19 Cases Prediction
 deep learning model to predict new cases of covid 19 in malaysia using the past 30 days number of cases
 
 
 ## data loading
 uses pandas to read csv file in the dataset folder
 
 ## EDA 
 check the data which have missing data and datatype
 
 ## data cleaning 
 change the datatype of newcases column 
 uses interpolate polinomial to convert NA to values
 
 ## feature selection 
 pick new cases as a target 
 
 ## model development 
change tha data to minmaxscaler
set the window size to 30 days
get x_train and y_train using for loop 

using LSTM, Dense, and Dropout layers to implement in the model.
LSTM layers set at 64
set callback with epoch 100


## compile train and test data
concate the train and test data 
make a prediction using model that has created

## graph 
change back mms to the actual data
and plot it using matplotlib

![img](/rahman_result.png)

source : GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.
