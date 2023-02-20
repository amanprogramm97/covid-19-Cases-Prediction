![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

# Covid 19 Cases Prediction
 - deep learning model to predict new cases of covid 19 in malaysia using the past 30 days number of cases
 
 
 ## Data loading
 - uses pandas to read csv file in the dataset folder
 
 ## EDA 
 - check the data which have missing data and datatype
 
 ## Data cleaning 
 - change the datatype of newcases column 
 - uses interpolate polinomial to convert NA to values
 
 ## Feature selection 
 - pick new cases as a target 
 
 ## Model development 
 - change tha data to minmaxscaler
 - set the window size to 30 days
 - get x_train and y_train using for loop 
 
 - using LSTM, Dense, and Dropout layers to implement in the model.
 - LSTM layers set at 64
 - set callback with epoch 100


 ## Compile train and test data
 - concate the train and test data 
 - make a prediction using model that has created
 - result of MAPE 
 - ![img](/MAPE_result.png)

 ## Graph 
 - change back mms to the actual data
 - and plot it using matplotlib

 ![img](/rahman_result.png)

 ## Data deployment
 - save model with pickel
 - save mms

 ## Acknowledgements

 - [MOH Malaysia](https://github.com/MoH-Malaysia/covid19-public)

