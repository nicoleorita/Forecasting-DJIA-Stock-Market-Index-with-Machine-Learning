# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:32:17 2019

@author: Nicole Rita
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 12:27:13 2018

@author: Nicole Rita
"""
import os
import numpy as np
import pandas as pd
import glob
import pickle
from datetime import datetime
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# style.use('ggplot')
scaler = MinMaxScaler()


#
aaba = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\AABA_2006-01-01_to_2018-01-01.csv") 
aapl = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\AAPL_2006-01-01_to_2018-01-01.csv") 
amzn = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\AMZN_2006-01-01_to_2018-01-01.csv") 
axp = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\AXP_2006-01-01_to_2018-01-01.csv") 
ba = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\BA_2006-01-01_to_2018-01-01.csv") 
cat = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\CAT_2006-01-01_to_2018-01-01.csv") 
csco = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\CSCO_2006-01-01_to_2018-01-01.csv") 
cvx = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\CVX_2006-01-01_to_2018-01-01.csv") 
dis= pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\DIS_2006-01-01_to_2018-01-01.csv") 
ge = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\GE_2006-01-01_to_2018-01-01.csv") 
googl = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\GOOGL_2006-01-01_to_2018-01-01.csv") 
gs = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\GS_2006-01-01_to_2018-01-01.csv") 
hd = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\HD_2006-01-01_to_2018-01-01.csv") 
ibm = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\IBM_2006-01-01_to_2018-01-01.csv") 
intc = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\INTC_2006-01-01_to_2018-01-01.csv") 
jnj = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\JNJ_2006-01-01_to_2018-01-01.csv") 
jpm= pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\JPM_2006-01-01_to_2018-01-01.csv") 
ko = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\KO_2006-01-01_to_2018-01-01.csv") 
mcd = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\MCD_2006-01-01_to_2018-01-01.csv") 
mmm = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\MMM_2006-01-01_to_2018-01-01.csv") 
mrk = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\MRK_2006-01-01_to_2018-01-01.csv") 
msft = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\MSFT_2006-01-01_to_2018-01-01.csv") 
nke = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\NKE_2006-01-01_to_2018-01-01.csv") 
pfe = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\PFE_2006-01-01_to_2018-01-01.csv") 
pg = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\PG_2006-01-01_to_2018-01-01.csv") 
trv = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\TRV_2006-01-01_to_2018-01-01.csv") 
unh = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\UNH_2006-01-01_to_2018-01-01.csv") 
utx = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\UTX_2006-01-01_to_2018-01-01.csv") 
vz = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\VZ_2006-01-01_to_2018-01-01.csv") 
wmt = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\WMT_2006-01-01_to_2018-01-01.csv") 
xom = pd.read_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\XOM_2006-01-01_to_2018-01-01.csv") 
#

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name


def import_df(df_name):
    path =r'C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset' # use my path
    for i in os.listdir(path):
        if i == df_name:
            path2 =path + "\\\\" + i
            cena = glob.glob(path2 +  "\\*.csv")
            df = pd.read_csv(cena[0], index_col =None, header=0)
    return df

#Fill missing values
def fillMissingValues(df):
    df.fillna(df.median(), inplace=True)
    return df

#add "Daily Change new var"
def addChange(df):
    aux=[]
    for i in range(len(df)):
        if i == 0:
            aux.append('0')
        else:
            change = df['Close'][i] - df['Close'][i-1]
            aux.append(change)
    df['dailyChangeClose'] = aux
    return df

#########JUST FOR GRAPH PURPOSE##########
#Convert dates into same format and datetime format
def guess_date(string):
    for fmt in ["%Y-%m-%d", "%d-%m-%Y"]:
        try:
            return datetime.strptime(string, fmt).date()
        except ValueError:
            continue
    raise ValueError(string)

def convertToDate(df): 
    df['Date'] = df['Date'].str.replace('/','-')
    new_dates = list()
    for d in df['Date']:
        d = guess_date(d)
        aux = datetime.strftime(d,"%Y-%m-%d" )
        aux2 = guess_date(aux)
        new_dates.append(aux2)
    df['Date'] = new_dates
    return df

#-------------------------------------------SWAP COLUMS - TARGET VAR TO LAST COLUMN-------------------------------------#

def swap_cols(df):
    df = df.drop(['Name'], 1)
    df = df.drop(['Date'], 1)
    col_list = list(df)
    col_list[3], col_list[4] = col_list[4] , col_list[3] 
    ''' 3, 5 //5 , 3 '''
    df = df.loc[:, col_list]
    
    return df

#-------------------------------------------SPLIT TRAIN AND TEST DATA-------------------------------------#
def splitData(df):
    n = df.shape[0]
    p = df.shape[1]
    
    #training and test data
    train_end = int(np.floor(0.8*n))
    test_start = train_end
    test_end = n 
    
    data_train = df[:train_end]
    data_test = df[test_start:test_end]
    # Build X and y
    X_train = data_train.loc[:, :'Volume'] 
    y_train = data_train[['Close']] 
    X_test = data_test.loc[:, :'Volume'] 
    y_test = data_test[['Close']]
    
    return X_train, y_train, X_test, y_test

#-------------------------------------------NORMALIZATION-------------------------------------#
# Scale the data to be between 0 and 1
#normalize both test and train data with respect to training data

def normalize_df(X_train, X_test, y_train, y_test):
    #NORMALIZE X_TRAIN
    x = X_train.values #returns a numpy array
    x_scaled = scaler.fit_transform(X=x)
    X_train_norm = pd.DataFrame(x_scaled)
    
    #NORMALIZE X_TEST
    x_test = X_test.values #returns a numpy array
    x_scaled = scaler.fit_transform(x_test)
    X_test_norm = pd.DataFrame(x_scaled)
    
    #NORMALIZE y_train
    Y = y_train.values #returns a numpy array
    x_scaled = scaler.fit_transform(Y)
    y_train_norm = pd.DataFrame(x_scaled)
    
    #NORMALIZE y_test
    Y_test = y_test.values #returns a numpy array
    x_scaled = scaler.fit_transform(Y_test)
    y_test_norm = pd.DataFrame(x_scaled)
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm

#-----Convert dataframes into numpy arrays of 3 dimensions
def df_to_3d(X_train_norm, y_train_norm, X_test_norm, y_test_norm):
    #--TRAIN
    X_train_norm = X_train_norm.values
    y_train_norm = y_train_norm.values
    
    n = X_train_norm.shape[0]
    L = 3
    X_train_seq = []
    Y_train_seq = []
    for k in range(n - L + 1):
        X_train_seq.append(X_train_norm[k : k + L])
        Y_train_seq.append(y_train_norm[k : k + L])
    
    X_train_seq = np.array(X_train_seq)
    Y_train_seq = np.array(Y_train_seq)
    
    #--TEST
    X_test_norm = X_test_norm.values
    y_test_norm = y_test_norm.values
    
    n = X_test_norm.shape[0]
    L = 3
    X_test_seq = []
    Y_test_seq = []
    for k in range(n - L + 1):
        X_test_seq.append(X_test_norm[k : k + L])
        Y_test_seq.append(y_test_norm[k : k + L])
    
    X_test_seq = np.array(X_test_seq)
    Y_test_seq = np.array(Y_test_seq)
    
    return X_train_seq, Y_train_seq, X_test_seq, Y_test_seq
    
#----------------------LSTM
def lstm_alg(X_train_seq, Y_train_seq):
    regressor = Sequential()
    
    regressor.add(LSTM(32, input_shape=(3,4), return_sequences=True))
    #regressor.add(LSTM(32, input_shape=(25,5), return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(1,kernel_initializer='uniform',activation='linear'))
    
    regressor.compile(loss='mse',optimizer='adam', metrics=['mae'])
    # Fitting the RNN to the Training set
    history = regressor.fit(X_train_seq, Y_train_seq, epochs = 1000, batch_size = 32, validation_split=0.1, verbose=0)
    return history, regressor

#----------------Convolutional NN
def conv_alg(X_train_seq, Y_train_seq):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(5,5),activation='relu',input_shape=(3, 4)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
    historyConv = model.fit(X_train_seq, Y_train_seq, batch_size=100, epochs=100, verbose=1)
    return historyConv, model

#------------------------------------ Making the predictions
def predictPrice(X_test_seq, Y_test_seq, regressor):
    predicted_stock_price = regressor.predict(X_test_seq)
    #predicted_stock_price = model.predict(X_test_seq)

    # show the inputs and predicted outputs
    for i in range(len(X_test_seq)):
    	print("X=%s, Predicted=%s" % (X_test_seq[i], predicted_stock_price[i]))
    
    predicted_stock_price = predicted_stock_price.reshape(-1,1)
    Y_test_seq = Y_test_seq.reshape(-1,1)
    return predicted_stock_price, Y_test_seq

'''
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
Y_test_seq=scaler.inverse_transform(Y_test_seq) 
'''
#-------------RMSE
def calculate_rmse(predicted_stock_price, Y_test_seq):
    rmse = np.sqrt(((predicted_stock_price - Y_test_seq) ** 2).mean(axis=0))
    return rmse

def rmse_average(rmse): 
    return sum(rmse) / len(rmse) 

#-----------------------------------CROSS-VALIDATION SPLIT------------------------
#Here I use TimeSeriesSplit
    
#
def cv_split(df):
    X = df.loc[:,:'Volume']
    Y = df[['Close']]
    errors = []
    #number splits = 10
    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        tr_i = train_index.item(len(train_index)-1)
        te_i1 = test_index.item(0)
        te_i2 = test_index.item(len(test_index)-1)
       
        X_train, X_test = X.loc[:tr_i, :'Volume'], X.loc[te_i1:te_i2, :'Volume']
        y_train, y_test_s = Y.loc[:tr_i, 'Close'], Y.loc[te_i1:te_i2, 'Close']

        y_train = pd.Series.to_frame(y_train)
        y_test = pd.Series.to_frame(y_test_s)

        X_train_norm, X_test_norm, y_train_norm, y_test_norm = normalize_df(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        X_train_seq, Y_train_seq, X_test_seq, Y_test_seq = df_to_3d(X_train_norm = X_train_norm, y_train_norm = y_train_norm, X_test_norm = X_test_norm, y_test_norm = y_test_norm)
        
        history, model = lstm_alg(X_train_seq, Y_train_seq)

        
        #history, regressor = lstm_alg(X_train_seq, Y_train_seq)
        predicted_stock_price, Y_test_seq = predictPrice(X_test_seq, Y_test_seq, model)
        rmse = calculate_rmse(predicted_stock_price, Y_test_seq)
        errors.append(rmse)
    return errors

####
    df = import_df('amzn')
    df = fillMissingValues(df)
    df = addChange(df)
    df = convertToDate(df)
    df = swap_cols(df)
    X_train, y_train, X_test, y_test = splitData(df)
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = normalize_df(X_train, X_test, y_train, y_test)
    X_train_seq, Y_train_seq, X_test_seq, Y_test_seq = df_to_3d(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
    history, regressor = lstm_alg(X_train_seq, Y_train_seq)
    #historyConv, model = conv_alg(X_train_seq, Y_train_seq)
    predicted_stock_price, Y_test_seq = predictPrice(X_test_seq, Y_test_seq, regressor)

####


if __name__ == '__main__':
    #print("What company's close price would you like to predict? \nPlease write the respective number or abbreviation")
    #answer_1 = input("1.aaba (Altaba Inc.)\n2.aapl (Apple Inc.)\n3.amzn (Amazon)\n4.axp (American Express Company)\n5.ba (Boeing Co.)\n6.cat (Caterpillar)\n7.csco (Cisco Systems, Inc.)\n8.cvx (Chevron Corporation)\n9.dis (Walt Disney Co)\n10.ge (General Electric Company)\n11.googl (Google)\n12.gs (Goldman Sachs Group Inc)\n13.hd (Home Depot Inc)\n14.ibm (IBM)\n15.intc (Intel Corporation)\n16.jnj (Johnson & Johnson)\n17.jpm (JPMorgan Chase & Co.)\n18.ko (The Coca-Cola Co.)\n19.mcd (Mcdonald's Corp)\n20.mmm (3M Co)\n21.mrk (Merck & Co., Inc.)\n22.msft (Microsoft Corporation)\n23.nke (Nike Inc)\n24.pfe (Pfizer Inc.)\n25.pg (Procter & Gamble Co)\n26.trv (Travelers Companies Inc)\n27.unh (UnitedHealth Group Inc)\n28.utx (United Technologies Corporation)\n29.vz (Verizon Communications Inc.)\n30.wmt (Walmart Inc)\n31.xom (Exxon Mobil Corporation)\nWRITE HERE:")
    #answer_1='amzn'
    
    allComp = ['aaba', 'aapl', 'amzn', 'axp', 'ba', 'cat', 'csco', 'cvx', 'dis', 'ge', 'googl', 'gs', 'hd', 'ibm', 'intc', 'jnj', 'jpm', 'ko', 'mcd', 'mmm', 'mrk', 'msft', 'nke', 'pfe', 'pg', 'trv', 'unh', 'utx', 
          'vz', 'wmt', 'xom']
    rmse_dict = dict()
 
# Load the dictionary back from the pickle file.
rmse_dict2 = pickle.load(open( "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Code\\Pickles\\rmse_dict_extra.pkl", "rb" ))
rmse_dict1 = pickle.load( open( "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Code\\Pickles\\rmse_dict_stdVars.pkl", "rb" ) )

for i in allComp:
    if i not in rmse_dict2:
        df = import_df(i)
        df = fillMissingValues(df)
        df = convertToDate(df)
        df = addChange(df)
        df = swap_cols(df)
        errors = cv_split(df)
        avg_error = rmse_average(errors)
        rmse_dict2[i] = avg_error
        pickle.dump(rmse_dict2, open("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Code\\Pickles\\rmse_dict_extra.pkl", "wb" ) )
        #print("Average RMSE = ", avg_error)

#Pickle RMSE_DICT with {Open, High, Low, Volume}
rmse_dict1 = pickle.load( open( "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Code\\Pickles\\rmse_dict_stdVars.pkl", "rb" ) )

#Pickle RMSE_DICT with {Open, High, Low, DailyChange, Volume}
rmse_dict2 = pickle.load(open( "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Code\\Pickles\\rmse_dict_extra.pkl", "rb" ))

#TO WEKA
df.to_csv("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\3rd Semester\\Dataset\\amzn\\to_WEKA_5vars.csv")
