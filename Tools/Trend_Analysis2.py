#read me : 使用到LSTM/LightGBM/ARIMA/SARIMA/MA模型
#https://discuss.streamlit.io/t/ta-lib-streamlit-deploy-error/7643/4 Talib部署 超難...


import requests
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import MinMaxScaler 
#import keras
#from keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
#from keras.layers import Dropout,BatchNormalization
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
from typing import Tuple
#import lightgbm as lgb
#from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as smt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
#import matplotlib.pyplot as plt
import warnings
import sys, os
from dateutil.relativedelta import relativedelta
from numpy.linalg import LinAlgError
import investpy
import calendar

def EMA(Open, timeperiod = 30, startIdx = 0):
    k = 2 / (timeperiod + 1)
    lookback_ema = timeperiod - 1
    if startIdx < lookback_ema:
        startIdx = lookback_ema
    endIdx = len(Open) - 1
    if lookback_ema >= len(Open):
        exit('too short')
    output_ema = np.zeros(len(Open))
    output_ema[startIdx] = np.mean(Open[startIdx - lookback_ema:startIdx + 1])
    t = startIdx + 1
    while(t <= endIdx):
        output_ema[t] =  k * Open[t] + (1 - k) * output_ema[t - 1]
        t += 1
    output_ema[:startIdx] = np.nan
    return output_ema
#################################################Main Functions######################################

class cal_Tool:

    def compute_day_month(self,m):
        '''
        判断指定的月份是大月、小月还是平月
        :param m: 指定的月份
        :return: 返回一个字典：包含月份的天数、月份的类型
        '''
        # 返回的数据
        big_month = [1,3,5,7,8,10,12]
        small_month = [4,6,9,11]
        flat_month = [2]
        out_data = {'code': 0, 'msg': 'success'}
        if m in big_month:
            # 大月
            out_data['days'] = 31
            out_data['type'] = 'Big Month'
        elif m in small_month:
            # 小月
            out_data['days'] = 30
            out_data['type'] = 'Small Month'
        elif m in flat_month:
            # 平月
            out_data['type'] = 'Flat Month'
            # 平月天数：闰年29天，平年28天
            year = datetime.datetime.today().year
            print('>>>>>年份信息是：%s<<<<<' % (year, ))
            if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
                # 闰年
                out_data['days'] = 29
            else:
                out_data['days'] = 28
        else:
            # 其他数据都是错误的
            out_data['code'] = -1
            out_data['msg'] = '数据异常，月份信息有误'
        return out_data


    def daterange(self,date1, date2):
        for n in range(int ((date2 - date1).days)+1):
            yield date1 + datetime.timedelta(n)
    
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def month_weekdays(self,yyyy,month):
        weekdays = 0
        for i in range(1,32):
            try:
                day = datetime.date(yyyy,month,i)
                if day.weekday() in [0,1,2,3,4]:
                    weekdays += 1
            except:
                pass
        return weekdays
    
    def weightedmovingaverage(self,Data, period):
        weighted = []
        for i in range(len(Data)):
                try:
                    total = np.arange(1, period + 1, 1) # weight matrix
                    matrix = Data[i - period + 1: i + 1]
                    matrix = np.ndarray.flatten(matrix)
                    matrix = total * matrix # multiplication
                    wma = (matrix.sum()) / (total.sum()) # WMA
                    weighted = np.append(weighted, wma) # add to array
                except ValueError:
                    pass
        return weighted
    
    def RSI(self,df, period = 6, ema = True):
        """
        Returns a pd.Series with the relative strength index.
        """
        close_delta = df.diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema == True:
            # Use exponential moving average
            ma_up = up.ewm(com = period - 1, adjust=True, min_periods = period).mean()
            ma_down = down.ewm(com = period - 1, adjust=True, min_periods = period).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window = period, adjust=False).mean()
            ma_down = down.rolling(window = period, adjust=False).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        return rsi

    def KAMA(self,price, n=10, pow1=2, pow2=30):
        ''' kama indicator '''    
        ''' accepts pandas dataframe of prices '''

        absDiffx = abs(price - price.shift(1) )  

        ER_num = abs( price - price.shift(n) )
        ER_den = absDiffx.rolling(n).sum() #pandas.stats.moments.rolling_sum(absDiffx,n)
        ER = ER_num / ER_den
        sc = ( ER*(2.0/(pow1+1)-2.0/(pow2+1.0))+2/(pow2+1.0) ) ** 2.0
        test = pd.DataFrame({'ER_num':ER_num,'ER_den':ER_den,'ER':ER,'SC':sc})
        for i in range(0,len(test)):
           if test['ER_num'][i] == 0.0 and test['ER_den'][i] == 0.0:
              test['ER'][i] = 0
              test['SC'][i] = 0

        answer = np.zeros(test['SC'].size)
        N = len(answer)
        first_value = True

        for i in range(N):
            if test['SC'][i] != test['SC'][i]:
                answer[i] = np.nan
            else:
                if first_value:
                    answer[i] = price[i]
                    first_value = False
                else:
                    answer[i] = answer[i-1] + test['SC'][i] * (price[i] - answer[i-1])
        return answer

    def BBANDS(self,data, window):
        sma = data.rolling(window = window).mean()
        std = data.rolling(window = window).std()
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        return upper_bb, sma, lower_bb

    def MOM(self,Open, timeperiod=10):
        res = np.array([np.nan]*len(Open)).astype('float')
        res[timeperiod:] = Open[timeperiod:] - Open[:-timeperiod]
        return res

    def EMA_(self,Open, timeperiod = 30, startIdx = 0):
        k = 2 / (timeperiod + 1)
        lookback_ema = timeperiod - 1
        if startIdx < lookback_ema:
            startIdx = lookback_ema
        endIdx = len(Open) - 1
        if lookback_ema >= len(Open):
            exit('too short')
        output_ema = np.zeros(len(Open))
        output_ema[startIdx] = np.mean(Open[startIdx - lookback_ema:startIdx + 1])
        t = startIdx + 1
        while(t <= endIdx):
            output_ema[t] =  k * Open[t] + (1 - k) * output_ema[t - 1]
            t += 1
        output_ema[:startIdx] = np.nan
        return output_ema

    def MACD(self,Open, fastperiod=12, slowperiod=26, signalperiod=9):
        lookback_slow = slowperiod - 1
        lookback_sign = signalperiod - 1
        lookback_total = lookback_sign + lookback_slow
        startIdx = lookback_total
        t = startIdx - lookback_sign
        shortma = EMA(Open, fastperiod, startIdx = t)
        longma = EMA(Open, slowperiod, startIdx = t)
        macd = shortma - longma
        macdsignal = np.zeros(len(Open))
        macdsignal[t:] = EMA(macd[t:], signalperiod)
        macdsignal[:t] = np.nan
        macd[:startIdx] = np.nan
        macdhist = macd - macdsignal
        return macd, macdsignal, macdhist

    def adf_test(self,timeseries):
        #Perform Dickey-Fuller test:
        print("Results of Dickey-Fuller Test\n================================================")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4], index = [
            "Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        for key, value in dftest[4].items():
            dfoutput["Criterical Value (%s)"%key] = value
        print(dfoutput)
        print("================================================")  
        #寫個自動判斷式
        if dfoutput[0] < dfoutput[4]:
            diff_signal = False
            print("The data is stationary. (Criterical Value 1%)")  
        elif dfoutput[0] < dfoutput[5]:
            diff_signal = False
            print("The data is stationary. (Criterical Value 5%)") 
        elif dfoutput[0] < dfoutput[6]:
            diff_signal = False
            print("The data is stationary. (Criterical Value 10%)")
        else:
            diff_signal = True
            print("The data is non-stationary, so do differencing!")
        return diff_signal

    def arima_rmse(self,data, p, d, q ,period):
        #period = 30 #預測30天
        L =len(data)
        train = data[:(L-period)]
        test = data[-period:]
        RMSE = []
        name = []
        for i in range(p):
            for j in range(0,d):
                for k in range(q):            
                    model = ARIMA(train, order=(i,j,k))
                    try:
                        fitted = model.fit(disp=-1)
                        fc, se, conf = fitted.forecast(period, alpha=0.05)  
                        rmse = sqrt(mean_squared_error(test,fc))
                        RMSE.append(rmse)
                        name.append(f"ARIMA({i},{j},{k})")
                        print(f"ARIMA({i},{j},{k})：RMSE={rmse}")
                    except:
                        pass
        best = np.argmin(RMSE)
        best_set = name[best]
        best_RMSE = RMSE[best]
        print("==========================================================================")
        print(f"This best model is {best_set}{best_RMSE} based on argmin RMSE.")
        rmse_fig = plt.figure(figsize=(30,8))
        plt.bar(name, RMSE)
        plt.bar(best_set, best_RMSE, color = "red")
        plt.xticks(rotation=30)
        plt.title("RMSE")
        #plt.savefig("Arima RMSE")
        #plt.show()
        plt.close()
        p = int(best_set[6])
        q = int(best_set[10])
        return p,q

class Data:
    def __init__(self, stock_number,yyyy,mm):
        self.stock_number = stock_number
        self.yyyy = yyyy
        self.mm = mm

    def get_stock_data(self,if_lstm): #輸入想要預測的月份
        if self.stock_number in ('Taiwan Paper','Taiwan Steel','Taiwan Plastic'):
            self.stock_data = investpy.indices.get_index_historical_data(index = self.stock_number, 
                                               country = 'Taiwan', 
                                               from_date = '01/01/2010', 
                                               to_date = datetime.datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                               )
            self.stock_data = self.stock_data.iloc[:,:4] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月
            if if_lstm == True:
                if self.mm == 1:
                   self.stock_data_real = self.stock_data[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)) & (self.stock_data.index>= '{}-{}-01'.format(self.yyyy-1,12))] #實際驗證前一個月的MAPE                
                else:
                   self.stock_data_real = self.stock_data[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)) & (self.stock_data.index>= '{}-{}-01'.format(self.yyyy,self.mm-1))] #實際驗證前一個月的MAPE
            else:
                pass
            self.analysis_data = investpy.indices.get_index_historical_data(index = self.stock_number, 
                                               country = 'Taiwan', 
                                               from_date = '01/01/2010', 
                                               to_date = datetime.datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                               )
            self.analysis_data = self.analysis_data.iloc[:,:4] 
        elif self.stock_number == 'Aluminum':
            self.stock_data = investpy.get_commodity_historical_data(commodity= self.stock_number,
                                                                country = "united kingdom", 
                                                                from_date='01/01/2010', 
                                                                to_date=datetime.datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                                               )
            self.stock_data = self.stock_data.iloc[:,:4] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月
            if if_lstm == True:
                if self.mm == 1:
                   self.stock_data_real = self.stock_data[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)) & (self.stock_data.index>= '{}-{}-01'.format(self.yyyy-1,12))] #實際驗證前一個月的MAPE                
                else:
                   self.stock_data_real = self.stock_data[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)) & (self.stock_data.index>= '{}-{}-01'.format(self.yyyy,self.mm-1))] #實際驗證前一個月的MAPE
            else:
                pass
            self.analysis_data = investpy.get_commodity_historical_data(commodity=self.stock_number,
                                                                country = "united kingdom", 
                                                                from_date='01/01/2010', 
                                                                to_date=datetime.datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                                               )
            self.analysis_data = self.analysis_data.iloc[:,:4] 
        else:
            self.stock_data = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross = self.stock_number, 
                                                             from_date = '01/01/2010', 
                                                             to_date = datetime.datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                                            )
            self.stock_data = self.stock_data.iloc[:,:4] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月
            if if_lstm == True:
                if self.mm == 1:
                   self.stock_data_real = self.stock_data[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)) & (self.stock_data.index>= '{}-{}-01'.format(self.yyyy-1,12))] #實際驗證前一個月的MAPE                
                else:
                   self.stock_data_real = self.stock_data[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)) & (self.stock_data.index>= '{}-{}-01'.format(self.yyyy,self.mm-1))] #實際驗證前一個月的MAPE
            else:
                pass
            self.analysis_data = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross = self.stock_number, 
                                                             from_date = '01/01/2010', 
                                                             to_date = datetime.datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                                            )
            self.analysis_data = self.analysis_data.iloc[:,:4] 
            
            
    
    def get_analysis_index(self,df,post_open):
        calculate = cal_Tool()
        df['pre_open'] = df['Open'].shift(22) #上個月收盤 扣掉假日
        if post_open == True:
           df['post_open'] = df['Open'].shift(-22)   # 未來一個月收盤價 扣掉假日
           #df['target'] = df['post_open']-df['Open'] 等同於偷看到答案
        else:
           pass
        df['close-open'] = (df['Close']-df['Open'])/df['Open']
        df['high-low'] = (df['High']-df['Low'])/df['Low']  #震幅
        df['price_change'] = df['Open']-df['pre_open'] #今日漲跌  
        df['p_change'] = (df['Open']-df['pre_open'])/df['pre_open']*100  #今日漲跌百分比
        
        df['MA5'] = df['Open'].rolling(5).mean()  #5日均線
        df['MA10'] = df['Open'].rolling(10).mean()
        df['MA20'] = df['Open'].rolling(20).mean()
        
        df['RSI6'] = calculate.RSI(df['Open'], period=6, ema = True)
        df['RSI12'] = calculate.RSI(df['Open'], period=12, ema = True)
        df['RSI24'] = calculate.RSI(df['Open'], period=24, ema = True)
        df["KAMA"] = calculate.KAMA(df['Open'], n=30 , pow1=2,pow2=30)
        df['upper'], df['middle'], df['lower'] = calculate.BBANDS(df['Open'], window=20)
        
        df['MOM'] = calculate.MOM(df['Open'].values, timeperiod=5) #月增長率
        df['EMA12'] = calculate.EMA_(df['Open'].values, timeperiod=12,startIdx=0) #指數移動平均線
        df['EMA26'] = calculate.EMA_(df['Open'].values, timeperiod=26,startIdx=0)
        
        df['DIFF'], df['DEA'], df['MACD'] = calculate.MACD(df['Open'].values, fastperiod=12, slowperiod=26, signalperiod=9) #平滑異同移動平均線
        df['MACD']  = df['MACD'] *2
        df.dropna(inplace=True)

        return df
    
    def data_split(self,if_lstm,period):
        #self.y_real = self.stock_data[(self.stock_data.index >= '{}-{}-01'.format(self.yyyy,self.mm-1))&(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm))]['Open'].values 
        if if_lstm == True:
            if self.mm == 2:            
                test = self.stock_data[(self.stock_data.index >= '{}-{}-01'.format(self.yyyy-1,12))&(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1))] #預計預測一個月的資料
                train = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,12)]  
            elif self.mm == 1:
                test = self.stock_data[(self.stock_data.index >= '{}-{}-01'.format(self.yyyy-1,11))&(self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,12))] #預計預測一個月的資料
                train = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,11)]                  
            else :
                test = self.stock_data[(self.stock_data.index >= '{}-{}-01'.format(self.yyyy,self.mm-2))&(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1))] #預計預測一個月的資料
                train = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-2)] 
            train_set = train['Open']
            test_set = test['Open']
            self.y_test = test['Open'].values
            self.sc = MinMaxScaler(feature_range=(0,1))
            train_set = train_set.values.reshape(-1,1) #行數自動計算，將array變成1列的格式
            training_set_scaled = self.sc.fit_transform(train_set)
            self.X_train = [] 
            self.y_train = []
            for i in range(period,len(train_set)):
                self.X_train.append(training_set_scaled[i-period:i, 0]) 
                self.y_train.append(training_set_scaled[i, 0]) 
            self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train) 
            self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
            self.y_train_before = train['Open'].values[period:]
            #產出X_test
            inputs = self.stock_data['Open'][len(self.stock_data) - len(test) -10:].values.reshape(-1,1) #再次抓出要預測的資料
            inputs = self.sc.transform(inputs)
            self.X_test = []
            for i in range(period, len(inputs)):
                self.X_test.append(inputs[i-period:i, 0]) #處理成預測第i天就需要第i天前的資料
            self.X_test = np.array(self.X_test)
            self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        else:
            target = 'post_open'
            mon_count = cal_Tool() 
            if self.mm == 2:  
                X = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns!=target]
                y = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns==target]      
                days = mon_count.month_weekdays(self.yyyy,self.mm-1)
                split = days
                #split = int(len(self.stock_data.loc[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm))&(self.stock_data.index >= '{}-{}-01'.format(self.yyyy,self.mm-1)),:]))
            elif self.mm == 1:
                X = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,12),self.stock_data.columns!=target]
                y = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,12),self.stock_data.columns==target]                
                days = mon_count.month_weekdays(self.yyyy-1,12)
                split = days
                #split = int(len(self.stock_data.loc[(self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm ))&(self.stock_data.index >= '{}-{}-01'.format(self.yyyy-1,12)),:]))                
            else :
                X = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns!=target]
                y = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns==target]                
                days = mon_count.month_weekdays(self.yyyy,self.mm-1)
                split = days           
            self.X_train, self.X_test = X[:-split], X[-split:]
            self.y_train, self.y_test = y[:-split], y[-split:]
            #calculate = cal_Tool()
            #days = calculate.month_weekdays(self.yyyy,self.mm)
            #if days > len(y):
            #   self.X_real = pd.concat([X.tail(days-len(y)),self.stock_data_real])
            #else:
            self.X_real = self.stock_data_real
            #TEST
            self.sc = MinMaxScaler(feature_range=(0,1))
            self.X_train = self.sc.fit_transform(self.X_train)
            self.X_train = pd.DataFrame(self.X_train,columns = X.columns)
            self.X_test = self.sc.fit_transform(self.X_test)
            self.X_test = pd.DataFrame(self.X_test,columns = X.columns)
            self.X_real = self.sc.fit_transform(self.X_real)
            self.y_train = self.sc.fit_transform(self.y_train)
            self.y_test = self.sc.fit_transform(self.y_test)
    
    def data_split_for_arima(self,yyyy):
        self.stock_data['year'] = self.stock_data.index.year
        three_year_ago = yyyy - 2
        self.stock_data = self.stock_data.loc[(self.stock_data['year']>=three_year_ago),:].drop('year',axis=1)

# class visual:
    
#     def Open_Close_Trend(self,df):
#         fig=plt.figure(figsize=(20,8))
#         plt.xticks(rotation = 90) #旋轉標籤文字90度 
#         ax1 = fig.add_subplot(111) #新增子圖
#         ax1.plot(df.Close,color='red',label='close')
#         ax1.plot(df.Open,color='green',label='open')
#         plt.legend() #加入指標說明看板
#         plt.close()
#         return fig
    
#     def Plot_Stock_Prediction(self,real,fcst,product):
#         fig = plt.figure(figsize=(16,8))
#         plt.plot(real, color = 'black', label = 'Real {} Stock Price'.format(product))
#         plt.plot(fcst, color = 'green', label = 'Predicted {} Stock Price'.format(product))
#         plt.title('{} Stock Price Prediction'.format(product))
#         plt.xlabel('Time')
#         plt.ylabel('Stock Price')
#         plt.legend()
#         plt.close()
#         return fig
#         #plt.show()
    
#     def Open_Price_Trend(self,df,product):
#         fig = plt.figure(figsize=(16,8))
#         plt.plot(df['Open'], label='{} Future'.format(product)) #請自行更改label名稱
#         plt.ylabel('Price') #請自行更改y軸名稱
#         plt.legend()
#         plt.close()
#         return fig
        
    
#     def ETS_Decomposition(self,df,product):
#         result = seasonal_decompose(df['Open'], model='multiplicative', freq=12)
#         ets_a = plt.figure(figsize=(30,8))
#         plt.subplot(4,1,1)
#         plt.plot(result.observed, label='{} Future'.format(product))
#         plt.ylabel('Price')
#         plt.xticks(df.index[::200], rotation=0) #調整x軸刻度的呈現(每隔12個)
#         plt.margins(0)
#         plt.close()
#         ets_b = plt.figure(figsize=(30,8))
#         plt.subplot(4,1,2)
#         plt.plot(result.trend)
#         plt.ylabel('Trend')
#         plt.xticks(df.index[::200], rotation=0)
#         plt.margins(0)
#         plt.close()
#         ets_c = plt.figure(figsize=(30,8))
#         plt.subplot(4,1,3)
#         plt.plot(result.seasonal)
#         plt.ylabel('Seasonal')
#         plt.xticks(df.index[::200], rotation=0)
#         plt.margins(0)
#         plt.close()
#         ets_d = plt.figure(figsize=(30,8))
#         plt.subplot(4,1,4)
#         plt.scatter(df.index,result.resid);
#         plt.ylabel('Resid')
#         plt.xticks(df.index[::200], rotation=0)
#         plt.margins(0)  #拆成四個subplot以利調整圖片間距、x軸刻度的呈現
#         plt.close()
#         return ets_a,ets_b,ets_c,ets_d
    
#     def ACF_PACF(self,df):
#         f = plt.figure(facecolor='white', figsize=(16,8))
#         ax1 = f.add_subplot(211)
#         plot_acf(df['Open'], lags=24, ax=ax1);
#         ax2 = f.add_subplot(212);
#         plot_pacf(df['Open'], lags=24, ax=ax2);
#         plt.rcParams['axes.unicode_minus'] = False 
#         plt.close()
#         return f
        
# class Model:

#     def __init__(self,product):
#         self.product = product
    
#     def LSTM(self,X_train,y_train,X_test,y_test,y_train_before,y_real,sc,mm,yyyy,period):
#         keras.backend.clear_session()
#         regressor = Sequential()
#         regressor.add(LSTM(units=100,input_shape=(X_train.shape[1],1)))
#         regressor.add(Dense(units=1))
#         regressor.compile(optimizer = 'rmsprop',loss = 'mean_squared_error')
#         history = regressor.fit(X_train, y_train, epochs = 1000, batch_size = 16,verbose=0)
#         #history = regressor.fit(X_train, y_train, epochs = 10, batch_size = 16,verbose=0)
#         #plt.title('train_loss')
#         #plt.ylabel('loss')
#         #plt.xlabel('Epoch')
#         #plt.plot( history.history["loss"])
#         trainPredict = regressor.predict(X_train)
#         calculate = cal_Tool()
#         trainPredict = sc.inverse_transform(trainPredict)
#         trainScore = calculate.mean_absolute_percentage_error(y_train_before,trainPredict[:,0])
#         print('Train Score by LSTM: %.2f MAPE' % (trainScore))
#         train_plot = pd.DataFrame({'Real Stock Price':y_train_before,'Predicted Stock Price': trainPredict[:,0]}) #20220103
#         #visualization = visual()
#         #train_plot = visualization.Plot_Stock_Prediction(y_train_before,trainPredict,self.product)
#         #預測測試集
#         predicted_stock_price = regressor.predict(X_test)
#         predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#         testScore = calculate.mean_absolute_percentage_error(y_test, predicted_stock_price)
#         print('Test Score by LSTM: %.2f MAPE' % (testScore))  
#         #test_plot = visualization.Plot_Stock_Prediction(y_test,predicted_stock_price,self.product)
#         test_plot = pd.DataFrame({'Real Stock Price':y_test,'Predicted Stock Price': predicted_stock_price[:,0]}) #20220103
#         if mm == 1:
#             day = calculate.month_weekdays(yyyy-1,12)
#         else:
#             day = calculate.month_weekdays(yyyy,mm-1)
#         #for_pred_input是為了滾到預測抓出來的最後十筆資料
#         total_inputs = y_test[-period:].tolist()
#         predict_list = []
#         for i in range(0,day):
#             inputs = np.array(total_inputs).reshape(-1,1)
#             inputs = sc.transform(inputs)
#             X_test = []
#             X_test.append(inputs[-period:, 0])
#             X_test = np.array(X_test)
#             X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#             #print(X_test)
#             predicted_stock_price = regressor.predict(X_test) 
#             predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#             #print(predicted_stock_price)
#             total_inputs.append(predicted_stock_price[0][0])
#             predict_list.append(predicted_stock_price[0][0])
#         if len(predict_list) > len(y_real): #因為有國定假日
#             predict_list = predict_list[:len(y_real)]
#         #print(predict_list)
#         RealTestScore = calculate.mean_absolute_percentage_error(y_real, np.array(predict_list))
#         print('Real Test Score by LSTM: %.2f MAPE' % (RealTestScore))        
#         #real_plot = visualization.Plot_Stock_Prediction(y_real,np.array(predict_list),self.product)
#         real_plot = pd.DataFrame({'Real Stock Price':y_real,'Predicted Stock Price': np.array(predict_list)}) #20220103
#         total_inputs = y_real[-period:].tolist()
#         day = calculate.month_weekdays(yyyy,mm)
#         output_list = []
#         for i in range(0,day):
#             inputs = np.array(total_inputs).reshape(-1,1)
#             inputs = sc.transform(inputs)
#             X_test = []
#             X_test.append(inputs[-period:, 0])
#             X_test = np.array(X_test)
#             X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#             predicted_stock_price = regressor.predict(X_test) 
#             predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#             total_inputs.append(predicted_stock_price[0][0])
#             output_list.append(predicted_stock_price[0][0]) 
#         return output_list, RealTestScore,train_plot,test_plot,real_plot
    
#     def WMA(self,df,yyyy,mm,period,y_real,predicted_span):
#        #matype 0 代表 SMA，1 代表 EMA，2 代表 WMA，3 代表 DEMA，4 代表 TEMA
#         if predicted_span == 1:
#             calculate = cal_Tool()
#             if mm == 1:
#                 df = df[df.index < '{}-{}-01'.format(yyyy-1,12)]
#             else:
#                 df = df[df.index < '{}-{}-01'.format(yyyy,mm-1)]
#             df1 = pd.DataFrame(calculate.weightedmovingaverage(df['Open'].values,period = period),index=df.index[period-1:],columns=['WMA'])
#             df['wma_short'] = df1
#             ma_fig = pd.DataFrame(df[['wma_short','Open']][-100:])
#             ma_fig.rename(columns={'wma_short':'Predicted Stock Price','Open':'Real Stock Price'},inplace = True)
#             #ma_fig =  plt.figure(figsize=(16,8))
#             #plt.plot(df[['wma_short','Open']][-100:]) 
#             #plt.title('WMA pred v.s. Actual historical data')
#             #plt.close()
#             na_num =  df['wma_short'].isna().sum()
#             BeforeFCSTScore = calculate.mean_absolute_percentage_error(df['Open'][na_num:].values, df['wma_short'][na_num:].values)
#             print('預測前計算分數 by WMA: %.2f MAPE' % (BeforeFCSTScore))
#             if mm == 1:
#                 day = calculate.month_weekdays(yyyy-1,12)
#             else:
#                 day = calculate.month_weekdays(yyyy,mm-1)
#             total_inputs = df['Open'][-period:].values.tolist()
#             predict_list = []
#             for i in range(0,day):
#                 inputs = np.array(total_inputs).reshape(-1,1)
#                 X_test = []
#                 X_test.append(inputs[-period:,0])
#                 X_test = np.array(X_test)
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#                 predicted_stock_price = calculate.weightedmovingaverage(X_test, period = period)
#                 total_inputs.append(predicted_stock_price[0])
#                 predict_list.append(predicted_stock_price[0])
#             if len(predict_list) > len(y_real): #因為有國定假日
#                 predict_list = predict_list[:len(y_real)]
#             RealScore = calculate.mean_absolute_percentage_error(y_real.values, predict_list)
#             print('預測分數 by WMA: %.2f MAPE' % (RealScore))
#             #visualization = visual()
#             #plot = visualization.Plot_Stock_Prediction(y_real.values,np.array(predict_list),self.product)
#             plot = pd.DataFrame({'Real Stock Price':y_real.values,'Predicted Stock Price': np.array(predict_list)},index = y_real.index) #20220103
#             #預測未來一個月
#             day = calculate.month_weekdays(yyyy,mm)
#             total_inputs = y_real[-period:].values.tolist()
#             total_predict_list = []
#             for i in range(0,day):
#                 inputs = np.array(total_inputs).reshape(-1,1)
#                 X_test = []
#                 X_test.append(inputs[-period:,0])
#                 X_test = np.array(X_test)
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#                 predicted_stock_price = calculate.weightedmovingaverage(X_test, period = period)
#                 total_inputs.append(predicted_stock_price[0])
#                 total_predict_list.append(predicted_stock_price[0])
#         elif predicted_span == 3:
#             calculate = cal_Tool()
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).month
#             df = df[df.index < '{}-{}-01'.format(new_yyyy,new_mm)] #這邊需要以三個月作為測試集
#             df1 = pd.DataFrame(calculate.weightedmovingaverage(df['Open'].values,period = period),index=df.index[period-1:],columns=['WMA'])
#             df['wma_short'] = df1
#             ma_fig = pd.DataFrame(df[['wma_short','Open']][-100:])
#             ma_fig.rename(columns={'wma_short':'Predicted Stock Price','Open':'Real Stock Price'},inplace = True)
#             #ma_fig =  plt.figure(figsize=(16,8))
#             #plt.plot(df[['wma_short','Open']][-100:]) 
#             #plt.title('WMA pred v.s. Actual historical data')
#             #plt.close()
#             na_num =  df['wma_short'].isna().sum()
#             BeforeFCSTScore = calculate.mean_absolute_percentage_error(df['Open'][na_num:].values, df['wma_short'][na_num:].values)
#             print('預測前計算分數 by WMA: %.2f MAPE' % (BeforeFCSTScore))
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).month
#             end_dt = datetime.date(yyyy,mm,1)
#             start_dt = datetime.date(new_yyyy,new_mm,1)

#             days = list()
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        days.append(dt.strftime("%Y-%m-%d"))
#             days = len(days)
#             total_inputs = df['Open'][-period:].values.tolist()
#             predict_list = []
#             for i in range(0,days):
#                 inputs = np.array(total_inputs).reshape(-1,1)
#                 X_test = []
#                 X_test.append(inputs[-period:,0])
#                 X_test = np.array(X_test)
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#                 predicted_stock_price = calculate.weightedmovingaverage(X_test, period = period)
#                 total_inputs.append(predicted_stock_price[0])
#                 predict_list.append(predicted_stock_price[0])
#             if len(predict_list) > len(y_real): #因為有國定假日
#                 predict_list = predict_list[:len(y_real)]
#             RealScore = calculate.mean_absolute_percentage_error(y_real.values, predict_list)
#             print('預測分數 by WMA: %.2f MAPE' % (RealScore))
#             #visualization = visual()
#             #plot = visualization.Plot_Stock_Prediction(y_real.values,np.array(predict_list),self.product)
#             plot = pd.DataFrame({'Real Stock Price':y_real.values,'Predicted Stock Price': np.array(predict_list)},index = y_real.index) #20220103
#             #預測未來三個月
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).month
#             start_dt = datetime.date(yyyy,mm,1)
#             end_dt = datetime.date(new_yyyy,new_mm,1)
#             fcst_days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        fcst_days += 1
#             total_inputs = y_real[-period:].values.tolist()
#             total_predict_list = []
#             for i in range(0,fcst_days):
#                 inputs = np.array(total_inputs).reshape(-1,1)
#                 X_test = []
#                 X_test.append(inputs[-period:,0])
#                 X_test = np.array(X_test)
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#                 predicted_stock_price = calculate.weightedmovingaverage(X_test, period = period)
#                 total_inputs.append(predicted_stock_price[0])
#                 total_predict_list.append(predicted_stock_price[0])
#         elif predicted_span == 6:
#             calculate = cal_Tool()
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).month
#             #取6個月前的資料當訓練集
#             df = df[df.index < '{}-{}-01'.format(new_yyyy,new_mm)]
#             df1 = pd.DataFrame(calculate.weightedmovingaverage(df['Open'].values,period = period),index=df.index[period-1:],columns=['WMA'])
#             df['wma_short'] = df1
#             ma_fig = pd.DataFrame(df[['wma_short','Open']][-100:])
#             ma_fig.rename(columns={'wma_short':'Predicted Stock Price','Open':'Real Stock Price'},inplace = True)
#             #ma_fig =  plt.figure(figsize=(16,8))
#             #plt.plot(df[['wma_short','Open']][-100:]) 
#             #plt.title('WMA pred v.s. Actual historical data')
#             #plt.close()
#             na_num =  df['wma_short'].isna().sum()
#             BeforeFCSTScore = calculate.mean_absolute_percentage_error(df['Open'][na_num:].values, df['wma_short'][na_num:].values)
#             print('預測前計算分數 by WMA: %.2f MAPE' % (BeforeFCSTScore))
#             #這邊要把6個月累加
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).month
#             end_dt = datetime.date(yyyy,mm,1)
#             start_dt = datetime.date(new_yyyy,new_mm,1)
#             days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        days += 1
#             total_inputs = df['Open'][-period:].values.tolist()
#             predict_list = []
#             for i in range(0,days):
#                 inputs = np.array(total_inputs).reshape(-1,1)
#                 X_test = []
#                 X_test.append(inputs[-period:,0])
#                 X_test = np.array(X_test)
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#                 predicted_stock_price = calculate.weightedmovingaverage(X_test, period = period)
#                 total_inputs.append(predicted_stock_price[0])
#                 predict_list.append(predicted_stock_price[0])
#             if len(predict_list) > len(y_real): #因為有國定假日
#                 predict_list = predict_list[:len(y_real)]
#             RealScore = calculate.mean_absolute_percentage_error(y_real.values, predict_list)
#             print('預測分數 by WMA: %.2f MAPE' % (RealScore))
#             #visualization = visual()
#             #plot = visualization.Plot_Stock_Prediction(y_real.values,np.array(predict_list),self.product)
#             plot = pd.DataFrame({'Real Stock Price':y_real.values,'Predicted Stock Price': np.array(predict_list)},index = y_real.index) #20220103
#             #預測未來6個月
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).month
#             start_dt = datetime.date(yyyy,mm,1)
#             end_dt = datetime.date(new_yyyy,new_mm,1)
#             fcst_days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        fcst_days += 1
 
#             total_inputs = y_real[-period:].values.tolist()
#             total_predict_list = []
#             for i in range(0,fcst_days):
#                 inputs = np.array(total_inputs).reshape(-1,1)
#                 X_test = []
#                 X_test.append(inputs[-period:,0])
#                 X_test = np.array(X_test)
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
#                 predicted_stock_price = calculate.weightedmovingaverage(X_test, period = period)
#                 total_inputs.append(predicted_stock_price[0])
#                 total_predict_list.append(predicted_stock_price[0])
#         return total_predict_list, RealScore,ma_fig,plot
        
#     def Light_gbm(self,X_train,y_train,X_test,y_test,X_real,sc):
#         lgb_train = lgb.Dataset(X_train, label= y_train)
#         lgb_eval = lgb.Dataset(X_test, label=y_test)
#         #調整max_depth & num_leaves
#         estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                               objective = 'regression',
#                               seed = 100,
#                               n_jobs = -1,
#                               verbose = -1,
#                               metric = 'mape',
#                               max_depth = 6,
#                               num_leaves = 40,
#                               learning_rate = 0.1,
#                               feature_fraction = 0.7,
#                               bagging_fraction = 1,
#                               bagging_freq = 2,
#                               reg_alpha = 0.001,
#                               reg_lambda = 8,
#                               cat_smooth = 0,
#                               num_iterations = 200
#                              )
#         params = {
#                     'max_depth': [4,6,8],
#                     'num_leaves': [20,30,40],
#                  }
#         with HiddenPrints():
#             gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
#             gbm.fit(X_train,y_train)
#         best_max_depth = list(gbm.best_params_.values())[0]
#         best_num_leaves = list(gbm.best_params_.values())[1]
#         #調整feature_fraction
#         estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                               objective = 'regression',
#                               seed = 100,
#                               n_jobs = -1,
#                               verbose = -1,
#                               metric = 'mape',
#                               max_depth = best_max_depth,
#                               num_leaves = best_num_leaves,
#                               learning_rate = 0.1,
#                               feature_fraction = 0.7,
#                               bagging_fraction = 1,
#                               bagging_freq = 2,
#                               reg_alpha = 0.001,
#                               reg_lambda = 8,
#                               cat_smooth = 0,
#                               num_iterations = 200
#                              )
#         params = {
#                     'feature_fraction': [0.6, 0.8, 1],
#                  }
#         with HiddenPrints():
#             gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
#             gbm.fit(X_train,y_train)
#         best_feature_fraction = list(gbm.best_params_.values())[0]
#         #調整bagging_fraction & bagging_freq
#         estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                               objective = 'regression',
#                               seed = 100,
#                               n_jobs = -1,
#                               verbose =  -1,
#                               metric = 'mape',
#                               max_depth = best_max_depth,
#                               num_leaves = best_num_leaves,
#                               learning_rate = 0.1,
#                               feature_fraction = best_feature_fraction,
#                               bagging_fraction = 1,
#                               bagging_freq = 2,
#                               reg_alpha = 0.001,
#                               reg_lambda = 8,
#                               cat_smooth = 0,
#                               num_iterations = 200
#                              )
#         params = {
#                      'bagging_fraction': [0.8,0.9,1],
#                      'bagging_freq': [2,3,4],
#                  }
#         with HiddenPrints():
#             gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
#             gbm.fit(X_train,y_train)
#         best_bagging_fraction = list(gbm.best_params_.values())[0]
#         best_bagging_freq = list(gbm.best_params_.values())[1]  
#         #調整lambda_l1(reg_alpha)和lambda_l2(reg_lambda)
#         estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                               objective = 'regression',
#                               seed = 100,
#                               n_jobs = -1,
#                               verbose =  -1,
#                               metric = 'mape',
#                               max_depth = best_max_depth,
#                               num_leaves = best_num_leaves,
#                               learning_rate = 0.1,
#                               feature_fraction = best_feature_fraction,
#                               bagging_fraction = best_bagging_fraction,
#                               bagging_freq = best_bagging_freq,
#                               reg_alpha = 0.001,
#                               reg_lambda = 8,
#                               cat_smooth = 0,
#                               num_iterations = 200
#                              )
#         params = {
#                      'reg_alpha': [0.001,0.005,0.01,0.02],
#                      'reg_lambda': [2,4,6,8,10]
#                 }
#         with HiddenPrints():
#             gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
#             gbm.fit(X_train,y_train)
#         best_reg_alpha = list(gbm.best_params_.values())[0]
#         best_reg_lambda = list(gbm.best_params_.values())[1]  
#         #調整cat_smooth
#         estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                               objective = 'regression',
#                               seed = 100,
#                               n_jobs = -1,
#                               verbose =  -1,
#                               metric = 'mape',
#                               max_depth = best_max_depth,
#                               num_leaves = best_num_leaves,
#                               learning_rate = 0.1,
#                               feature_fraction = best_feature_fraction,
#                               bagging_fraction = best_bagging_fraction,
#                               bagging_freq = best_bagging_freq,
#                               reg_alpha = best_reg_alpha,
#                               reg_lambda = best_reg_lambda,
#                               cat_smooth = 0,
#                               num_iterations = 200
#                              )        
#         params = {
#                      'cat_smooth': [0,10,20]
#                 }
#         with HiddenPrints():
#             gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
#             gbm.fit(X_train,y_train)
#         best_cat_smooth = list(gbm.best_params_.values())[0]
#         #調整learning rate & num_iterations
#         estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
#                               objective = 'regression',
#                               seed = 100,
#                               n_jobs = -1,
#                               verbose =  -1,
#                               metric = 'mape',
#                               max_depth = best_max_depth,
#                               num_leaves = best_num_leaves,
#                               learning_rate = 0.1,
#                               feature_fraction = best_feature_fraction,
#                               bagging_fraction = best_bagging_fraction,
#                               bagging_freq = best_bagging_freq,
#                               reg_alpha = best_reg_alpha,
#                               reg_lambda = best_reg_lambda,
#                               cat_smooth = best_cat_smooth,
#                               num_iterations = 200
#                              )            
#         params = {
#                      'learning_rate': [0.001,0.005,0.01,0.025,0.05],
#                      'num_iterations': [100,200,500,800]
#                 }
#         with HiddenPrints():
#             gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
#             gbm.fit(X_train,y_train)
#         best_learning_rate = list(gbm.best_params_.values())[0]
#         best_num_iterations = list(gbm.best_params_.values())[1]  
#         #Finish Fine-tuning
#         params = {
#               'boosting_type': 'gbdt',
#               'objective' : 'regression',
#               'seed' : 100,
#               'n_jobs' : -1,
#               'verbose' :  -1,
#               'metric' : 'mape',
#               'max_depth' : best_max_depth,
#               'num_leaves' : best_num_leaves,
#               'learning_rate' : best_learning_rate,
#               'feature_fraction' : best_feature_fraction,
#               'bagging_fraction' : best_bagging_fraction,
#               'bagging_freq' : best_bagging_freq,
#               'reg_alpha' : best_reg_alpha,
#               'reg_lambda' : best_reg_lambda,
#               'cat_smooth' : best_cat_smooth,
#               'num_iterations' : best_num_iterations,
#               }
#         gbm = lgb.train(params, lgb_train, num_boost_round=500)
#         y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration).reshape(-1,1)
#         ####Test
#         y_pred_prob = sc.inverse_transform(y_pred_prob)
#         y_train = sc.inverse_transform(X_test)
#         y_test = sc.inverse_transform(y_test)
#         calculate = cal_Tool()
#         TestScore = calculate.mean_absolute_percentage_error(y_test, y_pred_prob)
#         print('Test Score by GBM: %.2f MAPE' % (TestScore))
#         #visualization = visual()
#         #test_plot = visualization.Plot_Stock_Prediction(y_test,y_pred_prob,self.product)
#         test_plot = pd.DataFrame({'Real Stock Price':y_test[:,0],'Predicted Stock Price': y_pred_prob[:,0]}) #20220103
#         #漲幅觀察&準確度
#         pred_change = y_pred_prob[:-1] - y_pred_prob[1:]
#         real_change = y_test[:-1] - y_test[1:]
#         changing_plot = pd.DataFrame({'Real Stock Price Changing':real_change[:,0],'Predicted Stock Price Changing': pred_change[:,0]}) #20220103
#         #changing_plot = plt.figure(figsize=(16,8))
#         #plt.plot(real_change, color = 'black', label = 'Real {} Stock Price Changing'.format(self.product))
#         #plt.plot(pred_change, color = 'green', label = 'Predicted {} Stock Price Changing'.format(self.product))
#         #plt.title('{} Stock Price Changing'.format(self.product))
#         #plt.xlabel('Time')
#         #plt.ylabel('Stock Price')
#         #plt.legend()
#         #plt.close()
        
#         pred_change_trends = []
#         real_change_trends = []
#         for i in range(0,len(real_change)):
#             if pred_change[i] < 0:
#                pred_change_trend = -1
#                pred_change_trends.append(pred_change_trend)
#             else :
#                pred_change_trend = 1
#                pred_change_trends.append(pred_change_trend)       
#             if real_change[i] < 0:
#                real_change_trend = -1
#                real_change_trends.append(real_change_trend)
#             else :
#                real_change_trend = 1
#                real_change_trends.append(real_change_trend)         
#         acc = (np.array(real_change_trends) - np.array(pred_change_trends)).tolist().count(0)/len(real_change_trends)
#         print('漲幅預測準確度 by GBM為:{}'.format(acc))
#         #輸出目標月預測
#         y_real_prob = gbm.predict(X_real, num_iteration=gbm.best_iteration, predict_disable_shape_check=True).reshape(-1, 1)
#         y_real_prob = sc.inverse_transform(y_real_prob)
#         #plt.figure(figsize=(14,7))
#         #plt.rcParams["figure.figsize"] = (14, 7)
#         #importance_plot = lgb.plot_importance(gbm,max_num_features = 50)
#         #plt.close()
#         importance_plot = pd.DataFrame({'importance':gbm.feature_importance()},index = X_train.columns)
#         return y_real_prob, acc, TestScore, test_plot,changing_plot,importance_plot
    
#     def ARIMA(self,df,yyyy,mm,predicted_span):
#         visualization = visual()
#         trend_plot = visualization.Open_Price_Trend(df,self.product)
#         ets_plot_a,ets_plot_b,ets_plot_c,ets_plot_d = visualization.ETS_Decomposition(df,self.product)
#         calculate = cal_Tool()
#         with HiddenPrints():
#             test_signal = calculate.adf_test(df['Open'])
#             if test_signal == True:
#                 print('數據在0階差分未平穩')
#                 diff_1 = df['Open']- df['Open'].shift(1) 
#                 diff_1 = diff_1.dropna()
#                 diff_1.head()
#                 #diff_1.plot(figsize=(6,4), label="diff_1")
#                 #plt.legend()
#                 test_signal = calculate.adf_test(diff_1)
#                 best_d = 1
#                 if test_signal == True:
#                     best_d = 2
#             else:
#                 best_d = 0
        
#         acf_plot = visualization.ACF_PACF(df)
#         with HiddenPrints():
#             best_p,best_q = calculate.arima_rmse(df['Open'], p=5, d=best_d, q=5 ,period=30)
#         if predicted_span == 1:
#             if mm == 1:
#                 target_days = calculate.month_weekdays(yyyy,mm)
#                 days = len(df.loc[(df.index >= '{}-{}-01'.format(yyyy-1,12))&(df.index < '{}-{}-01'.format(yyyy,mm))])
#             else:
#                 target_days = calculate.month_weekdays(yyyy,mm)
#                 days = len(df.loc[(df.index >= '{}-{}-01'.format(yyyy,mm-1))&(df.index < '{}-{}-01'.format(yyyy,mm))])
#         elif predicted_span == 3:
#             #for 預測筆數
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).month
#             start_dt = datetime.date(yyyy,mm,1)
#             end_dt = datetime.date(new_yyyy,new_mm,1)
#             target_days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        target_days += 1
#             #for 訓練集        
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).month
#             end_dt = datetime.date(yyyy,mm,1)
#             start_dt = datetime.date(new_yyyy,new_mm,1)
#             days = len(df.loc[(df.index > '{}-{}-01'.format(new_yyyy,new_mm))&(df.index < '{}-{}-01'.format(yyyy,mm))])  
#         elif predicted_span == 6:
#             #for 預測筆數
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).month
#             start_dt = datetime.date(yyyy,mm,1)
#             end_dt = datetime.date(new_yyyy,new_mm,1)
#             target_days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        target_days += 1
#             #for 訓練集        
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).month
#             end_dt = datetime.date(yyyy,mm,1)
#             start_dt = datetime.date(new_yyyy,new_mm,1)
#             days = len(df.loc[(df.index > '{}-{}-01'.format(new_yyyy,new_mm))&(df.index < '{}-{}-01'.format(yyyy,mm))])       
        
#         title = f'ARIMA({best_p},{best_d},{best_q}) for Forecasting {days} days'
#         L = len(df['Open'])
#         x_train = df['Open'][:(L-days)]
#         x_test = df['Open'][-days:]
#         try:
#            model = ARIMA(x_train, order=(best_p, best_d, best_q)) 
#            fitted = model.fit(disp=-1,transparams=False)
#         except (ValueError, LinAlgError): 
#            pass
#         fc, se, conf = fitted.forecast(days, alpha=0.05) # 95% conf
#         fc_series = pd.Series(fc, index=x_test.index)
#         lower_series = pd.Series(conf[:, 0], index=x_test.index)
#         upper_series = pd.Series(conf[:, 1], index=x_test.index)
        
#         TestScore = calculate.mean_absolute_percentage_error(x_test, fc_series)
#         print('Test Score by ARIMA: %.2f MAPE' % (TestScore))
#         #visualization = visual()
#         #test_plot = visualization.Plot_Stock_Prediction(x_test,fc_series,self.product)
#         test_plot = pd.DataFrame({'Real Stock Price':x_test,'Predicted Stock Price': fc_series})
#         #實際預測
#         model = ARIMA(df['Open'], order=(best_p, best_d, best_q)) 
#         fitted = model.fit(disp=-1)
#         fc, se, conf = fitted.forecast(target_days, alpha=0.05) # 95% conf
#         fc_series = pd.Series(fc)
#         return fc_series, TestScore,trend_plot,ets_plot_a,ets_plot_b,ets_plot_c,ets_plot_d,acf_plot,test_plot
    
#     def SARIMA(self,df,yyyy,mm,predicted_span):
#         calculate = cal_Tool()
#         if predicted_span == 1:
#             if mm == 1:
#                 target_days = calculate.month_weekdays(yyyy,mm)
#                 days = len(df.loc[(df.index >= '{}-{}-01'.format(yyyy-1,12))&(df.index < '{}-{}-01'.format(yyyy,mm))])
#             else:
#                 target_days = calculate.month_weekdays(yyyy,mm)
#                 days = len(df.loc[(df.index >= '{}-{}-01'.format(yyyy,mm-1))&(df.index < '{}-{}-01'.format(yyyy,mm))])
#         elif predicted_span == 3:
#             #for 預測筆數
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).month
#             start_dt = datetime.date(yyyy,mm,1)
#             end_dt = datetime.date(new_yyyy,new_mm,1)
#             target_days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        target_days += 1
#             #for 訓練集        
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-3)).month
#             end_dt = datetime.date(yyyy,mm,1)
#             start_dt = datetime.date(new_yyyy,new_mm,1)
#             days = len(df.loc[(df.index > '{}-{}-01'.format(new_yyyy,new_mm))&(df.index < '{}-{}-01'.format(yyyy,mm))])  
#         elif predicted_span == 6:
#             #for 預測筆數
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).month
#             start_dt = datetime.date(yyyy,mm,1)
#             end_dt = datetime.date(new_yyyy,new_mm,1)
#             target_days = 0
#             weekend = [5,6]
#             for dt in calculate.daterange(start_dt, end_dt):
#                 if dt.weekday() not in weekend:
#                     if dt != end_dt:
#                        target_days += 1
#             #for 訓練集        
#             new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).year
#             new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).month
#             end_dt = datetime.date(yyyy,mm,1)
#             start_dt = datetime.date(new_yyyy,new_mm,1)
#             days = len(df.loc[(df.index > '{}-{}-01'.format(new_yyyy,new_mm))&(df.index < '{}-{}-01'.format(yyyy,mm))])  
#         with HiddenPrints():
#             smodel = pm.auto_arima(df['Open'][:-days],
#                            start_p=1, 
#                            start_q=1,
#                            test='adf', #如果stationary為假且d為None，用來檢測平穩性的單位根檢驗的類型。默認為‘kpss’;可設置為adf
#                            max_p=3, 
#                            max_q=3, 
#                            m=12, #frequency of series
#                            start_P=0, #The starting value of P, the order of the auto-regressive portion of the seasonal model. 
#                            seasonal=True, #加入季節性因素進去，為SARIMA的S
#                            d=None, #The order of first-differencing. If None (by default), the value will automatically be selected based on the result
#                            D=1,#The order of the seasonal differencing. If None (by default, the value will automatically be selected based on the results
#                            trace=True, #是否打印適合的狀態。如果值為False，則不會打印任何調試信息。值為真會打印一些
#                            error_action='ignore', #If unable to fit an ARIMA for whatever reason, this controls the error-handling behavior. 
#                            suppress_warnings=True, #statsmodel中可能會拋出許多警告。如果suppress_warnings為真，那麽來自ARIMA的所有警告都將被壓制
#                            stepwise=True
#                           )
#             smodel.summary()
#         title = f'Best SARIMA for Forecasting {days} days'
#         L = len(df)
#         x_train = df['Open'][:(L-days)]
#         x_test = df['Open'][-days:]
#         #Forecast
#         fc, conf = smodel.predict(n_periods=days,alpha=0.05, return_conf_int=True)
#         index_of_fc = np.arange(len(df), len(df)+days)
#         #Make as pandas series
#         fc_series = pd.Series(fc, index=x_test.index)
#         lower_series = pd.Series(conf[:, 0], index=x_test.index)
#         upper_series = pd.Series(conf[:, 1], index=x_test.index)
#         calculate = cal_Tool()
#         TestScore = calculate.mean_absolute_percentage_error(x_test, fc_series)
#         print('Test Score by SARIMA: %.2f MAPE' % (TestScore))
#         #visualization = visual()
#         #test_plot = visualization.Plot_Stock_Prediction(x_test,fc_series,self.product)
#         test_plot = pd.DataFrame({'Real Stock Price':x_test,'Predicted Stock Price': fc_series})
#         #實際預測
#         fc, conf = smodel.predict(target_days, alpha=0.05,return_conf_int=True) # 95% conf
#         fc_series = pd.Series(fc)
#         return fc_series, TestScore,test_plot

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    stock_numbers = ['Taiwan Paper','Taiwan Steel','Taiwan Plastic','Aluminum','USD/TWD']
    predicted_intervals = [1,3,6]
    yyyy = 2022
    mm = 1
    for stock_number in stock_numbers:
        if stock_number == 'USD/TWD':
           stock_number = "USD:TWD" 
        if not os.path.exists('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}'.format(stock_number,yyyy,mm)):
            os.mkdir('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}'.format(stock_number,yyyy,mm))
            os.mkdir('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}'.format(stock_number,yyyy,mm))
            os.mkdir('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}'.format(stock_number,yyyy,mm))
        product = stock_number
        for predicted_interval in predicted_intervals:
            if stock_number == 'USD:TWD':
                stock_number = "USD/TWD" 
            stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)           
            if predicted_interval == 1:
                print('{} {} 開始訓練'.format(product, predicted_interval))
                #LSTM
                stock_engine.get_stock_data(if_lstm=True)
                stock_engine.data_split(if_lstm=True,period=10)
                model = Model(product=product)
                list_lstm, score_lstm, lstm_train_plot,lstm_test_plot,lstm_real_plot = model.LSTM(  X_train = stock_engine.X_train,
                                                                                                    y_train = stock_engine.y_train,
                                                                                                    X_test = stock_engine.X_test,
                                                                                                    y_test = stock_engine.y_test,
                                                                                                    y_train_before = stock_engine.y_train_before,
                                                                                                    y_real = stock_engine.stock_data_real['Open'].values,
                                                                                                    sc = stock_engine.sc,                                                                   
                                                                                                    mm = stock_engine.mm,
                                                                                                    yyyy = stock_engine.yyyy,
                                                                                                    period = 10
                                                                                                )
                new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-2)).year
                new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-2)).month
                lstm_train_plot.index = stock_engine.stock_data[:len(lstm_train_plot)].index
                lstm_test_plot.index = stock_engine.stock_data.loc[(stock_engine.stock_data.index.year == new_yyyy)&(stock_engine.stock_data.index.month == new_mm)].index
                lstm_real_plot.index = stock_engine.stock_data_real.index
                #WMA
                list_wma, score_wma, ma_train,ma_test = model.WMA(df = stock_engine.stock_data,
                                                                    yyyy = yyyy,
                                                                    mm = mm,
                                                                    period = 10,
                                                                    y_real = stock_engine.stock_data_real['Open'],
                                                                    predicted_span = 1)
                #LightGBM
                mon_count = cal_Tool() 
                if mm == 1:
                    days = mon_count.month_weekdays(yyyy-1,12) #mm-1
                else:
                    days = mon_count.month_weekdays(yyyy,mm-1) #mm-1
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                fcst_days = mon_count.month_weekdays(yyyy,mm)
                stock_engine.stock_data_real = stock_engine.get_analysis_index(df = stock_engine.stock_data,post_open=False)[-fcst_days:]
                stock_engine.get_analysis_index(df = stock_engine.stock_data,post_open=True)
                stock_engine.data_split(if_lstm=False,period=10)
                model = Model(product=product)
                list_gbm,score_gbm_acc, score_gbm,gbm_test_plot,gbm_change_plot,gbm_importance_plot = model.Light_gbm(X_train = stock_engine.X_train,
                                                                                                                    y_train = stock_engine.y_train,
                                                                                                                    X_test = stock_engine.X_test,
                                                                                                                    y_test = stock_engine.y_test,
                                                                                                                    X_real = stock_engine.X_real,
                                                                                                                    sc = stock_engine.sc)                                                                                       
                weekdays1 = list()
                for i in range(1,32):
                    try:
                        if mm == 1:
                            day = datetime.date(yyyy-1,12,i)
                            if day.weekday() in [0,1,2,3,4]:
                                weekdays1.append(str(day))        
                        else:
                            day = datetime.date(yyyy,mm-1,i)
                            if day.weekday() in [0,1,2,3,4]:
                                weekdays1.append(str(day))
                    except:
                        pass
                gbm_test_plot.index = weekdays1
                gbm_change_plot.index = weekdays1[1:]

                #ARIMA
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                stock_engine.data_split_for_arima(yyyy=yyyy)    
                list_arima, score_arima,arima_trend,ets_a,ets_b,ets_c,ets_d,acf,arima_test = model.ARIMA(stock_engine.stock_data,yyyy=yyyy,mm=mm,predicted_span=1)

                #SARIMA
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                stock_engine.data_split_for_arima(yyyy=yyyy)    
                list_sarima, score_sarima,sarima_test = model.SARIMA(stock_engine.stock_data,yyyy=yyyy,mm=mm,predicted_span=1)

                com =  [('lstm',round(score_lstm,2),0),
                    ('LightGBM',round(score_gbm,2),round(score_gbm_acc,2)),
                    ('WMA',round(score_wma,2),0),
                    ('ARIMA',round(score_arima,2),0),
                    ('SARIMA',round(score_sarima,2),0)]
                compare = pd.DataFrame(com,columns=["model","score","acc_for_gbm"])
                compare.sort_values('score',inplace=True)
                compare.reset_index(drop = True,inplace=True)
                weekdays = list()
                for i in range(1,32):
                    try:
                        day = datetime.date(yyyy,mm,i)
                        if day.weekday() in [0,1,2,3,4]:
                            weekdays.append(str(day))
                    except:
                        pass
                if stock_number == 'USD/TWD':
                    stock_number = "USD:TWD"
                    product = "USD:TWD"
                compare.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
                if compare.iloc[0,0]== 'LightGBM' and score_gbm_acc <= 0.6:
                    print("""最佳預測模型：{}""".format(compare.iloc[1,0]))
                    if compare.iloc[1,0] == 'lstm':     
                        lstm_df = pd.DataFrame(list_lstm,columns = ['FCST'],index = weekdays) 
                        lstm_df['漲跌'] = lstm_df['FCST'] -  lstm_df['FCST'].shift(1)
                        lstm_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,product))
                        lstm_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        lstm_test_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_lstm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        lstm_real_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_lstm_real_plot.xlsx'.format(stock_number,yyyy,mm,product))

                    elif compare.iloc[1,0] == 'WMA':
                        wma_df = pd.DataFrame(list_wma,columns = ['FCST'],index = weekdays) 
                        wma_df['漲跌'] = wma_df['FCST'] -  wma_df['FCST'].shift(1)
                        wma_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
                        ma_train.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        ma_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))

                    elif compare.iloc[1,0] == 'ARIMA':
                        arima_df = pd.DataFrame(list_arima.values,columns = ['FCST'],index = weekdays) 
                        arima_df['漲跌'] = arima_df['FCST'] -  arima_df['FCST'].shift(1)
                        arima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                        arima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        ets_a.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_a.png'.format(stock_number,yyyy,mm,product))
                        ets_b.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_b.png'.format(stock_number,yyyy,mm,product))
                        ets_c.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_c.png'.format(stock_number,yyyy,mm,product))
                        ets_d.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_d.png'.format(stock_number,yyyy,mm,product))
                        acf.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_acf.png'.format(stock_number,yyyy,mm,product))
 
                    elif compare.iloc[1,0] == 'SARIMA':
                        sarima_df = pd.DataFrame(list_sarima.values,columns = ['FCST'],index = weekdays) 
                        sarima_df['漲跌'] = sarima_df['FCST'] -  sarima_df['FCST'].shift(1)
                        sarima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                        sarima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                else:
                    print("""最佳預測模型：{}""".format(compare.iloc[0,0]))
                    if compare.iloc[0,0] == 'lstm':
                        lstm_df = pd.DataFrame(list_lstm,columns = ['FCST'],index = weekdays) 
                        lstm_df['漲跌'] = lstm_df['FCST'] -  lstm_df['FCST'].shift(1)
                        lstm_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,product))
                        lstm_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        lstm_test_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_lstm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        lstm_real_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_lstm_real_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'WMA':
                        wma_df = pd.DataFrame(list_wma,columns = ['FCST'],index = weekdays) 
                        wma_df['漲跌'] = wma_df['FCST'] -  wma_df['FCST'].shift(1)
                        wma_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
                        ma_train.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        ma_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'ARIMA':
                        arima_df = pd.DataFrame(list_arima.values,columns = ['FCST'],index = weekdays) 
                        arima_df['漲跌'] = arima_df['FCST'] -  arima_df['FCST'].shift(1)
                        arima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                        arima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        ets_a.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_a.png'.format(stock_number,yyyy,mm,product))
                        ets_b.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_b.png'.format(stock_number,yyyy,mm,product))
                        ets_c.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_c.png'.format(stock_number,yyyy,mm,product))
                        ets_d.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_ets_d.png'.format(stock_number,yyyy,mm,product))
                        acf.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_arima_acf.png'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'SARIMA':
                        sarima_df = pd.DataFrame(list_sarima.values,columns = ['FCST'],index = weekdays) 
                        sarima_df['漲跌'] = sarima_df['FCST'] -  sarima_df['FCST'].shift(1)
                        sarima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                        sarima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'LightGBM':
                        gbm_df = pd.DataFrame(list_gbm,columns = ['FCST'],index = weekdays)
                        gbm_df['漲跌'] = gbm_df['FCST'] -  gbm_df['FCST'].shift(1) 
                        gbm_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_Best_by_light.xlsx'.format(stock_number,yyyy,mm,product))
                        gbm_test_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_gbm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        gbm_change_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_gbm_change_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        #st.pyplot(gbm_importance_plot.figure)
                        gbm_importance_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}{}/{}_for_gbm_importance_plot.xlsx'.format(stock_number,yyyy,mm,product))

            elif predicted_interval == 3:
                print('{} {} 開始訓練'.format(product, predicted_interval))
                stock_engine.get_stock_data(if_lstm=False)
                model = Model(product=product)
                # for 3 months
                if mm == 3:
                    stock_engine.stock_data_real = stock_engine.stock_data.loc[stock_engine.stock_data.index > '{}-{}-01'.format(yyyy-1,12)]
                elif mm == 2:
                    stock_engine.stock_data_real = stock_engine.stock_data.loc[stock_engine.stock_data.index > '{}-{}-01'.format(yyyy-1,11)]
                elif mm == 1:
                    stock_engine.stock_data_real = stock_engine.stock_data.loc[stock_engine.stock_data.index > '{}-{}-01'.format(yyyy-1,10)]
                else:
                    stock_engine.stock_data_real = stock_engine.stock_data.loc[stock_engine.stock_data.index > '{}-{}-01'.format(yyyy,mm-2)]
                list_wma, score_wma, ma_train,ma_test = model.WMA(df = stock_engine.stock_data,
                                                                    yyyy = yyyy,
                                                                    mm = mm,
                                                                    period = 20,
                                                                    y_real = stock_engine.stock_data_real['Open'],
                                                                    predicted_span = predicted_interval
                                                                    )
                #ARIMA
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                stock_engine.data_split_for_arima(yyyy=yyyy)    
                list_arima, score_arima,arima_trend,ets_a,ets_b,ets_c,ets_d,acf,arima_test = model.ARIMA(stock_engine.stock_data,yyyy=yyyy,mm=mm,predicted_span=3)

                #SARIMA
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                stock_engine.data_split_for_arima(yyyy=yyyy)    
                list_sarima, score_sarima,sarima_test = model.SARIMA(stock_engine.stock_data,yyyy=yyyy,mm=mm,predicted_span=3)
                
                com =  [('WMA',round(score_wma,2)),
                    ('ARIMA',round(score_arima,2)),
                    ('SARIMA',round(score_sarima,2))]
                compare = pd.DataFrame(com,columns=["model","score"])
                compare.sort_values('score',inplace=True)
                compare.reset_index(drop = True,inplace=True)
                if stock_number == 'USD/TWD':
                    stock_number = "USD:TWD"
                    product = "USD:TWD"
                compare.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
                calculate = cal_Tool()
                new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).year
                new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+3)).month
                new_day = calculate.compute_day_month(new_mm).get('days')
                start_dt = datetime.date(yyyy,mm,1)
                end_dt = datetime.date(new_yyyy,new_mm,1)

                weekdays = list()
                weekend = [5,6]
                for dt in calculate.daterange(start_dt, end_dt):
                    if dt.weekday() not in weekend:
                        if dt != end_dt:
                            weekdays.append(dt.strftime("%Y-%m-%d"))
                print("""最佳預測模型：{}""".format(compare.iloc[0,0]))
                if compare.iloc[0,0] == 'WMA':
                    wma_df = pd.DataFrame(list_wma,columns = ['FCST'],index = weekdays) 
                    wma_df['漲跌'] = wma_df['FCST'] -  wma_df['FCST'].shift(1)
                    wma_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
                    ma_train.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    ma_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'ARIMA':
                    arima_df = pd.DataFrame(list_arima.values,columns = ['FCST'],index = weekdays) 
                    arima_df['漲跌'] = arima_df['FCST'] -  arima_df['FCST'].shift(1)
                    arima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                    arima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    ets_a.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_arima_ets_a.png'.format(stock_number,yyyy,mm,product))
                    ets_b.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_arima_ets_b.png'.format(stock_number,yyyy,mm,product))
                    ets_c.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_arima_ets_c.png'.format(stock_number,yyyy,mm,product))
                    ets_d.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_arima_ets_d.png'.format(stock_number,yyyy,mm,product))
                    acf.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_arima_acf.png'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'SARIMA':
                    sarima_df = pd.DataFrame(list_sarima.values,columns = ['FCST'],index = weekdays) 
                    sarima_df['漲跌'] = sarima_df['FCST'] -  sarima_df['FCST'].shift(1)
                    sarima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                    sarima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            elif predicted_interval == 6:
                print('{} {} 開始訓練'.format(product, predicted_interval))
                stock_engine.get_stock_data(if_lstm=False)
                # for 6 months
                calculate = cal_Tool()
                new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).year
                new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=-6)).month
                stock_engine.stock_data_real = stock_engine.stock_data.loc[stock_engine.stock_data.index > '{}-{}-01'.format(new_yyyy,new_mm)]
                model = Model(product=product)
                list_wma, score_wma, ma_train,ma_test = model.WMA(df = stock_engine.stock_data,
                                                                    yyyy = yyyy,
                                                                    mm = mm,
                                                                    period = 10,
                                                                    y_real = stock_engine.stock_data_real['Open'],
                                                                    predicted_span = 6
                                                                )
                #ARIMA
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                stock_engine.data_split_for_arima(yyyy=yyyy)    
                list_arima, score_arima,arima_trend,ets_a,ets_b,ets_c,ets_d,acf,arima_test = model.ARIMA(stock_engine.stock_data,yyyy=yyyy,mm=mm,predicted_span=6)

                #SARIMA
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data(if_lstm=False)
                stock_engine.data_split_for_arima(yyyy=yyyy)    
                list_sarima, score_sarima,sarima_test = model.SARIMA(stock_engine.stock_data,yyyy=yyyy,mm=mm,predicted_span=6)

                com =  [('WMA',round(score_wma,2)),
                    ('ARIMA',round(score_arima,2)),
                    ('SARIMA',round(score_sarima,2))]
                compare = pd.DataFrame(com,columns=["model","score"])
                compare.sort_values('score',inplace=True)
                compare.reset_index(drop = True,inplace=True)
                if stock_number == 'USD/TWD':
                    stock_number = 'USD:TWD'
                    product = "USD:TWD"
                compare.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
                calculate = cal_Tool()
                new_yyyy = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).year
                new_mm = (datetime.date(yyyy,mm,1) + relativedelta(months=+6)).month
                new_day = calculate.compute_day_month(new_mm).get('days')
                start_dt = datetime.date(yyyy,mm,1)
                end_dt = datetime.date(new_yyyy,new_mm,1)

                weekdays = list()
                weekend = [5,6]
                for dt in calculate.daterange(start_dt, end_dt):
                    if dt.weekday() not in weekend:
                        if dt != end_dt:
                            weekdays.append(dt.strftime("%Y-%m-%d"))
                print("""最佳預測模型：{}""".format(compare.iloc[0,0]))
                if compare.iloc[0,0] == 'WMA':
                    wma_df = pd.DataFrame(list_wma,columns = ['FCST'],index = weekdays) 
                    wma_df['漲跌'] = wma_df['FCST'] -  wma_df['FCST'].shift(1)
                    wma_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
                    ma_train.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    ma_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'ARIMA':
                    arima_df = pd.DataFrame(list_arima.values,columns = ['FCST'],index = weekdays) 
                    arima_df['漲跌'] = arima_df['FCST'] -  arima_df['FCST'].shift(1)
                    arima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                    arima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    ets_a.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_arima_ets_a.png'.format(stock_number,yyyy,mm,product))
                    ets_b.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_arima_ets_b.png'.format(stock_number,yyyy,mm,product))
                    ets_c.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_arima_ets_c.png'.format(stock_number,yyyy,mm,product))
                    ets_d.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_arima_ets_d.png'.format(stock_number,yyyy,mm,product))
                    acf.savefig('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_arima_acf.png'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'SARIMA':
                    sarima_df = pd.DataFrame(list_sarima.values,columns = ['FCST'],index = weekdays) 
                    sarima_df['漲跌'] = sarima_df['FCST'] -  sarima_df['FCST'].shift(1)
                    sarima_df.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                    sarima_test.to_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))      
