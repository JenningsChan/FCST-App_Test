%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import date, datetime, timedelta
import matplotlib
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import time
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import investpy
from workalendar.asia import Taiwan
from workalendar.usa import core
from workalendar.europe import UnitedKingdom
from workalendar.asia import China
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import h5py
from sklearn.preprocessing import MinMaxScaler 
import keras
from keras.models import Sequential
from keras.layers import Dense,GRU,Dropout
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as smt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from typing import Tuple
import sys, os
from dateutil.relativedelta import relativedelta
from numpy.linalg import LinAlgError
import calendar

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Cal_Tool:
    
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
            year = today().year
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

    def get_mape(self,y_true, y_pred): 
        """
        Compute mean absolute percentage error (MAPE)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_mae(self,a, b):
        """
        Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
        Returns a vector of len = len(a) = len(b)
        """
        return np.mean(abs(np.array(a)-np.array(b)))

    def get_rmse(self,a, b):
        """
        Comp RMSE. a and b can be lists.
        Returns a scalar.
        """
        return math.sqrt(np.mean((np.array(a)-np.array(b))**2))
    def daterange(self,date1, date2):
        for n in range(int ((date2 - date1).days)+1):
            yield date1 + timedelta(n)
    
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def month_weekdays(self,yyyy,month):
        weekdays = 0
        for i in range(1,32):
            try:
                day = date(yyyy,month,i)
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

    def MOM(self,Close, timeperiod=10):
        res = np.array([np.nan]*len(Close)).astype('float')
        res[timeperiod:] = Close[timeperiod:] - Close[:-timeperiod]
        return res

    def EMA_(self,Close, timeperiod = 30, startIdx = 0):
        k = 2 / (timeperiod + 1)
        lookback_ema = timeperiod - 1
        if startIdx < lookback_ema:
            startIdx = lookback_ema
        endIdx = len(Close) - 1
        if lookback_ema >= len(Close):
            exit('too short')
        output_ema = np.zeros(len(Close))
        output_ema[startIdx] = np.mean(Close[startIdx - lookback_ema:startIdx + 1])
        t = startIdx + 1
        while(t <= endIdx):
            output_ema[t] =  k * Close[t] + (1 - k) * output_ema[t - 1]
            t += 1
        output_ema[:startIdx] = np.nan
        return output_ema

    def MACD(self,Close, fastperiod=12, slowperiod=26, signalperiod=9):
        lookback_slow = slowperiod - 1
        lookback_sign = signalperiod - 1
        lookback_total = lookback_sign + lookback_slow
        startIdx = lookback_total
        t = startIdx - lookback_sign
        shortma = self.EMA_(Close, fastperiod, startIdx = t)
        longma = self.EMA_(Close, slowperiod, startIdx = t)
        macd = shortma - longma
        macdsignal = np.zeros(len(Close))
        macdsignal[t:] = self.EMA_(macd[t:], signalperiod)
        macdsignal[:t] = np.nan
        macd[:startIdx] = np.nan
        macdhist = macd - macdsignal
        return macd, macdsignal, macdhist

    def adf_test(self,timeseries):
        #Perform Dickey-Fuller test:
        #print("Results of Dickey-Fuller Test\n================================================")
        dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(dftest[0:4], index = [
            "Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
        for key, value in dftest[4].items():
            dfoutput["Criterical Value (%s)"%key] = value
        print(dfoutput)
        #print("================================================")  
        #寫個自動判斷式
        if dfoutput[0] < dfoutput[4]:
            diff_signal = False
            #print("The data is stationary. (Criterical Value 1%)")  
        elif dfoutput[0] < dfoutput[5]:
            diff_signal = False
            #print("The data is stationary. (Criterical Value 5%)") 
        elif dfoutput[0] < dfoutput[6]:
            diff_signal = False
            #print("The data is stationary. (Criterical Value 10%)")
        else:
            diff_signal = True
            #print("The data is non-stationary, so do differencing!")
        return diff_signal

    def arima_mape(self,data, p, d, q ,period):
        #period = 30 #預測30天
        L =len(data)
        train = data[:(L-period)]
        test = data[-period:]
        MAPE = []
        name = []
        for i in range(p):
            for j in range(0,d):
                for k in range(q):            
                    model = ARIMA(train, order=(i,j,k))
                    try:
                        fitted = model.fit(disp=-1)
                        fc, se, conf = fitted.forecast(period, alpha=0.05)  
                        mape = sqrt(self.get_mape(test,fc))
                        MAPE.append(mape)
                        name.append(f"ARIMA({i},{j},{k})")
                        #print(f"ARIMA({i},{j},{k})：MAPE={mape}")
                    except:
                        pass
        best = np.argmin(MAPE)
        best_set = name[best]
        best_MAPE = MAPE[best]
        p = int(best_set[6])
        q = int(best_set[10])
        return p,q

class Data:
    def __init__(self, stock_number,yyyy,mm):
        self.stock_number = stock_number
        self.yyyy = yyyy
        self.mm = mm

    def get_stock_data(self): #輸入想要預測的月份
        if self.stock_number in ('Taiwan Paper','Taiwan Steel','Taiwan Plastic'):
            self.stock_data = investpy.indices.get_index_historical_data(index = self.stock_number, 
                                               country = 'Taiwan', 
                                               from_date = '01/01/2010', 
                                               to_date = datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                               )
            self.stock_data = self.stock_data.iloc[:,:5] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月
            lst = Taiwan().holidays(self.yyyy)
            hol,day_name = [item[0] for item in lst],[item[1] for item in lst]
            self.hols = pd.DataFrame({'ds':hol,'holiday':day_name,'lower_window':0,'upper_window':0})
        elif self.stock_number == 'UK Aluminum':
            self.stock_data = investpy.get_commodity_historical_data(commodity= 'Aluminum',
                                                                country = "united kingdom", 
                                                                from_date='01/01/2010', 
                                                                to_date=datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                                               )
            self.stock_data = self.stock_data.iloc[:,:5] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月
            lst = UnitedKingdom().holidays(self.yyyy)
            hol,day_name = [item[0] for item in lst],[item[1] for item in lst]
            self.hols = pd.DataFrame({'ds':hol,'holiday':day_name,'lower_window':0,'upper_window':0})            
        
        elif self.stock_number == 'CN Aluminum':
            search_results = investpy.search_quotes(text='SAFc1', products=['commodities'], countries=['china'])
            search_result = search_results.pop(0)
            self.stock_data = search_result.retrieve_historical_data(from_date='01/01/2010',to_date=datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y'))
            self.stock_data = self.stock_data.iloc[:,:5] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月
            lst = China().holidays(self.yyyy)
            hol,day_name = [item[0] for item in lst],[item[1] for item in lst]
            self.hols = pd.DataFrame({'ds':hol,'holiday':day_name,'lower_window':0,'upper_window':0})  
            
        else:
            self.stock_data = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross = self.stock_number, 
                                                             from_date = '01/01/2010', 
                                                             to_date = datetime(self.yyyy, self.mm, calendar.monthrange(self.yyyy, self.mm)[1]).strftime('%d/%m/%Y')
                                                            )
            self.stock_data = self.stock_data.iloc[:,:5] #Open/High/Low/Close
            self.stock_data = self.stock_data[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm)] #確保是直接預測目標月    
            lst = core.UnitedStates().holidays(self.yyyy)
            hol,day_name = [item[0] for item in lst],[item[1] for item in lst]
            self.hols = pd.DataFrame({'ds':hol,'holiday':day_name,'lower_window':0,'upper_window':0})               
    
    def get_analysis_index(self,df,post_close):
        calculate = Cal_Tool()
        df['pre_close'] = df['Close'].shift(22) #上個月收盤 扣掉假日
        if post_close == True:
           df['post_close'] = df['Close'].shift(-22)   # 未來一個月收盤價 扣掉假日
        else:
           pass
        df['close-open'] = (df['Open']-df['Close'])/df['Close']
        df['high-low'] = (df['High']-df['Low'])/df['Low']  #震幅
        df['price_change'] = df['Close']-df['pre_close'] #今日漲跌  
        df['p_change'] = (df['Close']-df['pre_close'])/df['pre_close']*100  #今日漲跌百分比
        
        df['MA5'] = df['Close'].rolling(5).mean()  #5日均線
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        
        df['RSI6'] = calculate.RSI(df['Close'], period=6, ema = True)
        df['RSI12'] = calculate.RSI(df['Close'], period=12, ema = True)
        df['RSI24'] = calculate.RSI(df['Close'], period=24, ema = True)
        df["KAMA"] = calculate.KAMA(df['Close'], n=30 , pow1=2,pow2=30)
        df['upper'], df['middle'], df['lower'] = calculate.BBANDS(df['Close'], window=20)
        
        df['MOM'] = calculate.MOM(df['Close'].values, timeperiod=5) #月增長率
        df['EMA12'] = calculate.EMA_(df['Close'].values, timeperiod=12,startIdx=0) #指數移動平均線
        df['EMA26'] = calculate.EMA_(df['Close'].values, timeperiod=26,startIdx=0)
        
        df['DIFF'], df['DEA'], df['MACD'] = calculate.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9) #平滑異同移動平均線
        df['MACD']  = df['MACD'] *2
        df.dropna(inplace=True)

        return df
    
    def data_split(self,period):
        target = 'post_close'
        mon_count = Cal_Tool()
        if self.mm == 2:  
            X = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns!=target]
            y = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns==target]      
            days = mon_count.month_weekdays(self.yyyy,self.mm-1)
            split = days
        elif self.mm == 1:
            X = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,12),self.stock_data.columns!=target]
            y = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy-1,12),self.stock_data.columns==target]                
            days = mon_count.month_weekdays(self.yyyy-1,12)
            split = days               
        else :
            X = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns!=target]
            y = self.stock_data.loc[self.stock_data.index < '{}-{}-01'.format(self.yyyy,self.mm-1),self.stock_data.columns==target]                
            days = mon_count.month_weekdays(self.yyyy,self.mm-1)
            split = days           
        self.X_train, self.X_test = X[:-split], X[-split:]
        self.y_train, self.y_test = y[:-split], y[-split:]
        self.X_real = self.stock_data_real
        self.sc = MinMaxScaler(feature_range=(0,1))
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_train = pd.DataFrame(self.X_train,columns = X.columns)
        self.X_test = self.sc.fit_transform(self.X_test)
        self.X_test = pd.DataFrame(self.X_test,columns = X.columns)
        self.X_real = self.sc.fit_transform(self.X_real)
        self.y_train = self.sc.fit_transform(self.y_train)
        self.y_test = self.sc.fit_transform(self.y_test)
        
    def transform_df(self,df,df_plot,stock_number):
        CNY = investpy.currency_crosses.get_currency_cross_recent_data(currency_cross = 'CNY/TWD').tail(1)['Open'].values[0]
        USD = investpy.currency_crosses.get_currency_cross_recent_data(currency_cross = 'USD/TWD').tail(1)['Open'].values[0]
        if stock_number in ('Taiwan Paper','Taiwan Steel','Taiwan Plastic'):
            df.rename(columns={'FCST':'NTD FCST','漲跌':'NTD Price Change'},inplace=True)
            df_plot.rename(columns={'Actual':'NTD Actual','Pred':'NTD FCST'},inplace=True)
            df['USD FCST'] = df['NTD FCST'] / USD
            df['USD Price Change'] = df['NTD Price Change'] / USD
            df_plot['USD Actual'] = df_plot['NTD Actual'] / USD
            df_plot['USD FCST'] = df_plot['NTD FCST'] / USD
        elif stock_number == 'UK Aluminum': 
            df.rename(columns={'FCST':'USD FCST','漲跌':'USD Price Change'},inplace=True)
            df_plot.rename(columns={'Actual':'USD Actual','Pred':'USD FCST'},inplace=True)
            df['NTD FCST'] = df['USD FCST'] * USD
            df['NTD Price Change'] = df['USD Price Change'] * USD
            df_plot['NTD Actual'] = df_plot['USD Actual'] * USD
            df_plot['NTD FCST'] = df_plot['USD FCST'] * USD
        elif stock_number == 'CN Aluminum':
            df.rename(columns={'FCST':'CNY FCST','漲跌':'CNY Price Change'},inplace=True)
            df_plot.rename(columns={'Actual':'CNY Actual','Pred':'CNY FCST'},inplace=True)
            df['USD FCST'] = df['CNY FCST'] * CNY / USD
            df['USD Price Change'] = df['CNY Price Change'] * CNY / USD
            df_plot['USD Actual'] = df_plot['CNY Actual'] * CNY / USD
            df_plot['USD FCST'] = df_plot['CNY FCST'] * CNY / USD
        else: #外匯
            df.rename(columns={'FCST':'NTD FCST','漲跌':'NTD Price Change'},inplace=True)
            df_plot.rename(columns={'Actual':'NTD Actual','Pred':'NTD FCST'},inplace=True)
        return df,df_plot
    
    def transform_change_df(self,df,stock_number):
        CNY = investpy.currency_crosses.get_currency_cross_recent_data(currency_cross = 'CNY/TWD').tail(1)['Open'].values[0]
        USD = investpy.currency_crosses.get_currency_cross_recent_data(currency_cross = 'USD/TWD').tail(1)['Open'].values[0]
        if stock_number in ('Taiwan Paper','Taiwan Steel','Taiwan Plastic'):
            df.rename(columns={'Real Stock Price Changing':'NTD Real Stock Price Changing','Predicted Stock Price Changing':'NTD Predicted Stock Price Changing'},inplace=True)
            df['USD Real Stock Price Changing'] = df['NTD Real Stock Price Changing'] / USD
            df['USD Predicted Stock Price Changing'] = df['NTD Predicted Stock Price Changing'] / USD
        elif stock_number == 'UK Aluminum': 
            df.rename(columns={'Real Stock Price Changing':'USD Real Stock Price Changing','Predicted Stock Price Changing':'USD Predicted Stock Price Changing'},inplace=True)
            df['NTD Real Stock Price Changing'] = df['USD Real Stock Price Changing'] * USD
            df['NTD Predicted Stock Price Changing'] = df['USD Predicted Stock Price Changing'] * USD
        elif stock_number == 'CN Aluminum':
            df.rename(columns={'Real Stock Price Changing':'CNY Real Stock Price Changing','Predicted Stock Price Changing':'CNY Predicted Stock Price Changing'},inplace=True)
            df['NTD Real Stock Price Changing'] = df['CNY Real Stock Price Changing'] * CNY
            df['NTD Predicted Stock Price Changing'] = df['CNY Predicted Stock Price Changing'] * CNY
        else: #外匯
            df.rename(columns={'Real Stock Price Changing':'NTD Real Stock Price Changing','Predicted Stock Price Changing':'NTD Predicted Stock Price Changing'},inplace=True)
        return df

#為了用多線程，其中get_preds_prophet函數內容在下面
def processInput(i, df, H, hols, changepoint_prior_scale, fourier_order, holidays):
    preds_list = get_preds_prophet(df[i-train_size:i], H, hols, changepoint_prior_scale, fourier_order, holidays)
    
    # Compute error metrics
    rmse = get_rmse(df[i:i+H]['y'], preds_list)
    mape = get_mape(df[i:i+H]['y'], preds_list)
    mae = get_mae(df[i:i+H]['y'], preds_list)
    
    return (rmse, mape, mae)

def get_preds_prophet_parallelized(df, H, hols, changepoint_prior_scale=0.05, fourier_order=None, holidays=None):
    """
    This is a parallelized implementation of get_preds_prophet.
    Given a dataframe consisting of both train+validation, do predictions of forecast horizon H on the validation set, 
    at H/2 intervals.
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        changepoint_prior_scale: to detect changepoints in time series analysis trajectories
        fourier_order          : determines how quickly seasonality can change
        holidays               : dataframe containing holidays you will like to model. 
                                 Must have 'holiday' and 'ds' columns
    Outputs
        mean of rmse, mean of mape, mean of mae, dict of predictions
    """
    inputs = range(train_size, len(df)-H, int(H/2))

    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i, df, H, hols, changepoint_prior_scale, fourier_order, holidays) for i in inputs)
    # results has format [(rmse1, mape1, mae1), (rmse2, mape2, mae2), ...]

    rmse = [errors[0] for errors in results]
    mape = [errors[1] for errors in results]
    mae = [errors[2] for errors in results]
    
    return np.mean(rmse), np.mean(mape), np.mean(mae)

def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_mae(a, b):
    """
    Comp mean absolute error e_t = E[|a_t - b_t|]. a and b can be lists.
    Returns a vector of len = len(a) = len(b)
    """
    return np.mean(abs(np.array(a)-np.array(b)))

def get_rmse(a, b):
    """
    Comp RMSE. a and b can be lists.
    Returns a scalar.
    """
    return math.sqrt(np.mean((np.array(a)-np.array(b))**2))

def get_preds_prophet(df, H, hols, changepoint_prior_scale=0.05, fourier_order=None, holidays=None):
    """
    Use Prophet to forecast for the next H timesteps, starting at df[len(df)]
    Inputs
        df: dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H : forecast horizon
        changepoint_prior_scale : to detect changepoints in time series analysis trajectories
        fourier_order           : determines how quickly seasonality can change
        holidays                : dataframe containing holidays you will like to model. 
                                  Must have 'holiday' and 'ds' columns
    Outputs
        A list of predictions
    """
    # Fit prophet model
    if holidays is not None:
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale, holidays=holidays)
    else:
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    if (fourier_order is not None) and (~np.isnan(fourier_order)): #fourier_order非任何形式NA # add monthly seasonality
        m.add_seasonality(name='monthly', period=21, fourier_order=int(fourier_order))

    m.fit(df)
    # Make future dataframe
    future = m.make_future_dataframe(periods=3*H)
    
    # Eliminate weekend from future dataframe
    future['day'] = future['ds'].dt.weekday
    future = future[future['day']<=4]
    future = future.loc[~future.ds.isin(pd.to_datetime(hols.ds)),:] #把國定假日避開
    
    # Predict
    forecast = m.predict(future) # Note this prediction includes the original dates
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return forecast['yhat'][len(df):len(df)+H]
    
def get_error_metrics(df, H, hols, train_size, val_size, changepoint_prior_scale=0.05, fourier_order=None, holidays=None):
    """
    Given a dataframe consisting of both train+validation, do predictions of forecast horizon H on the validation set, 
    at H/2 intervals.
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        changepoint_prior_scale: to detect changepoints in time series analysis trajectories
        fourier_order          : determines how quickly seasonality can change
        holidays               : dataframe containing holidays you will like to model. 
                                 Must have 'holiday' and 'ds' columns
    Outputs
        mean of rmse, mean of mape, mean of mae, dict of predictions
    """
    assert len(df) == train_size + val_size #發生例外時用來提示的語法
    
    # Predict using Prophet, and compute error metrics also
    rmse = [] # root mean square error
    mape = [] # mean absolute percentage error
    mae = []  # mean absolute error
    preds_dict = {}
    
    rmse_mean, mape_mean, mae_mean = get_preds_prophet_parallelized(df, H, hols, changepoint_prior_scale, fourier_order, holidays)

    return rmse_mean, mape_mean, mae_mean, _ #_代表最後一次執行的結果

def hyperparam_tune_cp(df, H, hols, train_size, val_size, changepoint_prior_scale_list):
    """
    Hyperparameter tuning - changepoint
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        changepoint_prior_scale_list: list of changepoint_prior_scale values to try
    Outputs
        optimum hyperparameters
    """
    rmse_mean_list = []
    mape_mean_list = []
    mae_mean_list = []
    for changepoint_prior_scale in tqdm_notebook(changepoint_prior_scale_list): #tdqm=Python中專門用於進度條美化的模塊
        print("changepoint_prior_scale = " + str(changepoint_prior_scale))
        rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df, H, hols, train_size, val_size, changepoint_prior_scale)
        rmse_mean_list.append(rmse_mean)
        mape_mean_list.append(mape_mean)
        mae_mean_list.append(mae_mean)
    
    # Create results dataframe
    results = pd.DataFrame({'changepoint_prior_scale': changepoint_prior_scale_list,
                            'rmse': rmse_mean_list,
                            'mape(%)': mape_mean_list,
                            'mae': mae_mean_list})
    
    # Return hyperparam corresponding to lowest error metric
    return changepoint_prior_scale_list[np.argmin(mape_mean_list)], results #返回沿軸的最小值的索引

def hyperparam_tune_fo(df, H,hols, train_size, val_size, fourier_order_list):
    """
    Hyperparameter tuning - fourier order
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        fourier_order_list     : list of fourier_order values to try
    Outputs
        optimum hyperparameters
    """
    rmse_mean_list = []
    mape_mean_list = []
    mae_mean_list = []
    for fourier_order in tqdm_notebook(fourier_order_list):
        print("fourier_order = " + str(fourier_order))
        rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df, H,hols, train_size, val_size, 0.05, fourier_order)
        rmse_mean_list.append(rmse_mean)
        mape_mean_list.append(mape_mean)
        mae_mean_list.append(mae_mean)
        
    # Create results dataframe
    results = pd.DataFrame({'fourier_order': fourier_order_list,
                            'rmse': rmse_mean_list,
                            'mape(%)': mape_mean_list,
                            'mae': mae_mean_list})
        
    # Return hyperparam corresponding to lowest error metric
    return fourier_order_list[np.argmin(mape_mean_list)], results

def hyperparam_tune_wd(df, H, hols, train_size, val_size, window_list, holidays):
    """
    Hyperparameter tuning - upper and lower windows for holidays
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        window_list            : list of upper and lower window values to try
        holidays               : dataframe containing holidays you will like to model. 
                                 Must have 'holiday' and 'ds' columns
    Outputs
        optimum hyperparameters
    """
    rmse_mean_list = []
    mape_mean_list = []
    mae_mean_list = []
    for window in tqdm_notebook(window_list):
        print("window = " + str(window))
        
        if window is None:
            rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df=df, 
                                                                  H=H, 
                                                                  hols=hols,
                                                                  train_size=train_size, 
                                                                  val_size=val_size, 
                                                                  holidays=None)
        else:
            # Add lower_window and upper_window which extend the holiday out to 
            # [lower_window, upper_window] days around the date
            holidays['lower_window'] = -window
            holidays['upper_window'] = +window
        
            rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df=df, 
                                                                  H=H, 
                                                                  hols=hols,
                                                                  train_size=train_size, 
                                                                  val_size=val_size, 
                                                                  holidays=holidays)
        rmse_mean_list.append(rmse_mean)
        mape_mean_list.append(mape_mean)
        mae_mean_list.append(mae_mean)
        
    # Create results dataframe
    results = pd.DataFrame({'window': window_list,
                            'rmse': rmse_mean_list,
                            'mape(%)': mape_mean_list,
                            'mae': mae_mean_list})
        
    # Return hyperparam corresponding to lowest error metric
    return window_list[np.argmin(rmse_mean_list)], results

def hyperparam_tune_cp_fo_wd(df, H, hols, train_size, val_size, changepoint_prior_scale_list, 
                             fourier_order_list, window_list, holidays):
    """
    Hyperparameter tuning - changepoint, fourier_order, holidays
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        changepoint_prior_scale_list: list of changepoint_prior_scale values to try
        fourier_order_list          : list of fourier_order values to try
        window_list                 : list of upper and lower window values to try
        holidays                    : dataframe containing holidays you will like to model. 
                                      Must have 'holiday' and 'ds' columns
    Outputs
        optimum hyperparameters
    """
    rmse_mean_list = []
    mape_mean_list = []
    mae_mean_list = []
    cp_list = []
    fo_list = []
    wd_list = []
    for changepoint_prior_scale in tqdm_notebook(changepoint_prior_scale_list):
        for fourier_order in tqdm_notebook(fourier_order_list):
            for window in tqdm_notebook(window_list):
                
                if window is None:
                    rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df, 
                                                                          H, 
                                                                          hols,
                                                                          train_size, 
                                                                          val_size, 
                                                                          changepoint_prior_scale, 
                                                                          fourier_order, 
                                                                          holidays=None)
                else:
                    # Add lower_window and upper_window which extend the holiday out to 
                    # [lower_window, upper_window] days around the date
                    holidays['lower_window'] = -window
                    holidays['upper_window'] = +window
        
                    rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df, 
                                                                          H, 
                                                                          hols,
                                                                          train_size, 
                                                                          val_size, 
                                                                          changepoint_prior_scale, 
                                                                          fourier_order, 
                                                                          holidays)
                rmse_mean_list.append(rmse_mean)
                mape_mean_list.append(mape_mean)
                mae_mean_list.append(mae_mean)
                cp_list.append(changepoint_prior_scale)
                fo_list.append(fourier_order)
                wd_list.append(window)
        
    # Return hyperparam corresponding to lowest error metric
    results = pd.DataFrame({'changepoint_prior_scale': cp_list, 
                            'fourier_order': fo_list,
                            'window': wd_list,
                            'rmse': rmse_mean_list,
                            'mape(%)': mape_mean_list,
                            'mae': mae_mean_list})
    temp = results[results['rmse'] == results['rmse'].min()] #待Check
    changepoint_prior_scale_opt = temp['changepoint_prior_scale'].values[0] #待Check
    fourier_order_opt = temp['fourier_order'].values[0] #待Check
    window_opt = temp['window'].values[0] #待Check
    
    return changepoint_prior_scale_opt, fourier_order_opt, window_opt, results

def hyperparam_tune_cp_fo_ss(df, H,hols, train_size, val_size, changepoint_prior_scale_list, 
                             fourier_order_list):
    """
    Hyperparameter tuning - changepoint, fourier_order, holidays
    Inputs
        df                     : dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H                      : forecast horizon
        train_size             : length of training set
        val_size               : length of validation set. Note len(df) = train_size + val_size
        changepoint_prior_scale_list: list of changepoint_prior_scale values to try
        fourier_order_list          : list of fourier_order values to try
        
    Outputs
        optimum hyperparameters
    """
    rmse_mean_list = []
    mape_mean_list = []
    mae_mean_list = []
    cp_list = []
    fo_list = []
    for changepoint_prior_scale in tqdm_notebook(changepoint_prior_scale_list):
        for fourier_order in tqdm_notebook(fourier_order_list):
                rmse_mean, mape_mean, mae_mean, _ = get_error_metrics(df, 
                                                                      H, 
                                                                      hols,
                                                                      train_size, 
                                                                      val_size, 
                                                                      changepoint_prior_scale, 
                                                                      fourier_order)
                rmse_mean_list.append(rmse_mean)
                mape_mean_list.append(mape_mean)
                mae_mean_list.append(mae_mean)
                cp_list.append(changepoint_prior_scale)
                fo_list.append(fourier_order)
        
    # Return hyperparam corresponding to lowest error metric
    results = pd.DataFrame({'changepoint_prior_scale': cp_list, 
                            'fourier_order': fo_list,
                            'rmse': rmse_mean_list,
                            'mape(%)': mape_mean_list,
                            'mae': mae_mean_list})
    temp = results[results['rmse'] == results['rmse'].min()] 
    changepoint_prior_scale_opt = temp['changepoint_prior_scale'].values[0] 
    fourier_order_opt = temp['fourier_order'].values[0] 
    
    return changepoint_prior_scale_opt, fourier_order_opt, results

def get_preds_prophet_final(df, H,hols, changepoint_prior_scale=0.05, fourier_order=None, holidays=None):
    """
    Use Prophet to forecast for the next H timesteps, starting at df[len(df)]
    Inputs
        df: dataframe with headers 'ds' and 'y' (necessary for Prophet)
        H : forecast horizon
        changepoint_prior_scale : to detect changepoints in time series analysis trajectories
        fourier_order           : determines how quickly seasonality can change
        holidays                : dataframe containing holidays you will like to model. 
                                  Must have 'holiday' and 'ds' columns
    Outputs
        A list of predictions
    """
    # Fit prophet model
    if holidays is not None:
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale, holidays=holidays)
    else:
        m = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    if (fourier_order is not None) and (~np.isnan(fourier_order)): #fourier_order非任何形式NA # add monthly seasonality
        m.add_seasonality(name='monthly', period=21, fourier_order=int(fourier_order))

    m.fit(df)
    
    # Make future dataframe
    future = m.make_future_dataframe(periods=2*H)
    
    # Eliminate weekend from future dataframe
    future['day'] = future['ds'].dt.weekday
    future = future[future['day']<=4]
    future = future.loc[~future.ds.isin(pd.to_datetime(hols.ds)),:] #把國定假日避開
    
    # Predict
    forecast = m.predict(future) # Note this prediction includes the original dates
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    return forecast[len(df):len(df)+H]


class Model:

    def __init__(self,product):
        self.product = product
        
    def Light_gbm(self,X_train,y_train,X_test,y_test,X_real,sc,yyyy,mm,holidays):
        lgb_train = lgb.Dataset(X_train, label= y_train)
        lgb_eval = lgb.Dataset(X_test, label=y_test)
        #調整max_depth & num_leaves
        estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              seed = 100,
                              n_jobs = -1,
                              verbose = -1,
                              metric = 'mape',
                              max_depth = 6,
                              num_leaves = 40,
                              learning_rate = 0.1,
                              feature_fraction = 0.7,
                              bagging_fraction = 1,
                              bagging_freq = 2,
                              reg_alpha = 0.001,
                              reg_lambda = 8,
                              cat_smooth = 0,
                              num_iterations = 200
                             )
        params = {
                    'max_depth': [4,6,8],
                    'num_leaves': [20,30,40],
                 }
        with HiddenPrints():
            gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
            gbm.fit(X_train,y_train)
        best_max_depth = list(gbm.best_params_.values())[0]
        best_num_leaves = list(gbm.best_params_.values())[1]
        #調整feature_fraction
        estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              seed = 100,
                              n_jobs = -1,
                              verbose = -1,
                              metric = 'mape',
                              max_depth = best_max_depth,
                              num_leaves = best_num_leaves,
                              learning_rate = 0.1,
                              feature_fraction = 0.7,
                              bagging_fraction = 1,
                              bagging_freq = 2,
                              reg_alpha = 0.001,
                              reg_lambda = 8,
                              cat_smooth = 0,
                              num_iterations = 200
                             )
        params = {
                    'feature_fraction': [0.6, 0.8, 1],
                 }
        with HiddenPrints():
            gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
            gbm.fit(X_train,y_train)
        best_feature_fraction = list(gbm.best_params_.values())[0]
        #調整bagging_fraction & bagging_freq
        estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              seed = 100,
                              n_jobs = -1,
                              verbose =  -1,
                              metric = 'mape',
                              max_depth = best_max_depth,
                              num_leaves = best_num_leaves,
                              learning_rate = 0.1,
                              feature_fraction = best_feature_fraction,
                              bagging_fraction = 1,
                              bagging_freq = 2,
                              reg_alpha = 0.001,
                              reg_lambda = 8,
                              cat_smooth = 0,
                              num_iterations = 200
                             )
        params = {
                     'bagging_fraction': [0.8,0.9,1],
                     'bagging_freq': [2,3,4],
                 }
        with HiddenPrints():
            gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
            gbm.fit(X_train,y_train)
        best_bagging_fraction = list(gbm.best_params_.values())[0]
        best_bagging_freq = list(gbm.best_params_.values())[1]  
        #調整lambda_l1(reg_alpha)和lambda_l2(reg_lambda)
        estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              seed = 100,
                              n_jobs = -1,
                              verbose =  -1,
                              metric = 'mape',
                              max_depth = best_max_depth,
                              num_leaves = best_num_leaves,
                              learning_rate = 0.1,
                              feature_fraction = best_feature_fraction,
                              bagging_fraction = best_bagging_fraction,
                              bagging_freq = best_bagging_freq,
                              reg_alpha = 0.001,
                              reg_lambda = 8,
                              cat_smooth = 0,
                              num_iterations = 200
                             )
        params = {
                     'reg_alpha': [0.001,0.005,0.01,0.02],
                     'reg_lambda': [2,4,6,8,10]
                }
        with HiddenPrints():
            gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
            gbm.fit(X_train,y_train)
        best_reg_alpha = list(gbm.best_params_.values())[0]
        best_reg_lambda = list(gbm.best_params_.values())[1]  
        #調整cat_smooth
        estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              seed = 100,
                              n_jobs = -1,
                              verbose =  -1,
                              metric = 'mape',
                              max_depth = best_max_depth,
                              num_leaves = best_num_leaves,
                              learning_rate = 0.1,
                              feature_fraction = best_feature_fraction,
                              bagging_fraction = best_bagging_fraction,
                              bagging_freq = best_bagging_freq,
                              reg_alpha = best_reg_alpha,
                              reg_lambda = best_reg_lambda,
                              cat_smooth = 0,
                              num_iterations = 200
                             )        
        params = {
                     'cat_smooth': [0,10,20]
                }
        with HiddenPrints():
            gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
            gbm.fit(X_train,y_train)
        best_cat_smooth = list(gbm.best_params_.values())[0]
        #調整learning rate & num_iterations
        estimator = lgb.LGBMRegressor(boosting_type = 'gbdt',
                              objective = 'regression',
                              seed = 100,
                              n_jobs = -1,
                              verbose =  -1,
                              metric = 'mape',
                              max_depth = best_max_depth,
                              num_leaves = best_num_leaves,
                              learning_rate = 0.1,
                              feature_fraction = best_feature_fraction,
                              bagging_fraction = best_bagging_fraction,
                              bagging_freq = best_bagging_freq,
                              reg_alpha = best_reg_alpha,
                              reg_lambda = best_reg_lambda,
                              cat_smooth = best_cat_smooth,
                              num_iterations = 200
                             )            
        params = {
                     'learning_rate': [0.001,0.005,0.01,0.025,0.05],
                     'num_iterations': [100,200,500,800]
                }
        with HiddenPrints():
            gbm = GridSearchCV(estimator,params,cv=3,verbose=-1)
            gbm.fit(X_train,y_train)
        best_learning_rate = list(gbm.best_params_.values())[0]
        best_num_iterations = list(gbm.best_params_.values())[1]  
        #Finish Fine-tuning
        params = {
              'boosting_type': 'gbdt',
              'objective' : 'regression',
              'seed' : 100,
              'n_jobs' : -1,
              'verbose' :  -1,
              'metric' : 'mape',
              'max_depth' : best_max_depth,
              'num_leaves' : best_num_leaves,
              'learning_rate' : best_learning_rate,
              'feature_fraction' : best_feature_fraction,
              'bagging_fraction' : best_bagging_fraction,
              'bagging_freq' : best_bagging_freq,
              'reg_alpha' : best_reg_alpha,
              'reg_lambda' : best_reg_lambda,
              'cat_smooth' : best_cat_smooth,
              'num_iterations' : best_num_iterations,
              }
        gbm = lgb.train(params, lgb_train, num_boost_round=500)
        y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration).reshape(-1,1)
        ####Test
        y_pred_prob = sc.inverse_transform(y_pred_prob)
        y_train = sc.inverse_transform(X_test)
        y_test = sc.inverse_transform(y_test)
        calculate = Cal_Tool()
        TestScore = calculate.mean_absolute_percentage_error(y_test, y_pred_prob)
        print('Test Score by GBM: %.2f MAPE' % (TestScore))
        test_plot = pd.DataFrame({'Real Stock Price':y_test[:,0],'Predicted Stock Price': y_pred_prob[:,0]}) #20220103
        #漲幅觀察&準確度
        pred_change = y_pred_prob[:-1] - y_pred_prob[1:]
        real_change = y_test[:-1] - y_test[1:]
        changing_plot = pd.DataFrame({'Real Stock Price Changing':real_change[:,0],'Predicted Stock Price Changing': pred_change[:,0]}) #20220103
        
        pred_change_trends = []
        real_change_trends = []
        for i in range(0,len(real_change)):
            if pred_change[i] < 0:
               pred_change_trend = -1
               pred_change_trends.append(pred_change_trend)
            else :
               pred_change_trend = 1
               pred_change_trends.append(pred_change_trend)       
            if real_change[i] < 0:
               real_change_trend = -1
               real_change_trends.append(real_change_trend)
            else :
               real_change_trend = 1
               real_change_trends.append(real_change_trend)         
        acc = (np.array(real_change_trends) - np.array(pred_change_trends)).tolist().count(0)/len(real_change_trends)
        print('漲幅預測準確度 by GBM為:{}'.format(acc))
        #輸出目標月預測
        y_real_prob = gbm.predict(X_real, num_iteration=gbm.best_iteration, predict_disable_shape_check=True).reshape(-1, 1)
        y_real_prob = sc.inverse_transform(y_real_prob)
        weekdays1 = list()
        for i in range(1,32):
            try:
                if mm == 1:
                    day = date(yyyy-1,12,i)
                    if day.weekday() in [0,1,2,3,4]:
                        weekdays1.append(str(day))        
                else:
                    day = date(yyyy,mm-1,i)
                    if day.weekday() in [0,1,2,3,4]:
                        weekdays1.append(str(day))
            except:
                pass
        importance_plot = pd.DataFrame({'importance':gbm.feature_importance()},index = X_train.columns)
        test_plot.index = weekdays1
        changing_plot.index = weekdays1[1:]
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        dates = list(filter(lambda date: date.weekday() <= 4 , dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        #print(y_real_prob)
        #print('---------')
        #print(dates)
        prediction = pd.DataFrame(y_real_prob[:len(dates)],columns = ['FCST'],index = dates)
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)
        #return y_real_prob, acc, TestScore, test_plot,changing_plot,importance_plot,dates
        return prediction, acc, TestScore, test_plot,changing_plot,importance_plot
    
    def lstm_model(self,data,yyyy,mm,holidays):
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        last = df.date[len(df)-1].date()  #- pd.DateOffset(months=1)
        y_test = df.loc[(df.date.dt.year == last.year)&(df.date.dt.month == last.month),:]
        train_size = len(df) -1 - len(y_test) - 22 #-1不然會吃到test set的資料
        val_size = 22        
        train_val_size = train_size + val_size # Size of train+validation set
        H=22
        val_list = range(252*6, train_val_size, H*2)
        preds_dict = {}
        mape_list = []
        ###############驗證集滾動預測 - 觀測穩定性###############
        for j in range(len(val_list)):
            val_train = df.loc[val_list[j] - train_size:val_list[j]]
            val_train.set_index('date',inplace=True)
            val_test = df.loc[val_list[j]+1:val_list[j]+H]
            val_test.set_index('date',inplace=True)
            train_set = val_train['close']
            test_set = val_test['close']
            #歸一化
            sc = MinMaxScaler(feature_range=(0,1))
            train_set = train_set.values.reshape(-1,1) #行數自動計算，將array變成1列的格式
            training_set_scaled = sc.fit_transform(train_set)
            X_train = [] 
            y_train = []
            for i in range(10,len(train_set)):
                X_train.append(training_set_scaled[i-10:i, 0]) 
                y_train.append(training_set_scaled[i, 0]) 
            X_train, y_train = np.array(X_train), np.array(y_train) 
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            keras.backend.clear_session()
            regressor = Sequential()
            regressor.add(LSTM(units = 100,input_shape=(X_train.shape[1],1)))
            regressor.add(Dense(units=1))
            regressor.compile(optimizer = 'rmsprop',loss = 'mean_squared_error')
            history = regressor.fit(X_train, y_train, epochs = 1000, batch_size = 16,verbose=0)
            total_inputs = val_train.close[-10:].values.tolist()
            predict_list = []
            for i in range(0,H):
                inputs = np.array(total_inputs).reshape(-1,1)
                inputs = sc.transform(inputs)
                X_test = []
                X_test.append(inputs[-10:, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
                predicted_stock_price = regressor.predict(X_test) 
                predicted_stock_price = sc.inverse_transform(predicted_stock_price)
                total_inputs.append(predicted_stock_price[0][0])
                predict_list.append(predicted_stock_price[0][0])
            calculate = Cal_Tool()
            mape = calculate.get_mape(val_test.close,predict_list)
            mape_list.append(mape)
            target = val_list[j]
            preds_dict[target] = predict_list
        ####################前一個月預測####################
        x_test = df['close'][:train_val_size]
        train_set = x_test
        test_set = y_test['close']
        #歸一化
        sc = MinMaxScaler(feature_range=(0,1))
        train_set = train_set.values.reshape(-1,1) #行數自動計算，將array變成1列的格式
        training_set_scaled = sc.fit_transform(train_set)
        X_train = [] 
        y_train = []
        for i in range(10,len(train_set)):
            X_train.append(training_set_scaled[i-10:i, 0]) 
            y_train.append(training_set_scaled[i, 0]) 
        X_train, y_train = np.array(X_train), np.array(y_train) 
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        keras.backend.clear_session()
        regressor = Sequential()
        regressor.add(LSTM(units=100,input_shape=(X_train.shape[1],1)))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer = 'rmsprop',loss = 'mean_squared_error')
        history = regressor.fit(X_train, y_train, epochs = 1000, batch_size = 16,verbose=0)
        total_inputs = val_train.close[-10:].values.tolist()
        preds_list = []
        for i in range(0,len(test_set)):
            inputs = np.array(total_inputs).reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            X_test.append(inputs[-10:, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
            predicted_stock_price = regressor.predict(X_test) 
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            total_inputs.append(predicted_stock_price[0][0])
            preds_list.append(predicted_stock_price[0][0])
        mape = calculate.get_mape(y_test.close,preds_list)
        ##儲存滾動訓練集預測成效圖表，以便後續做圖表
        mix_plot = df.loc[:,['date','close']]
        mix_plot.set_index('date',inplace=True)
        mix_plot.rename(columns={'close':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            test['Date'] = df['date'][key:key+H]
            test['Pred'] = preds_dict[key]#.values
            test.set_index('Date',inplace=True)
            for i in range(len(test)):
               mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]
        mix_plot.iloc[-len(preds_list):,1] = preds_list #補進去t-1 Month
        ##Overall Accuracy
        overall_score = (np.mean(mape_list) + mape)/2
        #######################產出預測清單######################
        total_inputs = df.close[-10:].values.tolist()
        preds_list = []
        calculate = Cal_Tool()
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        dates = list(filter(lambda date: date.weekday() <= 4 , dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        for i in range(0,len(dates)):
            inputs = np.array(total_inputs).reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            X_test.append(inputs[-10:, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
            predicted_stock_price = regressor.predict(X_test) 
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            total_inputs.append(predicted_stock_price[0][0])
            preds_list.append(predicted_stock_price[0][0])
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)
        return prediction,overall_score,mix_plot

    def gru_model(self,data,yyyy,mm,holidays):
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        last = df.date[len(df)-1].date()  #- pd.DateOffset(months=1)
        y_test = df.loc[(df.date.dt.year == last.year)&(df.date.dt.month == last.month),:]
        train_size = len(df) -1 - len(y_test) - 22 #-1不然會吃到test set的資料
        val_size = 22        
        train_val_size = train_size + val_size # Size of train+validation set
        H=22
        val_list = range(252*6, train_val_size, H*2)
        preds_dict = {}
        mape_list = []
        ###############驗證集滾動預測 - 觀測穩定性###############
        for j in range(len(val_list)):
            val_train = df.loc[val_list[j] - train_size:val_list[j]]
            val_train.set_index('date',inplace=True)
            val_test = df.loc[val_list[j]+1:val_list[j]+H]
            val_test.set_index('date',inplace=True)
            train_set = val_train['close']
            test_set = val_test['close']
            #歸一化
            sc = MinMaxScaler(feature_range=(0,1))
            train_set = train_set.values.reshape(-1,1) #行數自動計算，將array變成1列的格式
            training_set_scaled = sc.fit_transform(train_set)
            X_train = [] 
            y_train = []
            for i in range(10,len(train_set)):
                X_train.append(training_set_scaled[i-10:i, 0]) 
                y_train.append(training_set_scaled[i, 0]) 
            X_train, y_train = np.array(X_train), np.array(y_train) 
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            keras.backend.clear_session()
            regressor = Sequential()
            regressor.add(GRU(15,
                         activation = "tanh",
                         recurrent_activation = "sigmoid",
                         input_shape=(X_train.shape[1], X_train.shape[2])))
            regressor.add(Dropout(rate=0.2))
            regressor.add(Dense(1))
            regressor.compile(loss='mean_squared_error', optimizer = 'adam')
            history = regressor.fit(X_train, y_train,shuffle=False, epochs = 1000, batch_size = 32,validation_split=0.2,verbose=0)
            total_inputs = val_train.close[-10:].values.tolist()
            predict_list = []
            for i in range(0,H):
                inputs = np.array(total_inputs).reshape(-1,1)
                inputs = sc.transform(inputs)
                X_test = []
                X_test.append(inputs[-10:, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
                predicted_stock_price = regressor.predict(X_test) 
                predicted_stock_price = sc.inverse_transform(predicted_stock_price)
                total_inputs.append(predicted_stock_price[0][0])
                predict_list.append(predicted_stock_price[0][0])
            calculate = Cal_Tool()
            mape = calculate.get_mape(val_test.close,predict_list)
            mape_list.append(mape)
            target = val_list[j]
            preds_dict[target] = predict_list
        ####################前一個月預測####################
        x_test = df['close'][:train_val_size]
        train_set = x_test
        test_set = y_test['close']
        #歸一化
        sc = MinMaxScaler(feature_range=(0,1))
        train_set = train_set.values.reshape(-1,1) #行數自動計算，將array變成1列的格式
        training_set_scaled = sc.fit_transform(train_set)
        X_train = [] 
        y_train = []
        for i in range(10,len(train_set)):
            X_train.append(training_set_scaled[i-10:i, 0]) 
            y_train.append(training_set_scaled[i, 0]) 
        X_train, y_train = np.array(X_train), np.array(y_train) 
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        keras.backend.clear_session()
        regressor = Sequential()
        regressor.add(GRU(15,
                     activation = "tanh",
                     recurrent_activation = "sigmoid",
                     input_shape=(X_train.shape[1], X_train.shape[2])))
        regressor.add(Dropout(rate=0.2))
        regressor.add(Dense(1))
        regressor.compile(loss='mean_squared_error', optimizer = 'adam')
        history = regressor.fit(X_train, y_train,shuffle=False, epochs = 1000, batch_size = 32,validation_split=0.2,verbose=0)
        total_inputs = val_train.close[-10:].values.tolist()
        preds_list = []
        for i in range(0,len(test_set)):
            inputs = np.array(total_inputs).reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            X_test.append(inputs[-10:, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
            predicted_stock_price = regressor.predict(X_test) 
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            total_inputs.append(predicted_stock_price[0][0])
            preds_list.append(predicted_stock_price[0][0])
        mape = calculate.get_mape(y_test.close,preds_list)
        ##儲存滾動訓練集預測成效圖表，以便後續做圖表
        mix_plot = df.loc[:,['date','close']]
        mix_plot.set_index('date',inplace=True)
        mix_plot.rename(columns={'close':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            test['Date'] = df['date'][key:key+H]
            test['Pred'] = preds_dict[key]#.values
            test.set_index('Date',inplace=True)
            for i in range(len(test)):
               mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]
        mix_plot.iloc[-len(preds_list):,1] = preds_list #補進去t-1 Month
        ##Overall Accuracy
        overall_score = (np.mean(mape_list) + mape)/2
        #######################產出預測清單######################
        total_inputs = df.close[-10:].values.tolist()
        preds_list = []
        calculate = Cal_Tool()
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        dates = list(filter(lambda date: date.weekday() <= 4 , dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        for i in range(0,len(dates)):
            inputs = np.array(total_inputs).reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            X_test.append(inputs[-10:, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))    
            predicted_stock_price = regressor.predict(X_test) 
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            total_inputs.append(predicted_stock_price[0][0])
            preds_list.append(predicted_stock_price[0][0])
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)
        return prediction,overall_score,mix_plot
    
    def arima_model(self,data,yyyy,mm,H,holidays,predicted_interval):
        calculate = Cal_Tool()
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        train_size = 252*5
        val_size = 252         
        train_val_size = train_size + val_size # Size of train+validation set
        val_list = range(train_size, train_val_size, H*2)
        val_df = pd.DataFrame(columns= ['p','d','q','Mean_Mape'])
        for i in range(len(val_list)):
            val_train = df.loc[val_list[i] - train_size:val_list[i]]
            val_train.set_index('date',inplace=True)
            val_test = df.loc[val_list[i]+1:val_list[i]+H]
            val_test.set_index('date',inplace=True)
            test_signal = calculate.adf_test(val_train['close'])
            if test_signal == True:
                diff_1 = val_train['close'].diff(1) 
                diff_1 = diff_1.dropna()
                test_signal = calculate.adf_test(diff_1)
                best_d = 1
                if test_signal == True:
                    diff_2 = val_train['close'].diff(2) 
                    diff_2 = diff_2.dropna()
                    test_signal = calculate.adf_test(diff_2)
                    best_d = 2
                    if test_signal == True:
                        best_d = 3
            else:
                best_d = 1
            best_p,best_q = calculate.arima_mape(val_train['close'], p=5, d=best_d, q=5 ,period=H)
            try:
                model = ARIMA(val_train['close'], order=(best_p, best_d, best_q)) 
                fitted = model.fit(disp=-1,transparams=False)
            except:
                try:
                    model = ARIMA(val_train['close'], order=(best_p-1, best_d, best_q)) 
                    fitted = model.fit(disp=-1,transparams=False)       
                except:
                    model = ARIMA(val_train['close'], order=(best_p, best_d, best_q-1)) 
                    fitted = model.fit(disp=-1,transparams=False)                      
            fc, se, conf = fitted.forecast(H, alpha=0.05) # 95% conf
            fc_series = pd.Series(fc, index=val_test.index)
            mape = calculate.get_mape(val_test['close'],fc_series)
            line = pd.DataFrame({'p':best_p,'d':best_d,'q':best_q,'Mean_Mape':np.mean(mape)},index=[0])
            val_df = val_df.append(line,ignore_index=True)
        temp = val_df[val_df['Mean_Mape'] == val_df['Mean_Mape'].min()] 
        best_p = temp['p'].values[0]
        best_d = temp['d'].values[0]
        best_q = temp['q'].values[0]

        mape = [] # mean absolute percentage error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            if len(df[i:i+H]['close']) == H :
                try:
                    model = ARIMA(df.loc[i-train_val_size:i,'close'], order=(best_p, best_d, best_q)) 
                    fitted = model.fit(disp=-1,transparams=False)
                except:
                    try:
                        model = ARIMA(df.loc[i-train_val_size:i,'close'], order=(best_p-1, best_d, best_q)) 
                        fitted = model.fit(disp=-1,transparams=False)       
                    except:
                        model = ARIMA(df.loc[i-train_val_size:i,'close'], order=(best_p, best_d, best_q-1)) 
                        fitted = model.fit(disp=-1,transparams=False) 
                fitted = model.fit(disp=-1,transparams=False)
                fc, se, conf = fitted.forecast(H, alpha=0.05) # 95% conf
                preds_list = pd.Series(fc, index=val_test.index)

                # Collect the predictions
                preds_dict[i] = preds_list

                # Compute error metrics
                mape.append(calculate.get_mape(df[i:i+H]['close'], preds_list))

        #print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))
        #print("For forecast horizon %d, the mean MAPE is %f" % (H, np.mean(mape)))
        mix_plot = df.loc[:,['date','close']]
        mix_plot.set_index('date',inplace=True)
        mix_plot.rename(columns={'close':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            test['Date'] = df['date'][key:key+H]
            test['Pred'] = preds_dict[key].values
            test.set_index('Date',inplace=True)
            for i in range(len(test)):
               mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]
        last = df.date[len(df)-1].date()
        lastMonth = last - pd.offsets.MonthBegin(predicted_interval) #這裡可以填入predicted_interval
        y_test = df.loc[(df.date >= lastMonth),:]
        try:
            model = ARIMA(df.loc[~df.index.isin(y_test.index),'close'], order=(best_p, best_d, best_q)) 
            fitted = model.fit(disp=-1,transparams=False)
        except:
            try:
                model = ARIMA(df.loc[~df.index.isin(y_test.index),'close'], order=(best_p-1, best_d, best_q)) 
                fitted = model.fit(disp=-1,transparams=False)       
            except:
                model = ARIMA(df.loc[~df.index.isin(y_test.index),'close'], order=(best_p, best_d, best_q-1)) 
                fitted = model.fit(disp=-1,transparams=False) 
        fitted = model.fit(disp=-1,transparams=False)
        fc, se, conf = fitted.forecast(len(y_test), alpha=0.05) # 95% conf
        preds_list_last_month = pd.Series(fc)
        Overall_score = (np.mean(mape) + calculate.get_mape(y_test.close,preds_list_last_month))/2
        try:
            model = ARIMA(df.loc[:,'close'], order=(best_p, best_d, best_q)) 
            fitted = model.fit(disp=-1,transparams=False)
        except:
            try:
                model = ARIMA(df.loc[:,'close'], order=(best_p-1, best_d, best_q)) 
                fitted = model.fit(disp=-1,transparams=False)       
            except:
                model = ARIMA(df.loc[:,'close'], order=(best_p, best_d, best_q-1)) 
                fitted = model.fit(disp=-1,transparams=False) 
        fitted = model.fit(disp=-1,transparams=False)
        fc, se, conf = fitted.forecast(H, alpha=0.05) # 95% conf
        preds_list = pd.Series(fc)
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        for j in range(1,predicted_interval):
            dates.extend([datetime(yyyy, mm+j,i) for i in range(1,calculate.compute_day_month(mm+j).get('days')+1)])
        dates = list(filter(lambda date: date.weekday() <= 4, dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)
        return prediction,Overall_score,mix_plot
    
    def sarima_model(self,data,yyyy,mm,H,holidays,predicted_interval):
        calculate = Cal_Tool()
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        train_size = 252*5
        val_size = 252         
        train_val_size = train_size + val_size # Size of train+validation set
        val_list = range(train_size, train_val_size, H*2)
        val_df = pd.DataFrame(columns= ['order','seasonal order','Mean_Mape'])
        for i in range(len(val_list)):
            val_train = df.loc[val_list[i] - train_size:val_list[i]]
            val_train.set_index('date',inplace=True)
            val_test = df.loc[val_list[i]+1:val_list[i]+H]
            val_test.set_index('date',inplace=True)
            smodel = pm.auto_arima(val_train['close'][:-H],
                           start_p=1, 
                           start_q=1,
                           test='adf', #如果stationary為假且d為None，用來檢測平穩性的單位根檢驗的類型。默認為‘kpss’;可設置為adf
                           max_p=5, 
                           max_q=5, 
                           m=12, #frequency of series
                           start_P=0, #The starting value of P, the order of the auto-regressive portion of the seasonal model. 
                           seasonal=True, #加入季節性因素進去，為SARIMA的S
                           d=None, #The order of first-differencing. If None (by default), the value will automatically be selected based on the result
                           D=1,#The order of the seasonal differencing. If None (by default, the value will automatically be selected based on the results
                           trace=True, #是否打印適合的狀態。如果值為False，則不會打印任何調試信息。值為真會打印一些
                           error_action='ignore', #If unable to fit an ARIMA for whatever reason, this controls the error-handling behavior. 
                           suppress_warnings=True, #statsmodel中可能會拋出許多警告。如果suppress_warnings為真，那麽來自ARIMA的所有警告都將被壓制
                           stepwise=True
                          )
            fc, conf = smodel.predict(n_periods=H,alpha=0.05, return_conf_int=True)
            index_of_fc = np.arange(len(val_train), len(val_train)+H)
            #Make as pandas series
            fc_series = pd.Series(fc, index=val_test.index)
            lower_series = pd.Series(conf[:, 0], index=val_test.index)
            upper_series = pd.Series(conf[:, 1], index=val_test.index)    
            mape = calculate.get_mape(val_test['close'],fc_series)
            line = pd.DataFrame({'order':str(smodel.order),'seasonal order':str(smodel.seasonal_order),'Mean_Mape':np.mean(mape)},index=[0])
            val_df = val_df.append(line,ignore_index=True)
        temp = val_df[val_df['Mean_Mape'] == val_df['Mean_Mape'].min()] 
        best_order = temp['order'].values[0]
        best_seasonal_order = temp['seasonal order'].values[0]
        mape = [] # mean absolute percentage error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
            if len(df[i:i+H]['close']) == H:
                model = SARIMAX(df.loc[i-train_val_size:i,'close'], order=eval(best_order), seasonal_order=eval(best_seasonal_order)).fit(dis=-1)
                preds_list = model.predict(start=df.loc[i-train_val_size:i,'close'].index[-1]+1,end=df.loc[i-train_val_size:i,'close'].index[-1]+H,dynamic=True)
                preds_dict[i] = preds_list
                mape.append(calculate.get_mape(df[i:i+H]['close'], preds_list))
        #print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))
        #print("For forecast horizon %d, the mean MAPE is %f" % (H, np.mean(mape)))
        mix_plot = df.loc[:,['date','close']]
        mix_plot.set_index('date',inplace=True)
        mix_plot.rename(columns={'close':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            if len(df['date'][key:key+H]) == H:
                test = pd.DataFrame()
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]
        last = df.date[len(df)-1].date()
        lastMonth = last - pd.offsets.MonthBegin(predicted_interval) #這裡可以填入predicted_interval
        y_test = df.loc[(df.date >= lastMonth),:]
        model = SARIMAX(df.loc[~df.index.isin(y_test.index),'close'], order=eval(best_order), seasonal_order=eval(best_seasonal_order)).fit(dis=-1)
        preds_list_last_month = model.predict(start=df.loc[~df.index.isin(y_test.index),'close'].index[-1]+1,end=df.loc[~df.index.isin(y_test.index),'close'].index[-1]+len(y_test),dynamic=True)
        Overall_score = (np.mean(mape) + calculate.get_mape(y_test.close,preds_list_last_month))/2
        model = SARIMAX(df.loc[:,'close'], order=eval(best_order), seasonal_order=eval(best_seasonal_order)).fit(dis=-1)
        preds_list = model.predict(start=df.loc[:,'close'].index[-1]+1,end=df.loc[:,'close'].index[-1]+H,dynamic=True)
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        for j in range(1,predicted_interval):
            dates.extend([datetime(yyyy, mm+j,i) for i in range(1,calculate.compute_day_month(mm+j).get('days')+1)])
        dates = list(filter(lambda date: date.weekday() <= 4, dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)        
        return prediction,Overall_score,mix_plot
    
    def holt_model(self,data,yyyy,mm,H,holidays,predicted_interval):
        calculate = Cal_Tool()
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        train_size = 252*5
        val_size = 252         
        train_val_size = train_size + val_size # Size of train+validation set
        val_list = range(train_size, train_val_size, H*2)
        smoothing_lev = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        smoothing_sl = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        optimize = [True,False,None]
        exponent = [True,False,None]
        damp = [True,False,None]
        hyper_tuning = pd.DataFrame(columns=['smoothing_lev','smoothing_sl','optimized','exponent','damp','Mean_Mape'])
        for smoothing in tqdm_notebook(smoothing_lev):
            for smooth_sl in tqdm_notebook(smoothing_sl):
                for opt in tqdm_notebook(optimize):
                    for exp in tqdm_notebook(exponent):
                        for dam in tqdm_notebook(damp):
                            mape = []
                            for i in range(len(val_list)):
                                val_train = df.loc[val_list[i] - train_size:val_list[i]]
                                val_test = df.loc[val_list[i]+1:val_list[i]+H]
                                try:
                                   fit1 = Holt(val_train.loc[:,'close'],exponential=exp, damped=dam).fit(smoothing_level=smoothing, smoothing_slope= smooth_sl,optimized=opt)
                                   preds_list = fit1.forecast(H)
                                   mape.append(calculate.get_mape(val_test.loc[:,'close'],preds_list))
                                except :
                                   pass
                            line = pd.DataFrame({'smoothing_lev':smoothing,'smoothing_sl':smooth_sl,'optimized':opt,'exponent':exp,'damp':dam,'Mean_Mape':np.mean(mape)},index=[0])
                            hyper_tuning = hyper_tuning.append(line,ignore_index=True)

        temp = hyper_tuning[hyper_tuning['Mean_Mape'] == hyper_tuning['Mean_Mape'].min()] 
        best_smoothing_level = temp['smoothing_lev'].values[0]
        best_smoothing_slope = temp['smoothing_sl'].values[0]
        best_damped = temp['damp'].values[0]
        best_optimized = temp['optimized'].values[0]
        best_exponential = temp['exponent'].values[0]
        mape = [] # mean absolute percentage error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            
            if len(df[i:i+H]['close']) == H:
                fit1 = Holt(df.loc[i-train_val_size:i,'close'],exponential=best_exponential, damped=best_damped).fit(smoothing_level=best_smoothing_level, smoothing_slope= best_smoothing_slope,optimized=best_optimized)
                preds_list = fit1.forecast(H)

                # Collect the predictions
                preds_dict[i] = preds_list

                # Compute error metrics
                mape.append(calculate.get_mape(df[i:i+H]['close'], preds_list))

        mix_plot = df.loc[:,['date','close']]
        mix_plot.set_index('date',inplace=True)
        mix_plot.rename(columns={'close':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            if len(df['date'][key:key+H]) == H:
                test = pd.DataFrame()
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]
        Overall_score = np.mean(mape)
        fit1 = Holt(df.loc[:,'close'],exponential=best_exponential, damped=best_damped).fit(smoothing_level=best_smoothing_level, smoothing_slope= best_smoothing_slope,optimized=best_optimized)
        preds_list = fit1.forecast(H)
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        for j in range(1,predicted_interval):
            dates.extend([datetime(yyyy, mm+j,i) for i in range(1,calculate.compute_day_month(mm+j).get('days')+1)])
        dates = list(filter(lambda date: date.weekday() <= 4, dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)        
        return prediction,Overall_score,mix_plot

    def holt_winter_model(self,data,yyyy,mm,H,holidays,predicted_interval):
        calculate = Cal_Tool()
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        train_size = 252*5
        val_size = 252         
        train_val_size = train_size + val_size # Size of train+validation set
        val_list = range(train_size, train_val_size, H*2)
        trend_list = ['add','mul',None]
        seasonal_list = ['add','mul',None]
        damped_list = ['add','mul',None]
        seasonal_periods = [2,3,4,5,6,7,8]
        hyper_tuning = pd.DataFrame(columns=['Trend','Seasonal','Damped','Period','Mean_Mape'])
        for trend1 in tqdm_notebook(trend_list):
            for seasonal1 in tqdm_notebook(seasonal_list):
                for damped1 in tqdm_notebook(damped_list):
                    for period1 in tqdm_notebook(seasonal_periods):
                        mape = []
                        for i in range(len(val_list)):
                            val_train = df.loc[val_list[i] - train_size:val_list[i]]
                            val_test = df.loc[val_list[i]+1:val_list[i]+H]
                            try:
                               fit1 = ExponentialSmoothing(val_train.loc[:,'close'], seasonal_periods=period1, trend=trend1, seasonal=seasonal1,damped=damped1).fit(use_boxcox=True)
                               preds_list = fit1.forecast(H)
                               mape.append(calculate.get_mape(val_test.loc[:,'close'],preds_list))
                            except ValueError:
                               pass
                        line = pd.DataFrame({'Trend':trend1,'Seasonal':seasonal1,'Damped':damped1,'Period':period1,'Mean_Mape':np.mean(mape)},index=[0])
                        hyper_tuning = hyper_tuning.append(line,ignore_index=True)

        temp = hyper_tuning[hyper_tuning['Mean_Mape'] == hyper_tuning['Mean_Mape'].min()] 
        best_trend = temp['Trend'].values[0]
        best_seasonal = temp['Seasonal'].values[0]
        best_damped = temp['Damped'].values[0]
        best_period = temp['Period'].values[0]
        mape = [] # mean absolute percentage error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            
            if len(df[i:i+H]['close']) == H:
                fit1 = ExponentialSmoothing(df.loc[i-train_val_size:i,'close'], seasonal_periods=best_period, trend=best_trend, seasonal=best_seasonal,damped=best_damped).fit(use_boxcox=True)
                preds_list = fit1.forecast(H)

                # Collect the predictions
                preds_dict[i] = preds_list

                # Compute error metrics
                mape.append(calculate.get_mape(df[i:i+H]['close'], preds_list))

        mix_plot = df.loc[:,['date','close']]
        mix_plot.set_index('date',inplace=True)
        mix_plot.rename(columns={'close':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            if len(df['date'][key:key+H]) == H:
                test = pd.DataFrame()
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]
        Overall_score = np.mean(mape)
        fit1 = ExponentialSmoothing(df.loc[:,'close'], seasonal_periods=best_period, trend=best_trend, seasonal=best_seasonal,damped=best_damped).fit(use_boxcox=True)
        preds_list = fit1.forecast(H)
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        for j in range(1,predicted_interval):
            dates.extend([datetime(yyyy, mm+j,i) for i in range(1,calculate.compute_day_month(mm+j).get('days')+1)])
        dates = list(filter(lambda date: date.weekday() <= 4, dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(holidays.ds).tolist(), dates))
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)        
        return prediction,Overall_score,mix_plot
    
    def prophet_model(self,data,yyyy,mm,H,holidays,predicted_interval):
        
        global train_size
        train_size = 252*5              # Use 5 years of data as train set. Note there are about 252 trading days in a year
        val_size = 252                  # Use 1 year of data as validation set
        changepoint_prior_scale_list = [0.05, 0.5, 1, 1.5, 2.5]     # for hyperparameter tuning
        fourier_order_list = [None, 2, 4, 6, 8, 10]                 # for hyperparameter tuning
        window_list = [None, 0, 1, 2]                               # for hyperparameter tuning

        fontsize = 14
        ticklabelsize = 14
        train_val_size = train_size + val_size # Size of train+validation set

        hols_df = holidays.copy()
        m_list = pd.DataFrame(columns=['Method','CP','Season','Holidays','Mape'])
        calculate = Cal_Tool()
        df = data.copy()
        df = df.reset_index() #把Date找回來
        df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
        df_prophet = df[['date', 'close']].rename(columns={'date':'ds', 'close':'y'})

        # Predict using Prophet, and compute error metrics also
        print('***************Start prediction based on just validation set*************')
        rmse = [] # root mean square error
        mape = [] # mean absolute percentage error
        mae = []  # mean absolute error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            print("Predicting on day %d, date %s " % (i, df_prophet['ds'][i]))

            preds_list = get_preds_prophet(df_prophet[i-train_val_size:i],H,hols_df)

            # Collect the predictions
            preds_dict[i] = preds_list

            # Compute error metrics
            if len(df_prophet[i:i+H]['y']) == H:
                rmse.append(calculate.get_rmse(df_prophet[i:i+H]['y'], preds_list))
                mape.append(calculate.get_mape(df_prophet[i:i+H]['y'], preds_list))
                mae.append(calculate.get_mae(df_prophet[i:i+H]['y'], preds_list))

        print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))
        m_list = m_list.append({'Method':'No Hyper param Tune','CP':0.05,'Season':None,'Holidays':None,'Mape': np.mean(mape)},ignore_index=True)
        results_1 = pd.DataFrame({'day': i_list[:len(mape)],
                                      'changepoint_opt': 0.05,
                                      'fourier_order_opt': None,
                                      'window_opt': None,
                                      'rmse': rmse,
                                      'mape': mape,
                                      'mae': mae})
        no_plot = df_prophet.loc[:,['ds','y']]
        no_plot.set_index('ds',inplace=True)
        no_plot.rename(columns={'y':'Actual'},inplace=True)
        no_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            if len(df['date'][key:key+H]) == H:
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   no_plot.loc[no_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        no_plot = no_plot.loc[no_plot.Pred.index >= no_plot.Pred.first_valid_index(),:]

        print('***************Tuning CP based on validation set*************')
        i = train_val_size # Predict for day i, for the next H-1 days. Note indexing of days start from 0.
        # Get optimum hyperparams
        tic = time.time()
        changepoint_opt, results =       hyperparam_tune_cp(df_prophet[i-train_val_size:i], 
                                                            H, 
                                                            hols_df,
                                                            train_size, 
                                                            val_size, 
                                                            changepoint_prior_scale_list)
        toc = time.time()
        print("Time taken = " + str((toc-tic)/60.0) + " mins")

        print("changepoint_opt = " + str(changepoint_opt))
        # Compute error metrics
        preds_list = get_preds_prophet(df_prophet[i-train_val_size:i], H,hols_df, changepoint_prior_scale=changepoint_opt)
        # Predict using Prophet, and compute error metrics also
        rmse = [] # root mean square error
        mape = [] # mean absolute percentage error
        mae = []  # mean absolute error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            print("Predicting on day %d, date %s" % (i, df_prophet['ds'][i]))

            # Get predictions using tuned hyperparams
            preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                 H, 
                                                 hols_df,
                                                 changepoint_opt
                                                 )
            # Collect the predictions
            preds_dict[i] = preds_list

            # Compute error metrics
            if len(df_prophet[i:i+H]['y']) == H:
                rmse.append(calculate.get_rmse(df_prophet[i:i+H]['y'], preds_list))
                mape.append(calculate.get_mape(df_prophet[i:i+H]['y'], preds_list))
                mae.append(calculate.get_mae(df_prophet[i:i+H]['y'], preds_list))

        print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))
        results_2 = pd.DataFrame({'day': i_list[:len(mape)],
                                      'changepoint_opt': changepoint_opt,
                                      'fourier_order_opt': None,
                                      'window_opt': None,
                                      'rmse': rmse,
                                      'mape': mape,
                                      'mae': mae})
        m_list = m_list.append({'Method':'CP Tune','CP':changepoint_opt,'Season':None,'Holidays':None,'Mape': np.mean(mape)},ignore_index=True)
        cp_plot = df_prophet.loc[:,['ds','y']]
        cp_plot.set_index('ds',inplace=True)
        cp_plot.rename(columns={'y':'Actual'},inplace=True)
        cp_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            if len(df['date'][key:key+H]) == H:
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   cp_plot.loc[cp_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        cp_plot = cp_plot.loc[cp_plot.Pred.index >= cp_plot.Pred.first_valid_index(),:]

        print('***************Tuning Season based on validation set*************')
        i = train_val_size
        # Get optimum hyperparams
        tic = time.time()
        fourier_order_opt, results =       hyperparam_tune_fo(df_prophet[i-train_val_size:i], 
                                                              H, 
                                                              hols_df,
                                                              train_size, 
                                                              val_size, 
                                                              fourier_order_list)
        toc = time.time()
        print("Time taken = " + str((toc-tic)/60.0) + " mins")

        print("fourier_order_opt = " + str(fourier_order_opt))
        preds_list = get_preds_prophet(df_prophet[i-train_val_size:i], H,hols_df,fourier_order=fourier_order_opt)
        # Predict using Prophet, and compute error metrics also
        rmse = [] # root mean square error
        mape = [] # mean absolute percentage error
        mae = []  # mean absolute error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            print("Predicting on day %d, date %s" % (i, df_prophet['ds'][i]))

            # Get predictions using tuned hyperparams
            preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                 H, 
                                                 hols_df,
                                                 0.05,
                                                 fourier_order_opt)
            # Collect the predictions
            preds_dict[i] = preds_list

            # Compute error metrics
            if len(df_prophet[i:i+H]['y']) == H:
                rmse.append(calculate.get_rmse(df_prophet[i:i+H]['y'], preds_list))
                mape.append(calculate.get_mape(df_prophet[i:i+H]['y'], preds_list))
                mae.append(calculate.get_mae(df_prophet[i:i+H]['y'], preds_list))

        print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))
        results_3 = pd.DataFrame({'day': i_list[:len(mape)],
                                      'changepoint_opt': 0.05,
                                      'fourier_order_opt': fourier_order_opt,
                                      'window_opt': None,
                                      'rmse': rmse,
                                      'mape': mape,
                                      'mae': mae})
        m_list = m_list.append({'Method':'Seanson Tune','CP':0.05,'Season':fourier_order_opt,'Holidays':None,'Mape': np.mean(mape)},ignore_index=True)
        season_plot = df_prophet.loc[:,['ds','y']]
        season_plot.set_index('ds',inplace=True)
        season_plot.rename(columns={'y':'Actual'},inplace=True)
        season_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            if len(df['date'][key:key+H]) == H:
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   season_plot.loc[season_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        season_plot = season_plot.loc[season_plot.Pred.index >= season_plot.Pred.first_valid_index(),:]

        print('***************Tuning Holiday based on validation set*************')
        i = train_val_size
        hols = holidays.copy()
        # Get optimum hyperparams
        tic = time.time()
        window_opt, results =       hyperparam_tune_wd(df_prophet[i-train_val_size:i], 
                                                       H, 
                                                       hols_df,
                                                       train_size, 
                                                       val_size, 
                                                       window_list,
                                                       hols)
        toc = time.time()
        print("Time taken = " + str((toc-tic)/60.0) + " mins")

        print("window_opt = " + str(window_opt))
        # Get predictions using tuned hyperparams
        if window_opt is None:
            preds_list = get_preds_prophet(df_prophet[i-train_val_size:i], H,hols_df, holidays=None)
        else:
            hols['lower_window'] = -window_opt
            hols['upper_window'] = +window_opt
            preds_list = get_preds_prophet(df_prophet[i-train_val_size:i], H,hols_df, holidays=hols)

        # Predict using Prophet, and compute error metrics also
        hols1 = holidays.copy()
        rmse = [] # root mean square error
        mape = [] # mean absolute percentage error
        mae = []  # mean absolute error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            print("Predicting on day %d, date %s" % (i, df_prophet['ds'][i]))

            # Get predictions using tuned hyperparams
            if window_opt != None:
                hols1['lower_window'] = -window_opt
                hols1['upper_window'] = +window_opt
                preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                 H,
                                                 hols_df,
                                                 holidays=hols1
                                                )
            else:
                preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                 H,
                                                 hols_df,
                                                 holidays=None
                                                )                
            # Collect the predictions
            preds_dict[i] = preds_list

            # Compute error metrics
            if len(df_prophet[i:i+H]['y']) == H:
                rmse.append(calculate.get_rmse(df_prophet[i:i+H]['y'], preds_list))
                mape.append(calculate.get_mape(df_prophet[i:i+H]['y'], preds_list))
                mae.append(calculate.get_mae(df_prophet[i:i+H]['y'], preds_list))

        print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))

        results_4 = pd.DataFrame({'day': i_list[:len(mape)],
                                  'changepoint_opt': 0.05,
                                  'fourier_order_opt': None,
                                  'window_opt':window_opt,
                                  'rmse': rmse,
                                  'mape': mape,
                                  'mae': mae})

        m_list = m_list.append({'Method':'Holidays Tune','CP':0.05,'Season':None,'Holidays':window_opt,'Mape': np.mean(mape)},ignore_index=True)
        holidays_plot = df_prophet.loc[:,['ds','y']]
        holidays_plot.set_index('ds',inplace=True)
        holidays_plot.rename(columns={'y':'Actual'},inplace=True)
        holidays_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            if len(df['date'][key:key+H]) == H:
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   holidays_plot.loc[holidays_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        holidays_plot = holidays_plot.loc[holidays_plot.Pred.index >= holidays_plot.Pred.first_valid_index(),:]

        print('***************Tuning CP/Season/Holidays based on validation set*************')
        i = train_val_size
        hols1 = holidays.copy()
        # Get optimum hyperparams
        tic = time.time()
        changepoint_prior_scale_opt, fourier_order_opt, window_opt_mix, results = \
                  hyperparam_tune_cp_fo_wd(df_prophet[i-train_val_size:i], 
                                           H, 
                                           hols_df,
                                           train_size, 
                                           val_size, 
                                           changepoint_prior_scale_list,
                                           fourier_order_list,
                                           window_list,
                                           hols1)
        toc = time.time()
        print("Time taken = " + str((toc-tic)/60.0) + " mins")

        print("changepoint_prior_scale_opt = " + str(changepoint_prior_scale_opt))
        print("fourier_order_opt = " + str(fourier_order_opt))
        print("window_opt = " + str(window_opt_mix))

        # Get predictions using tuned hyperparams
        if (window_opt_mix is None) or (np.isnan(window_opt_mix)):
            preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                 H, 
                                                 hols_df,
                                                 changepoint_prior_scale_opt, 
                                                 fourier_order_opt, 
                                                 holidays=None)
        else:
            hols1['lower_window'] = -window_opt_mix
            hols1['upper_window'] = +window_opt_mix
            preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                 H, 
                                                 hols_df,
                                                 changepoint_prior_scale_opt, 
                                                 fourier_order_opt, 
                                                 holidays=hols1)

        hols1 = holidays.copy()
        # Predict using Prophet, and compute error metrics also
        rmse = [] # root mean square error
        mape = [] # mean absolute percentage error
        mae = []  # mean absolute error
        preds_dict = {}
        i_list = range(train_val_size, len(df), H*2) #在預測完驗證集，把後面的數據做為測試集計算平均表現
        for i in i_list:
        # for i in tqdm_notebook(range(train_val_size, len(df)-H, int(H/2))): # Do a forecast on day i
            print("Predicting on day %d, date %s" % (i, df_prophet['ds'][i]))

            # Get predictions using tuned hyperparams
            if (window_opt_mix is None) or (np.isnan(window_opt_mix)):
                preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                     H, 
                                                     hols_df,
                                                     changepoint_prior_scale_opt, 
                                                     fourier_order_opt, 
                                                     holidays=None)
            else:
                hols1['lower_window'] = -window_opt_mix
                hols1['upper_window'] = +window_opt_mix
                preds_list =       get_preds_prophet(df_prophet[i-train_val_size:i], 
                                                     H, 
                                                     hols_df,
                                                     changepoint_prior_scale_opt, 
                                                     fourier_order_opt, 
                                                     holidays=hols1)

            # Collect the predictions
            preds_dict[i] = preds_list

            # Compute error metrics
            if len(df_prophet[i:i+H]['y']) == H:
                rmse.append(get_rmse(df_prophet[i:i+H]['y'], preds_list))
                mape.append(get_mape(df_prophet[i:i+H]['y'], preds_list))
                mae.append(get_mae(df_prophet[i:i+H]['y'], preds_list))

        print("Altogether we made %d forecasts, each of length %d days" % (len(rmse), H))

        m_list = m_list.append({'Method':'CP/SS/Holidays Tune','CP':changepoint_prior_scale_opt,'Season':fourier_order_opt,'Holidays':window_opt_mix,'Mape': np.mean(mape)},ignore_index=True)

        results_final_no_tune = pd.DataFrame({'day': i_list[:len(mape)],
                                              'changepoint_opt': [changepoint_prior_scale_opt]*len(i_list[:len(mape)]),
                                              'fourier_order_opt': [fourier_order_opt]*len(i_list[:len(mape)]),
                                              'window_opt_mix': [window_opt_mix]*len(i_list[:len(mape)]),
                                              'rmse': rmse,
                                              'mape': mape,
                                              'mae': mae})
        mix_plot = df_prophet.loc[:,['ds','y']]
        mix_plot.set_index('ds',inplace=True)
        mix_plot.rename(columns={'y':'Actual'},inplace=True)
        mix_plot['Pred'] = None
        for key in preds_dict:
            test = pd.DataFrame()
            if len(df['date'][key:key+H]) == H:
                test['Date'] = df['date'][key:key+H]
                test['Pred'] = preds_dict[key].values
                test.set_index('Date',inplace=True)
                for i in range(len(test)):
                   mix_plot.loc[mix_plot.index==test.index[i],'Pred'] = test.loc[test.index[i],'Pred']
        mix_plot = mix_plot.loc[mix_plot.Pred.index >= mix_plot.Pred.first_valid_index(),:]

        ######Overall Prediction########
        i = train_val_size
        if m_list.loc[m_list['Mape'] == min(m_list['Mape']),'Method'].values[0] == 'CP Tune':
            preds_list =       get_preds_prophet(df_prophet, 
                                                       H, 
                                                       hols_df,
                                                       changepoint_prior_scale = m_list.loc[m_list['Method']=='CP Tune','CP'].values[0]
                                                       )
            train_plot = cp_plot.copy()
        elif m_list.loc[m_list['Mape'] == min(m_list['Mape']),'Method'].values[0] == 'No Hyper param Tune':
            preds_list = get_preds_prophet(df_prophet,H,hols_df)
            train_plot = no_plot.copy()
        elif m_list.loc[m_list['Mape'] == min(m_list['Mape']),'Method'].values[0] == 'Seanson Tune':
            preds_list =       get_preds_prophet(df_prophet, 
                                                 H, 
                                                 hols_df,
                                                 fourier_order= m_list.loc[m_list['Method']=='Seanson Tune','Season'].values[0]
                                                 )
            train_plot = season_plot.copy()
        elif m_list.loc[m_list['Mape'] == min(m_list['Mape']),'Method'].values[0] == 'Holidays Tune':
            if (window_opt is None) or (np.isnan(window_opt)):
                preds_list = get_preds_prophet(df_prophet, 
                                                     H,
                                                     hols_df,
                                                     holidays=None
                                                     ) 
                train_plot = holidays_plot.copy()
            else:
                hols1 = holidays.copy()
                hols1['lower_window'] = -window_opt
                hols1['upper_window'] = +window_opt
                preds_list = get_preds_prophet(df_prophet, 
                                                     H,
                                                     hols_df,
                                                     holidays=hols1
                                                     ) 
                train_plot = holidays_plot.copy()
        else:
            if (window_opt_mix is None) or (np.isnan(window_opt_mix)):
                preds_list = get_preds_prophet(df_prophet, 
                                                     H,
                                                     hols_df,
                                                     changepoint_prior_scale = m_list.loc[m_list['Method']=='CP/SS/Holidays Tune','CP'].values[0], 
                                                     fourier_order = m_list.loc[m_list['Method']=='CP/SS/Holidays Tune','Season'].values[0], 
                                                     holidays=None)
                train_plot = mix_plot.copy()
            else:
                hols1 = holidays.copy()
                hols1['lower_window'] = -window_opt_mix
                hols1['upper_window'] = +window_opt_mix
                preds_list = get_preds_prophet(df_prophet, 
                                                     H,
                                                     hols_df,
                                                     changepoint_prior_scale = m_list.loc[m_list['Method']=='CP/SS/Holidays Tune','CP'].values[0], 
                                                     fourier_order = m_list.loc[m_list['Method']=='CP/SS/Holidays Tune','Season'].values[0], 
                                                     holidays=hols1)
                train_plot = mix_plot.copy()                 

        Overall_score = min(m_list['Mape'])
        dates = [datetime(yyyy, mm,i) for i in range(1,calculate.compute_day_month(mm).get('days')+1)]
        for j in range(1,predicted_interval):
            dates.extend([datetime(yyyy, mm+j,i) for i in range(1,calculate.compute_day_month(mm+j).get('days')+1)])
        dates = list(filter(lambda date: date.weekday() <= 4, dates))
        dates = list(filter(lambda date: date not in pd.to_datetime(hols.ds).tolist(), dates))
        prediction = pd.DataFrame({'Date': dates,'FCST':preds_list[:len(dates)]})
        prediction['漲跌'] = prediction['FCST'] -  prediction['FCST'].shift(1)
        return prediction, Overall_score, train_plot
    


################Implementation##################
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    stock_numbers = ['Taiwan Paper','Taiwan Steel','Taiwan Plastic','CN Aluminum','UK Aluminum','USD/TWD']
    stock_numbers = ['USD/TWD']
    predicted_intervals = [1,3,6]
    yyyy = 2022
    mm = 8
    for stock_number in stock_numbers:
        if stock_number == 'USD/TWD':
            stock_number = "USD:TWD" 
        if not os.path.exists('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}'.format(stock_number,yyyy,mm)):
            os.mkdir('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}'.format(stock_number,yyyy,mm))
            os.mkdir('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}'.format(stock_number,yyyy,mm))
            os.mkdir('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}'.format(stock_number,yyyy,mm))
        product = stock_number
        for predicted_interval in predicted_intervals:
            if stock_number == 'USD:TWD':
                stock_number = "USD/TWD" 
            if predicted_interval == 1:
                print('{} {} 開始訓練'.format(product, predicted_interval))
                #LightGBM
                mon_count = Cal_Tool() 
                if mm == 1:
                    days = mon_count.month_weekdays(yyyy-1,12) #mm-1
                else:
                    days = mon_count.month_weekdays(yyyy,mm-1) #mm-1
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data()
                fcst_days = mon_count.month_weekdays(yyyy,mm)
                stock_engine.stock_data_real = stock_engine.get_analysis_index(df = stock_engine.stock_data,post_close=False)[-fcst_days:]
                stock_engine.get_analysis_index(df = stock_engine.stock_data,post_close=True)
                stock_engine.data_split(period=10)
                model = Model(product=product)
                list_gbm,score_gbm_acc, score_gbm,gbm_test_plot,gbm_change_plot,gbm_importance_plot = model.Light_gbm(X_train = stock_engine.X_train,
                                                                                                                    y_train = stock_engine.y_train,
                                                                                                                    X_test = stock_engine.X_test,
                                                                                                                    y_test = stock_engine.y_test,
                                                                                                                    X_real = stock_engine.X_real,
                                                                                                                    sc = stock_engine.sc,
                                                                                                                    yyyy = stock_engine.yyyy,
                                                                                                                    mm = stock_engine.mm,
                                                                                                                    holidays = stock_engine.hols)    
                #LSTM
                stock_engine.get_stock_data()
                pred_lstm, lstm_score, lstm_train_plot = model.lstm_model(data=stock_engine.stock_data,
                                                                          yyyy= stock_engine.yyyy,
                                                                          mm=stock_engine.mm,
                                                                          holidays=stock_engine.hols)
                #GRU
                pred_gru, gru_score, gru_train_plot = model.gru_model(data=stock_engine.stock_data,
                                                                      yyyy= stock_engine.yyyy,
                                                                      mm=stock_engine.mm,
                                                                      holidays=stock_engine.hols)

                #ARIMA
                pred_arima , arima_score , arima_train_plot =  model.arima_model(data = stock_engine.stock_data,
                                                                                 yyyy = stock_engine.yyyy,
                                                                                 mm = stock_engine.mm,
                                                                                 H = 22 * predicted_interval,
                                                                                 holidays = stock_engine.hols,
                                                                                 predicted_interval = predicted_interval)


                #SARIMA
                pred_sarima , sarima_score , sarima_train_plot =  model.sarima_model(data = stock_engine.stock_data,
                                                                                     yyyy = stock_engine.yyyy,
                                                                                     mm = stock_engine.mm,
                                                                                     H = 22 * predicted_interval,
                                                                                     holidays = stock_engine.hols,
                                                                                     predicted_interval = predicted_interval)

                #Holt 
                pred_holt, holt_score , holt_train_plot = model.holt_model(data = stock_engine.stock_data,
                                                                           yyyy = stock_engine.yyyy,
                                                                           mm = stock_engine.mm,
                                                                           H = 22 * predicted_interval,
                                                                           holidays = stock_engine.hols,
                                                                           predicted_interval = predicted_interval)

                #Holt Winter
                pred_holt_winter, holt_winter_score , holt_winter_plot = model.holt_winter_model(data = stock_engine.stock_data,
                                                                                                 yyyy = stock_engine.yyyy,
                                                                                                 mm = stock_engine.mm,
                                                                                                 H = 22 * predicted_interval,
                                                                                                 holidays = stock_engine.hols,
                                                                                                 predicted_interval = predicted_interval)

                #Prophet
                pred_prophet, prophet_score , prophet_train_plot = model.prophet_model(  data = stock_engine.stock_data,
                                                                                   yyyy = stock_engine.yyyy,
                                                                                   mm = stock_engine.mm,
                                                                                   H = 22 * predicted_interval,
                                                                                   holidays = stock_engine.hols,
                                                                                   predicted_interval = predicted_interval
                                                                                   )
                com =  [('lstm',round(lstm_score,2),0),
                        ('GRU',round(gru_score,2),0),
                        ('LightGBM',round(score_gbm,2),round(score_gbm_acc,2)),
                        ('Holt',round( holt_score,2),0),
                        ('Holt Winter',round(holt_winter_score,2),0),
                        ('ARIMA',round(arima_score,2),0),
                        ('SARIMA',round(sarima_score,2),0),
                        ('Prophet',round(prophet_score,2),0)]
                compare = pd.DataFrame(com,columns=["model","score","acc_for_gbm"])
                compare.sort_values('score',inplace=True)
                compare.reset_index(drop = True,inplace=True)
                if stock_number == 'USD/TWD':
                    stock_number = "USD:TWD"
                    product = "USD:TWD"
                compare.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
                if compare.iloc[0,0]== 'LightGBM' and round(score_gbm_acc,1) < 0.6:
                    print("""最佳預測模型：{}""".format(compare.iloc[1,0]))
                    if compare.iloc[1,0] == 'lstm': 
                        pred_lstm,lstm_train_plot = stock_engine.transform_df(pred_lstm,lstm_train_plot,stock_number)
                        lstm_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_lstm.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[1,0] == 'GRU': 
                        pred_gru,gru_train_plot = stock_engine.transform_df(pred_gru,gru_train_plot,stock_number)
                        gru_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_gru_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_gru.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_gru.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[1,0] == 'Holt': 
                        pred_holt,holt_train_plot = stock_engine.transform_df(pred_holt,holt_train_plot,stock_number)
                        holt_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_holt.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[1,0] == 'Holt Winter': 
                        pred_holt_winter,holt_winter_plot = stock_engine.transform_df(pred_holt_winter,holt_winter_plot,stock_number)
                        holt_winter_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_holt_winter.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[1,0] == 'ARIMA': 
                        pred_arima,arima_train_plot = stock_engine.transform_df(pred_arima,arima_train_plot,stock_number)
                        arima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_arima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[1,0] == 'SARIMA': 
                        pred_sarima,sarima_train_plot = stock_engine.transform_df(pred_sarima,sarima_train_plot,stock_number)
                        sarima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_sarima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[1,0] == 'Prophet': 
                        pred_prophet,prophet_train_plot = stock_engine.transform_df(pred_prophet,prophet_train_plot,stock_number)
                        prophet_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_prophet.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,product))
                else:
                    print("""最佳預測模型：{}""".format(compare.iloc[0,0]))
                    if compare.iloc[0,0] == 'lstm': 
                        pred_lstm,lstm_train_plot = stock_engine.transform_df(pred_lstm,lstm_train_plot,stock_number)
                        lstm_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_lstm.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'GRU': 
                        pred_gru,gru_train_plot = stock_engine.transform_df(pred_gru,gru_train_plot,stock_number)
                        gru_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_gru_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_gru.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_gru.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'Holt':
                        pred_holt,holt_train_plot = stock_engine.transform_df(pred_holt,holt_train_plot,stock_number)
                        holt_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_holt.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'Holt Winter':
                        pred_holt_winter,holt_winter_plot = stock_engine.transform_df(pred_holt_winter,,holt_winter_plot,stock_number)
                        holt_winter_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_holt_winter.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'ARIMA':
                        pred_arima,arima_train_plot = stock_engine.transform_df(pred_arima,arima_train_plot,stock_number)
                        arima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_arima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'SARIMA': 
                        pred_sarima,sarima_train_plot = stock_engine.transform_df(pred_sarima,sarima_train_plot,stock_number)
                        sarima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_sarima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'Prophet':
                        pred_prophet,prophet_train_plot = stock_engine.transform_df(pred_prophet,prophet_train_plot,stock_number)
                        prophet_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        pred_prophet.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,product))
                    elif compare.iloc[0,0] == 'LightGBM':
                        list_gbm,gbm_test_plot = stock_engine.transform_df(list_gbm,gbm_test_plot,stock_number)
                        gbm_change_plot = stock_engine.transform_change_df(gbm_change_plot,stock_number)
                        list_gbm.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_Best_by_light.xlsx'.format(stock_number,yyyy,mm,product))
                        gbm_test_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_gbm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        gbm_change_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_gbm_change_plot.xlsx'.format(stock_number,yyyy,mm,product))
                        gbm_importance_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/1 MONTH/{}{}/{}_for_gbm_importance_plot.xlsx'.format(stock_number,yyyy,mm,product))

            elif predicted_interval == 3:
                print('{} {} 開始訓練'.format(product, predicted_interval))
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data()   
                model = Model(product=product)
                #ARIMA
                pred_arima , arima_score , arima_train_plot =  model.arima_model(data = stock_engine.stock_data,
                                                                                 yyyy = stock_engine.yyyy,
                                                                                 mm = stock_engine.mm,
                                                                                 H = 22 * predicted_interval,
                                                                                 holidays = stock_engine.hols,
                                                                                 predicted_interval = predicted_interval)


                #SARIMA
                pred_sarima , sarima_score , sarima_train_plot =  model.sarima_model(data = stock_engine.stock_data,
                                                                                     yyyy = stock_engine.yyyy,
                                                                                     mm = stock_engine.mm,
                                                                                     H = 22 * predicted_interval,
                                                                                     holidays = stock_engine.hols,
                                                                                     predicted_interval = predicted_interval)

                #Holt 
                pred_holt, holt_score , holt_train_plot = model.holt_model(data = stock_engine.stock_data,
                                                                           yyyy = stock_engine.yyyy,
                                                                           mm = stock_engine.mm,
                                                                           H = 22 * predicted_interval,
                                                                           holidays = stock_engine.hols,
                                                                           predicted_interval = predicted_interval)

                #Holt Winter
                pred_holt_winter, holt_winter_score , holt_winter_plot = model.holt_winter_model(data = stock_engine.stock_data,
                                                                                                 yyyy = stock_engine.yyyy,
                                                                                                 mm = stock_engine.mm,
                                                                                                 H = 22 * predicted_interval,
                                                                                                 holidays = stock_engine.hols,
                                                                                                 predicted_interval = predicted_interval)

                #Prophet
                pred_prophet, prophet_score , prophet_train_plot = model.prophet_model(  data = stock_engine.stock_data,
                                                                                   yyyy = stock_engine.yyyy,
                                                                                   mm = stock_engine.mm,
                                                                                   H = 22 * predicted_interval,
                                                                                   holidays = stock_engine.hols,
                                                                                   predicted_interval = predicted_interval
                                                                                   )    

                com =  [('Holt',round( holt_score,2)),
                        ('Holt Winter',round(holt_winter_score,2)),
                        ('ARIMA',round(arima_score,2)),
                        ('SARIMA',round(sarima_score,2)),
                        ('Prophet',round(prophet_score,2))]
                compare = pd.DataFrame(com,columns=["model","score"])
                compare.sort_values('score',inplace=True)
                compare.reset_index(drop = True,inplace=True)
                if stock_number == 'USD/TWD':
                    stock_number = "USD:TWD"
                    product = "USD:TWD"
                compare.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
                print("""最佳預測模型：{}""".format(compare.iloc[0,0]))
                if compare.iloc[0,0] == 'Holt': 
                    pred_holt,holt_train_plot = stock_engine.transform_df(pred_holt,holt_train_plot,stock_number)
                    holt_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_holt.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'Holt Winter':
                    pred_holt_winter,holt_winter_plot = stock_engine.transform_df(pred_holt_winter,,holt_winter_plot,stock_number)
                    holt_winter_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_holt_winter.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'ARIMA':
                    pred_arima,arima_train_plot = stock_engine.transform_df(pred_arima,arima_train_plot,stock_number)
                    arima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_arima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'SARIMA':
                    pred_sarima,sarima_train_plot = stock_engine.transform_df(pred_sarima,sarima_train_plot,stock_number)
                    sarima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_sarima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'Prophet':
                    pred_prophet,prophet_train_plot = stock_engine.transform_df(pred_prophet,prophet_train_plot,stock_number)
                    prophet_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_prophet.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/3 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,product))
            elif predicted_interval == 6:
                print('{} {} 開始訓練'.format(product, predicted_interval))
                stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
                stock_engine.get_stock_data()   
                model = Model(product=product)
                #ARIMA
                pred_arima , arima_score , arima_train_plot =  model.arima_model(data = stock_engine.stock_data,
                                                                                 yyyy = stock_engine.yyyy,
                                                                                 mm = stock_engine.mm,
                                                                                 H = 22 * predicted_interval,
                                                                                 holidays = stock_engine.hols,
                                                                                 predicted_interval = predicted_interval)


                #SARIMA
                pred_sarima , sarima_score , sarima_train_plot =  model.sarima_model(data = stock_engine.stock_data,
                                                                                     yyyy = stock_engine.yyyy,
                                                                                     mm = stock_engine.mm,
                                                                                     H = 22 * predicted_interval,
                                                                                     holidays = stock_engine.hols,
                                                                                     predicted_interval = predicted_interval)

                #Holt 
                pred_holt, holt_score , holt_train_plot = model.holt_model(data = stock_engine.stock_data,
                                                                           yyyy = stock_engine.yyyy,
                                                                           mm = stock_engine.mm,
                                                                           H = 22 * predicted_interval,
                                                                           holidays = stock_engine.hols,
                                                                           predicted_interval = predicted_interval)

                #Holt Winter
                pred_holt_winter, holt_winter_score , holt_winter_plot = model.holt_winter_model(data = stock_engine.stock_data,
                                                                                                 yyyy = stock_engine.yyyy,
                                                                                                 mm = stock_engine.mm,
                                                                                                 H = 22 * predicted_interval,
                                                                                                 holidays = stock_engine.hols,
                                                                                                 predicted_interval = predicted_interval)

                #Prophet
                pred_prophet, prophet_score , prophet_train_plot = model.prophet_model(  data = stock_engine.stock_data,
                                                                                   yyyy = stock_engine.yyyy,
                                                                                   mm = stock_engine.mm,
                                                                                   H = 22 * predicted_interval,
                                                                                   holidays = stock_engine.hols,
                                                                                   predicted_interval = predicted_interval
                                                                                   )    

                com =  [('Holt',round( holt_score,2)),
                        ('Holt Winter',round(holt_winter_score,2)),
                        ('ARIMA',round(arima_score,2)),
                        ('SARIMA',round(sarima_score,2)),
                        ('Prophet',round(prophet_score,2))]
                compare = pd.DataFrame(com,columns=["model","score"])
                compare.sort_values('score',inplace=True)
                compare.reset_index(drop = True,inplace=True)
                if stock_number == 'USD/TWD':
                    stock_number = "USD:TWD"
                    product = "USD:TWD"
                compare.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
                print("""最佳預測模型：{}""".format(compare.iloc[0,0]))
                if compare.iloc[0,0] == 'Holt':
                    pred_holt,holt_train_plot = stock_engine.transform_df(pred_holt,holt_train_plot,stock_number)
                    holt_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_holt.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'Holt Winter':
                    pred_holt_winter,holt_winter_plot = stock_engine.transform_df(pred_holt_winter,,holt_winter_plot,stock_number)
                    holt_winter_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_holt_winter.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'ARIMA':
                    pred_arima,arima_train_plot = stock_engine.transform_df(pred_arima,arima_train_plot,stock_number)
                    arima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_arima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'SARIMA': 
                    pred_sarima,sarima_train_plot = stock_engine.transform_df(pred_sarima,sarima_train_plot,stock_number)
                    sarima_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_sarima.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                elif compare.iloc[0,0] == 'Prophet': 
                    pred_prophet,prophet_train_plot = stock_engine.transform_df(pred_prophet,prophet_train_plot,stock_number)
                    prophet_train_plot.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                    pred_prophet.to_excel('/Users/jennings.chan/Desktop/FCST App_Test 2.0/{}/6 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,product))