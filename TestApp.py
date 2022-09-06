####https://developer.mozilla.org/en-US/docs/Web/CSS/color_value 顏色參考在
#https://discuss.streamlit.io/t/change-background-color-based-on-value/2614 DataFrame顏色code參考
#https://discuss.streamlit.io/t/ta-lib-streamlit-deploy-error/7643/4 Talib部署 超難...
#https://getemoji.com/ For Streamlit emoji大全

import streamlit as st
from PIL import Image
import sys
#sys.path.append("/app/fcst-app_test/Tools")
#from Trend_Analysis3 import Data, Cal_Tool
import pandas as pd
#import yfinance as yf
#import matplotlib.pyplot as plt
import numpy as np
import datetime
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import plotly.express as px 
from dateutil.relativedelta import relativedelta
from xlsxwriter import Workbook
from openpyxl import load_workbook
import investpy
import calendar

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=True, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def color_survived(val):
    color = 'green' if val >= 0 else ('red' if val< 0 else 'white')  
    return f'background-color: {color}'

@st.cache(allow_output_mutation=True)
def load_data(path):
    wb = load_workbook(filename=path,read_only=False ,data_only=True, keep_vba=True)
    ws = wb.active
    ws = wb['Sheet1']
    df = pd.DataFrame(ws.values)#.iloc[:,1:]
    df.set_index([0],inplace=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:,:]
    df.index.name = None
    df.rename_axis(None, axis=1, inplace=True)
    return df
##############################################STREAMLIT####################################################

###################Page Title##################

image = Image.open('./peloton-logo.png')

st.image(image, use_column_width=True)
st.markdown('***Designed by Jennings Chan *** \f\f\f 📩\f *** jennings.chan@onepeloton.com***')
st.header(
     """
     🔍🔍🔍Future Price Forecast Analysis🔎🔎🔎

     """
) #***秀出灰色水平線

st.markdown("""**Please choose one material and press "Confirm" button**""")
#st.markdown("""**因程式需要時間建構模型及_Prediction，請約於30分鐘後回來觀看。**""")
###################Side Bar##################

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header('Please choose material and predicted month')
st.sidebar.header('[Reference]\nTaiwan Paper : Index of Paper in Taiwan\n\nTaiwan Plastic :Index of Plastic in Taiwan\n\nTaiwan Steel :Index of Steel in Taiwan\n\nUK Aluminum : Aluminum Future in UK Market\n\nCN Aluminum : Aluminum Future in CN Market\n\nUSD/TWD Foregin Exchange between US & TW')
#selected_stock_symbol = st.sidebar.
price_unit = st.sidebar.selectbox('Price Unit',['CNY','USD','NTD'])
if price_unit == 'CNY':
   symbols = ['CN Aluminum']
   stock_number = st.sidebar.selectbox('Stock Symbol',symbols)
elif price_unit == 'USD':
   symbols = ['Taiwan Paper','Taiwan Plastic','Taiwan Steel','UK Aluminum','CN Aluminum']
   stock_number = st.sidebar.selectbox('Stock Symbol',symbols)
else : 
   symbols = ['Taiwan Paper','Taiwan Plastic','Taiwan Steel','UK Aluminum','USD/TWD']
   stock_number = st.sidebar.selectbox('Stock Symbol',symbols)    
yyyy = st.sidebar.selectbox('Year',list(range(2022,datetime.datetime.today().year+1)))
if yyyy == datetime.datetime.today().year:
   mm = st.sidebar.selectbox('Predicted Month',list(range(8,datetime.datetime.today().month+1)))
else:
   mm = st.sidebar.selectbox('Predicted Month',list(reversed(range(1,13))))
#if yyyy == datetime.datetime.today().year and mm > datetime.datetime.today().month:

predicted_interval = st.sidebar.selectbox('Predicted Interval',[1,3,6]) #20220104
if stock_number == 'Taiwan Paper':
    product = 'Paper'
elif stock_number == 'UK Aluminum':
    product = 'UK Aluminum'
elif stock_number == 'CN Aluminum':
    product = 'CN Aluminum'
elif stock_number == 'Taiwan Plastic':
    product = 'Plastic'
elif stock_number == 'Taiwan Steel':
    product = 'Steel'
elif stock_number == 'USD/TWD':
    product = 'US Dollar'

if st.sidebar.button('Confirm'):
    ###################Input##################

    #stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
    #stock_engine.get_stock_data(if_lstm=False)
    if stock_number == 'UK Aluminum':
        price_data = investpy.get_commodity_historical_data(commodity= 'Aluminum',
                                                            country = "united kingdom", 
                                                            from_date='01/01/2010', 
                                                            to_date=datetime.datetime(yyyy, mm, calendar.monthrange(yyyy, mm)[1]).strftime('%d/%m/%Y')
                                                            )
        market = 'USD'
    elif stock_number == 'CN Aluminum':
        search_results = investpy.search_quotes(text='SAFc1', products=['commodities'], countries=['china'])
        search_result = search_results.pop(0)
        price_data = search_result.retrieve_historical_data(from_date='01/01/2010',
                                                            to_date=datetime.datetime(yyyy, mm, calendar.monthrange(yyyy, mm)[1]).strftime('%d/%m/%Y')
                                                            )
        market = 'CNY'
    elif stock_number in ('Taiwan Paper','Taiwan Steel','Taiwan Plastic'):
        price_data = investpy.indices.get_index_historical_data(index = stock_number, 
                                        country = 'Taiwan', 
                                        from_date = '01/01/2010', 
                                        to_date = datetime.datetime(yyyy, mm, calendar.monthrange(yyyy, mm)[1]).strftime('%d/%m/%Y')
                                        )
        market = 'NTD'
    else:
        price_data = investpy.currency_crosses.get_currency_cross_historical_data(currency_cross = 'USD/TWD', 
                                                    from_date = '01/01/2010', 
                                                    to_date = datetime.datetime(yyyy, mm, calendar.monthrange(yyyy, mm)[1]).strftime('%d/%m/%Y')
                                                )
        market = 'NTD'
    st.write(
        """
        ## Recent Close Price - {}
        """.format(product)
    )        

    st.metric(label="Material Price {}".format(price_data.index[-1].strftime('%Y-%m-%d')),value = "{} {}".format(market,price_data['Open'][-1]), delta=round(price_data['Open'][-2]-price_data['Open'][-1],2))
    
    st.write(
        """
        ## Historical Close Price - {}
        """.format(product)
    )
    st.write('**Average Historical Price : {}**'.format(round(np.mean(price_data['Open']),2)))

    st.line_chart(price_data.Open)
    if stock_number == 'USD/TWD':
        stock_number = "USD:TWD" 
    if predicted_interval == 1:
        compare = load_data('./{}/1 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
        if compare.iloc[0,0]== 'LightGBM' and round(compare.loc[compare.model=='LightGBM','acc_for_gbm'].values[0],2) < 0.6: #更新GBM規則 20220301
            st.write('***')
            st.write("""Best Model：{}""".format(compare.iloc[1,0]))
            if compare.iloc[1,0] == 'lstm':    
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='lstm','score'].values[0],2)))
                #st.pyplot(lstm_train_plot)
                lstm_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                lstm_train_plot = lstm_train_plot.loc[:,lstm_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(lstm_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                lstm_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                lstm_df['{} Price Change'.format(price_unit)] = lstm_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    lstm_df.set_index('Date',inplace=True)
                    lstm_df.index = lstm_df.index.strftime('%Y-%-m-%-d') 
                except:
                    pass    
                lstm_df = lstm_df.loc[:,lstm_df.columns.str.contains(price_unit)]
                st.dataframe(lstm_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(lstm_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[1,0] == 'GRU':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='GRU','score'].values[0],2)))
                gru_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gru_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                gru_train_plot = gru_train_plot.loc[:,gru_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(gru_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                gru_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_gru.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                gru_df['{} Price Change'.format(price_unit)] = gru_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    gru_df.set_index('Date',inplace=True)
                    gru_df.index = gru_df.index.strftime('%Y-%-m-%-d')     
                except:
                    pass
                gru_df = gru_df.loc[:,gru_df.columns.str.contains(price_unit)]
                st.dataframe(gru_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(gru_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[1,0] == 'Holt':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt','score'].values[0],2)))
                holt_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                holt_train_plot = holt_train_plot.loc[:,holt_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(holt_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                holt_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                holt_df['{} Price Change'.format(price_unit)] = holt_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    holt_df.set_index('Date',inplace=True)
                    holt_df.index = holt_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                holt_df = holt_df.loc[:,holt_df.columns.str.contains(price_unit)]
                st.dataframe(holt_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(holt_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[1,0] == 'Holt Winter':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt Winter','score'].values[0],2)))
                holt_winter_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                holt_winter_train_plot = holt_winter_train_plot.loc[:,holt_winter_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(holt_winter_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                holt_winter_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                holt_winter_df['{} Price Change'.format(price_unit)] = holt_winter_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    holt_winter_df.set_index('Date',inplace=True)
                    holt_winter_df.index = holt_winter_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                holt_winter_df = holt_winter_df.loc[:,holt_winter_df.columns.str.contains(price_unit)]
                st.dataframe(holt_winter_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(holt_winter_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[1,0] == 'ARIMA':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
                arima_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                arima_train_plot = arima_train_plot.loc[:,arima_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(arima_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                arima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                arima_df['{} Price Change'.format(price_unit)] = arima_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    arima_df.set_index('Date',inplace=True)
                    arima_df.index = arima_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                arima_df = arima_df.loc[:,arima_df.columns.str.contains(price_unit)]
                st.dataframe(arima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(arima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))      
            elif compare.iloc[1,0] == 'SARIMA':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
                sarima_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                sarima_train_plot = sarima_train_plot.loc[:,sarima_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(sarima_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                sarima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                sarima_df['{} Price Change'.format(price_unit)] = sarima_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    sarima_df.set_index('Date',inplace=True)
                    sarima_df.index = sarima_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                sarima_df = sarima_df.loc[:,sarima_df.columns.str.contains(price_unit)]
                st.dataframe(sarima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(sarima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))   
            elif  compare.iloc[1,0] == 'Prophet':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Prophet','score'].values[0],2)))
                prophet_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                prophet_train_plot = prophet_train_plot.loc[:,prophet_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(prophet_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                prophet_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                prophet_df['{} Price Change'.format(price_unit)] = prophet_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    prophet_df.set_index('Date',inplace=True)
                    prophet_df.index = prophet_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                prophet_df = prophet_df.loc[:,prophet_df.columns.str.contains(price_unit)]
                st.dataframe(prophet_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(prophet_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))                                          
        else:
            st.write('***')
            st.write('Best Model：{}'.format(compare.iloc[0,0]))
            if compare.iloc[0,0] == 'lstm':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='lstm','score'].values[0],2)))
                #st.pyplot(lstm_train_plot)
                lstm_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                lstm_train_plot = lstm_train_plot.loc[:,lstm_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(lstm_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                lstm_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                lstm_df['{} Price Change'.format(price_unit)] = lstm_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    lstm_df.set_index('Date',inplace=True)
                    lstm_df.index = lstm_df.index.strftime('%Y-%-m-%-d')     
                except:
                    pass
                lstm_df = lstm_df.loc[:,lstm_df.columns.str.contains(price_unit)]
                st.dataframe(lstm_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(lstm_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[0,0] == 'GRU':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='GRU','score'].values[0],2)))
                gru_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gru_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                gru_train_plot = gru_train_plot.loc[:,gru_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(gru_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                gru_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_gru.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                gru_df['{} Price Change'.format(price_unit)] = gru_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    gru_df.set_index('Date',inplace=True)
                    gru_df.index = gru_df.index.strftime('%Y-%-m-%-d')      
                except:
                    pass         
                gru_df = gru_df.loc[:,gru_df.columns.str.contains(price_unit)]
                st.dataframe(gru_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(gru_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[0,0] == 'Holt':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt','score'].values[0],2)))
                holt_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                holt_train_plot = holt_train_plot.loc[:,holt_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(holt_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                holt_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                holt_df['{} Price Change'.format(price_unit)] = holt_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    holt_df.set_index('Date',inplace=True)
                    holt_df.index = holt_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                holt_df = holt_df.loc[:,holt_df.columns.str.contains(price_unit)]
                st.dataframe(holt_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(holt_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[0,0] == 'Holt Winter':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt Winter','score'].values[0],2)))
                holt_winter_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                holt_winter_train_plot = holt_winter_train_plot.loc[:,holt_winter_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(holt_winter_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                holt_winter_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                holt_winter_df['{} Price Change'.format(price_unit)] = holt_winter_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    holt_winter_df.set_index('Date',inplace=True)
                    holt_winter_df.index = holt_winter_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                holt_winter_df = holt_winter_df.loc[:,holt_winter_df.columns.str.contains(price_unit)]
                st.dataframe(holt_winter_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(holt_winter_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
            elif compare.iloc[0,0] == 'ARIMA':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
                arima_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                arima_train_plot = arima_train_plot.loc[:,arima_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(arima_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                arima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                arima_df['{} Price Change'.format(price_unit)] = arima_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    arima_df.set_index('Date',inplace=True)
                    arima_df.index = arima_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                arima_df = arima_df.loc[:,arima_df.columns.str.contains(price_unit)]
                st.dataframe(arima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(arima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))      
            elif compare.iloc[0,0] == 'SARIMA':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
                sarima_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                sarima_train_plot = sarima_train_plot.loc[:,sarima_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(sarima_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                sarima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                sarima_df['{} Price Change'.format(price_unit)] = sarima_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    sarima_df.set_index('Date',inplace=True)
                    sarima_df.index = sarima_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                sarima_df = sarima_df.loc[:,sarima_df.columns.str.contains(price_unit)]
                st.dataframe(sarima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(sarima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))   
            elif  compare.iloc[0,0] == 'Prophet':
                st.write('***')
                st.header('**Model Performance**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Prophet','score'].values[0],2)))
                prophet_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                prophet_train_plot = prophet_train_plot.loc[:,prophet_train_plot.columns.str.contains(price_unit)]
                fig1 = px.line(prophet_train_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                prophet_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
                st.header('Prediction List of {}'.format(calendar.month_name[mm]))
                prophet_df['{} Price Change'.format(price_unit)] = prophet_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    prophet_df.set_index('Date',inplace=True)
                    prophet_df.index = prophet_df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                prophet_df = prophet_df.loc[:,prophet_df.columns.str.contains(price_unit)]
                st.dataframe(prophet_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(prophet_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product)) 
            elif compare.iloc[0,0] == 'LightGBM':
                st.write('***')
                st.header('**MODEL PERFORMANCE**')
                st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
                st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='LightGBM','score'].values[0],2)))
                st.write('Accuracy:{}'.format(round(compare.loc[compare.model=='LightGBM','acc_for_gbm'].values[0],2)))
                st.write('**Prediction - T-1 Month**')
                #st.pyplot(gbm_test_plot)
                gbm_test_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gbm_test_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                gbm_test_plot = gbm_test_plot.loc[:,gbm_test_plot.columns.str.contains(price_unit)]
                #gbm_test_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_gbm_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(gbm_test_plot,color_discrete_map={
                                "{} FCST".format(price_unit): "#0000cd",
                                "{} Actual".format(price_unit): "#008080"
                            })
                st.write(fig1)
                st.write('**Price Changing Plot**')
                #st.pyplot(gbm_change_plot)
                gbm_change_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gbm_change_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                gbm_change_plot = gbm_change_plot.loc[:,gbm_change_plot.columns.str.contains(price_unit)]
                #gbm_change_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_gbm_change_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig2 = px.line(gbm_change_plot,color_discrete_map={
                                "{} Predicted Stock Price Changing".format(price_unit): "#0000cd",
                                "{} Real Stock Price Changing".format(price_unit): "#008080"
                            })
                st.write(fig2)
                st.write('**Important Features**')
                st.write('In the furturem, users can take the most important feature as reference to observe the material price trend.')
                exp =  [('pre_open','Open pric of last 30 days ago'),
                ('close-open','Price between Close and Open'),
                ('high-low','Price between Highest and Lowest'),
                ('price_change','Difference of Close price between T and T-1 day'),
                ('p_change','Percentage of Close price in T minus T-1 day'),
                ('MA5','Moving average of 5 days'),
                ('MA10','Moving average of 10 days'),
                ('MA20','Moving average of 20 days'),
                ('RSI6','Relative Strength Index of 6 days, value and buy volume have positive correlation'),
                ('RSI12','Relative Strength Index of 12 days, value and buy volume have positive correlation'),
                ('RSI24','Relative Strength Index of 24 days, value and buy volume have positive correlation'),
                ('KAMA','Kaufman\'s Adaptive Moving Average,account for market noise or volatility, the higher the better'),
                ('upper','Upper rail of bollinger channel,possible pressure line of stock price within 95% confident interval'),
                ('middle','Middle rail of bollinger channel,is also the moving average line'),
                ('lower','Lower rail of bollinger channel,possible pressure line of stock price within 95% confident interval'),
                ('MOM','Momentum, is mainly used to observe the range of changes in price trends and the direction of market trends'),
                ('EMA12','Exponential moving average of 12 days for predicting the trend of the future stock price'),
                ('EMA26','Exponential moving average of 26 days for predicting the trend of the future stock price'),
                ('DIFF','MA12 - MA26,the difference between the fast smooth moving average and the slow smooth moving average'),
                ('DEA','The moving average of the DIF, which is the arithmetic average of the DIFF for several consecutive days'),
                ('MACD','The exponentially smoothed moving average of similarities and differences to determine the band increase and find buying and selling points')]
                explain = pd.DataFrame(exp,columns=["指數名稱","Definition"])
                explain.set_index('指數名稱',inplace = True)
                st.dataframe(explain)
                #st.pyplot(gbm_importance_plot.figure)
                gbm_importance_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gbm_importance_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
                #gbm_importance_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_gbm_importance_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                gbm_importance_plot.sort_values('importance',ascending=False,inplace=True)
                gbm_importance_plot = gbm_importance_plot[:15].sort_values('importance')
                fig=px.bar(gbm_importance_plot,x=gbm_importance_plot['importance'],y=gbm_importance_plot.index, orientation='h')
                st.write(fig)
                gbm_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_light.xlsx'.format(stock_number,yyyy,mm,stock_number))
                #gbm_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_light_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('prediction list of {}'.format(calendar.month_name[mm]))
                #st.dataframe(gbm_df)
                gbm_df['{} Price Change'.format(price_unit)] = gbm_df['{} Price Change'.format(price_unit)].astype(float)
                try:
                    gbm__df.set_index('Date',inplace=True)
                    gbm__df.index = gbm__df.index.strftime('%Y-%-m-%-d')
                except:
                    pass
                gbm_df = gbm_df.loc[:,gbm_df.columns.str.contains(price_unit)]
                st.dataframe(gbm_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
                df_xlsx = to_excel(gbm_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))

    elif predicted_interval == 3:
        compare = load_data('./{}/3 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
        #compare = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/comparison_{}{}.xlsx'.format(stock_number,yyyy,mm),index_col=0)
        st.write('***')
        st.write("""Best Model：{}""".format(compare.iloc[0,0]))
        if compare.iloc[0,0] == 'Holt':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt','score'].values[0],2)))
            holt_train_plot = load_data('./{}/3 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            holt_train_plot = holt_train_plot.loc[:,holt_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(holt_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            holt_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            holt_df['{} Price Change'.format(price_unit)] = holt_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                holt_df.set_index('Date',inplace=True)
                holt_df.index = holt_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            holt_df = holt_df.loc[:,holt_df.columns.str.contains(price_unit)]
            st.dataframe(holt_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(holt_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
        elif compare.iloc[0,0] == 'Holt Winter':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt Winter','score'].values[0],2)))
            holt_winter_train_plot = load_data('./{}/3 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            holt_winter_train_plot = holt_winter_train_plot.loc[:,holt_winter_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(holt_winter_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            holt_winter_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            holt_winter_df['{} Price Change'.format(price_unit)] = holt_winter_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                holt_winter_df.set_index('Date',inplace=True)
                holt_winter_df.index = holt_winter_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            holt_winter_df = holt_winter_df.loc[:,holt_winter_df.columns.str.contains(price_unit)]
            st.dataframe(holt_winter_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(holt_winter_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
        elif compare.iloc[0,0] == 'ARIMA':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
            arima_train_plot = load_data('./{}/3 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            arima_train_plot = arima_train_plot.loc[:,arima_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(arima_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            arima_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            arima_df['{} Price Change'.format(price_unit)] = arima_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                arima_df.set_index('Date',inplace=True)
                arima_df.index = arima_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            arima_df = arima_df.loc[:,arima_df.columns.str.contains(price_unit)]
            st.dataframe(arima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(arima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))      
        elif compare.iloc[0,0] == 'SARIMA':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
            sarima_train_plot = load_data('./{}/3 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            sarima_train_plot = sarima_train_plot.loc[:,sarima_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(sarima_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            sarima_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            sarima_df['{} Price Change'.format(price_unit)] = sarima_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                sarima_df.set_index('Date',inplace=True)
                sarima_df.index = sarima_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            sarima_df = sarima_df.loc[:,sarima_df.columns.str.contains(price_unit)]
            st.dataframe(sarima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(sarima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))   
        elif  compare.iloc[0,0] == 'Prophet':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Prophet','score'].values[0],2)))
            prophet_train_plot = load_data('./{}/3 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            prophet_train_plot = prophet_train_plot.loc[:,prophet_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(prophet_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            prophet_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            prophet_df['{} Price Change'.format(price_unit)] = prophet_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                prophet_df.set_index('Date',inplace=True)
                prophet_df.index = prophet_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            prophet_df = prophet_df.loc[:,prophet_df.columns.str.contains(price_unit)]
            st.dataframe(prophet_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(prophet_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product)) 

    elif predicted_interval == 6:
        compare = load_data('./{}/6 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
        st.dataframe(compare)
        st.write('***')
        st.write("""Best Model：{}""".format(compare.iloc[0,0]))
        if compare.iloc[0,0] == 'Holt':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt','score'].values[0],2)))
            holt_train_plot = load_data('./{}/6 MONTH/{}{}/{}_for_holt_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            holt_train_plot = holt_train_plot.loc[:,holt_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(holt_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            holt_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_holt.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            holt_df['{} Price Change'.format(price_unit)] = holt_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                holt_df.set_index('Date',inplace=True)
                holt_df.index = holt_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            holt_df = holt_df.loc[:,holt_df.columns.str.contains(price_unit)]
            st.dataframe(holt_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(holt_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
        elif compare.iloc[0,0] == 'Holt Winter':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Holt Winter','score'].values[0],2)))
            holt_winter_train_plot = load_data('./{}/6 MONTH/{}{}/{}_for_holt_winter_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            holt_winter_train_plot = holt_winter_train_plot.loc[:,holt_winter_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(holt_winter_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            holt_winter_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_holt_winter.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            holt_winter_df['{} Price Change'.format(price_unit)] = holt_winter_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                holt_winter_df.set_index('Date',inplace=True)
                holt_winter_df.index = holt_winter_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            holt_winter_df = holt_winter_df.loc[:,holt_winter_df.columns.str.contains(price_unit)]
            st.dataframe(holt_winter_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(holt_winter_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))
        elif compare.iloc[0,0] == 'ARIMA':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
            arima_train_plot = load_data('./{}/6 MONTH/{}{}/{}_for_arima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            arima_train_plot = arima_train_plot.loc[:,arima_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(arima_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            arima_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            arima_df['{} Price Change'.format(price_unit)] = arima_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                arima_df.set_index('Date',inplace=True)
                arima_df.index = arima_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            arima_df = arima_df.loc[:,arima_df.columns.str.contains(price_unit)]
            st.dataframe(arima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(arima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))      
        elif compare.iloc[0,0] == 'SARIMA':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
            sarima_train_plot = load_data('./{}/6 MONTH/{}{}/{}_for_sarima_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            sarima_train_plot = sarima_train_plot.loc[:,sarima_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(sarima_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            sarima_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            sarima_df['{} Price Change'.format(price_unit)] = sarima_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                sarima_df.set_index('Date',inplace=True)
                sarima_df.index = sarima_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            sarima_df = sarima_df.loc[:,sarima_df.columns.str.contains(price_unit)]
            st.dataframe(sarima_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(sarima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product))   
        elif  compare.iloc[0,0] == 'Prophet':
            st.write('***')
            st.header('**Model Performance**')
            st.write('**In order to predict the tarend, the model performance will be measured by mape, the lower mape, the better model performance.**')
            st.write('Average MAPE:{}% (Average change rate between actual value & predicted value)'.format(round(compare.loc[compare.model=='Prophet','score'].values[0],2)))
            prophet_train_plot = load_data('./{}/6 MONTH/{}{}/{}_for_prophet_train_plot.xlsx'.format(stock_number,yyyy,mm,stock_number))
            prophet_train_plot = prophet_train_plot.loc[:,prophet_train_plot.columns.str.contains(price_unit)]
            fig1 = px.line(prophet_train_plot,color_discrete_map={
                            "{} FCST".format(price_unit): "#0000cd",
                            "{} Actual".format(price_unit): "#008080"
                        })
            st.write(fig1)
            prophet_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_prophet.xlsx'.format(stock_number,yyyy,mm,stock_number)) 
            st.header('Prediction List of {}'.format(calendar.month_name[mm]))
            prophet_df['{} Price Change'.format(price_unit)] = prophet_df['{} Price Change'.format(price_unit)].astype(float)
            try:
                prophet_df.set_index('Date',inplace=True)
                prophet_df.index = prophet_df.index.strftime('%Y-%-m-%-d')
            except:
                pass
            prophet_df = prophet_df.loc[:,prophet_df.columns.str.contains(price_unit)]
            st.dataframe(prophet_df.style.applymap(color_survived, subset=['{} Price Change'.format(price_unit)]))
            df_xlsx = to_excel(prophet_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}_{}_Prediction.xlsx'.format(calendar.month_name[mm],product)) 
