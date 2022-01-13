####https://developer.mozilla.org/en-US/docs/Web/CSS/color_value 顏色參考在
#https://discuss.streamlit.io/t/change-background-color-based-on-value/2614 DataFrame顏色code參考
#https://discuss.streamlit.io/t/ta-lib-streamlit-deploy-error/7643/4 Talib部署 超難...
#https://getemoji.com/ For Streamlit emoji大全

import streamlit as st
from PIL import Image
import sys
sys.path.append("./Tools")
#from Trend_Analysis2 import HiddenPrints, Data, cal_Tool, visual, Model
from Trend_Analysis2 import Data, cal_Tool
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import plotly.express as px 
from dateutil.relativedelta import relativedelta
from xlsxwriter import Workbook

#%matplotlib inline

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
    df = pd.read_excel(path, index_col=0)
    return df
##############################################STREAMLIT####################################################

###################Page Title##################

image = Image.open('/Users/jennings.chan/Desktop/peloton-logo.png')

st.image(image, use_column_width=True)
st.markdown('***Designed by Jennings Chan *** \f\f\f 📩\f *** jennings.chan@onepeloton.com***')
st.header(
     """
     🔍🔍🔍期貨價格預測分析🔎🔎🔎

     """
) #***秀出灰色水平線

st.markdown("""**你/妳可拉選想要觀看的原物料，並按下確認觀測模型預測結果。**""")
#st.markdown("""**因程式需要時間建構模型及預測，請約於30分鐘後回來觀看。**""")
###################Side Bar##################

st.sidebar.header('請選擇原物料及預測年月份')
st.sidebar.header('[參考原物料代號]\n^TPAI 造紙類指數\n\n^TPLI 塑膠類指數\n\n2002.TW 中鋼\n\n601600.SS 中國鋁業(上海證交所)\n\nUSDTWD=X 美元匯率')
#selected_stock_symbol = st.sidebar.
symbols = ['^TPAI','^TPLI','2002.TW','601600.SS','USDTWD=X']
stock_number = st.sidebar.selectbox('Stock Symbol',symbols)
yyyy = st.sidebar.selectbox('Year',list(range(2022,datetime.datetime.today().year+1)))
if yyyy == datetime.datetime.today().year:
   mm = st.sidebar.selectbox('Predicted Month',list(range(1,datetime.datetime.today().month+1)))
else:
   mm = st.sidebar.selectbox('Predicted Month',list(reversed(range(1,13))))
#if yyyy == datetime.datetime.today().year and mm > datetime.datetime.today().month:
#   st.sidebar.write('***月份請重選***')

predicted_interval = st.sidebar.selectbox('Predicted Interval',[1,3,6]) #20220104
if stock_number == '^TPAI':
    product = 'Paper'
elif stock_number == '^TPLI':
    product = 'Plastic'
elif stock_number == '2002.TW':
    product = 'Iron'
elif stock_number == '601600.SS':
    product = 'Aluminum'
elif stock_number == 'USDTWD=X':
    product = 'US Dollar'

if st.sidebar.button('Confirm'):
    ###################Input##################
    stock_engine = Data(stock_number = stock_number,yyyy=yyyy,mm=mm)
    stock_engine.get_stock_data(if_lstm=False)
    st.write(
        """
        ## 歷史開盤價 - {}
        """.format(product)
    )
    st.write('**歷史平均股價為{}**'.format(round(np.mean(stock_engine.stock_data['Open']),2)))
    st.write("原物料如為鋁，則價格單位為CNY")

    st.line_chart(stock_engine.stock_data.Open)

    if predicted_interval == 1:
        compare = load_data('./{}/1 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
        if compare.iloc[0,0]== 'LightGBM' and compare.loc[compare.model=='LightGBM','acc_for_gbm'].values[0] <= 0.6:
            st.write('***')
            st.write("""最佳預測模型：{}""".format(compare.iloc[1,0]))
            if compare.iloc[1,0] == 'lstm':    
                lstm_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,product)) 
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                st.dataframe(lstm_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(lstm_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='lstm','score'].values[0],2)))
                st.write('**訓練集訓練概況**')
                #st.pyplot(lstm_train_plot)
                lstm_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                fig1 = px.line(lstm_train_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                st.write('**測試集預測概況**')
                #st.pyplot(lstm_test_plot)
                lstm_test_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                fig2 = px.line(lstm_test_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig2)
                st.write('**真實預測一個月概況**')
                #st.pyplot(lstm_real_plot)
                lstm_real_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_real_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #lstm_real_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_lstm_real_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig3 = px.line(lstm_real_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig3)
            elif compare.iloc[1,0] == 'WMA':
                wma_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
                #wma_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_wma_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                st.dataframe(wma_df.style.applymap(color_survived, subset=['漲跌']))
                #st.dataframe(wma_df)
                df_xlsx = to_excel(wma_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='WMA','score'].values[0],2)))
                st.write('**訓練集訓練概況**')
                ma_train = load_data('./{}/1 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #ma_train = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_train_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(ma_train,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                #st.pyplot(ma_train)
                st.write('**測試集預測概況**')
                #st.pyplot(ma_test)
                ma_test = load_data('./{}/1 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #ma_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig2 = px.line(ma_test,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig2)
            elif compare.iloc[1,0] == 'ARIMA':
                arima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                #arima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_arima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                #st.dataframe(arima_df)
                st.dataframe(arima_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(arima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
                #st.write('**趨勢圖**')
                #st.pyplot(arima_trend)
                st.write('**測試集預測概況**')
                #st.pyplot(arima_test)
                arima_test = load_data('./{}/1 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #arima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_arima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(arima_test,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                #st.write('***')
                #st.header('備註(參考用)'.format(mm))
                #st.write('**ETS圖表**')
                #st.pyplot(ets_a)
                #st.pyplot(ets_b)
                #st.pyplot(ets_c)
                #st.pyplot(ets_d)
                #st.write('**ACF圖表**') 
                #st.pyplot(acf)
            elif compare.iloc[1,0] == 'SARIMA':
                sarima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                #sarima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_sarima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                #st.dataframe(sarima_df)
                st.dataframe(sarima_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(sarima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
                st.write('**測試集預測概況**')
                #st.pyplot(sarima_test)
                sarima_test = load_data('./{}/1 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #sarima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_sarima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig = px.line(sarima_test,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig)
        else:
            st.write('***')
            st.write('最佳預測模型：{}'.format(compare.iloc[0,0]))
            if compare.iloc[0,0] == 'lstm':
                lstm_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_lstm.xlsx'.format(stock_number,yyyy,mm,product)) 
                #lstm_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_lstm_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                st.dataframe(lstm_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(lstm_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='lstm','score'].values[0],2)))
                st.write('**訓練集訓練概況**')
                #st.pyplot(lstm_train_plot)
                lstm_train_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #lstm_train_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_lstm_train_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(lstm_train_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                st.write('**測試集預測概況**')
                #st.pyplot(lstm_test_plot)
                lstm_test_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #lstm_test_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_lstm_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig2 = px.line(lstm_test_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig2)
                st.write('**真實預測一個月概況**')
                #st.pyplot(lstm_real_plot)
                lstm_real_plot = load_data('./{}/1 MONTH/{}{}/{}_for_lstm_real_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #lstm_real_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_lstm_real_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig3 = px.line(lstm_real_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig3)
            elif compare.iloc[0,0] == 'WMA':
                wma_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
                #wma_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_wma_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                st.dataframe(wma_df.style.applymap(color_survived, subset=['漲跌']))
                #st.dataframe(wma_df)
                df_xlsx = to_excel(wma_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='WMA','score'].values[0],2)))
                st.write('**訓練集訓練概況**')
                ma_train = load_data('./{}/1 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #ma_train = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_train_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(ma_train,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                #st.pyplot(ma_train)
                st.write('**測試集預測概況**')
                #st.pyplot(ma_test)
                ma_test = load_data('./{}/1 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #ma_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig2 = px.line(ma_test,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig2)
            elif compare.iloc[0,0] == 'ARIMA':
                arima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
                #arima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_arima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                #st.dataframe(arima_df)
                st.dataframe(arima_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(arima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
                #st.write('**趨勢圖**')
                #st.pyplot(arima_trend)
                st.write('**測試集預測概況**')
                #st.pyplot(arima_test)
                arima_test = load_data('./{}/1 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #arima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_arima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(arima_test,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                #st.write('***')
                #st.header('備註(參考用)'.format(mm))
                #st.write('**ETS圖表**')
                #st.pyplot(ets_a)
                #st.pyplot(ets_b)
                #st.pyplot(ets_c)
                #st.pyplot(ets_d)
                #st.write('**ACF圖表**') 
                #st.pyplot(acf)
            elif compare.iloc[0,0] == 'SARIMA':
                sarima_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
                #sarima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_sarima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                #st.dataframe(sarima_df)
                st.dataframe(sarima_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(sarima_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
                st.write('**測試集預測概況**')
                #st.pyplot(sarima_test)
                sarima_test = load_data('./{}/1 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #sarima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_sarima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig = px.line(sarima_test,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig)
            elif compare.iloc[0,0] == 'LightGBM':
                gbm_df = load_data('./{}/1 MONTH/{}{}/{}_Best_by_light.xlsx'.format(stock_number,yyyy,mm,product))
                #gbm_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_light_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                st.header('{}月份預測數值名單'.format(mm))
                st.write('''🔔\f
                    如果遇到國定假日，該平日請直接忽略預測值''')
                #st.dataframe(gbm_df)
                st.dataframe(gbm_df.style.applymap(color_survived, subset=['漲跌']))
                df_xlsx = to_excel(gbm_df)
                st.write('📥')
                st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
                st.write('***')
                st.header('**模型訓練狀況**')
                st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='LightGBM','score'].values[0],2)))
                st.write('漲跌預測準確度為:{}'.format(round(score_gbm_acc,2)))
                st.write('**測試集預測概況**')
                #st.pyplot(gbm_test_plot)
                gbm_test_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gbm_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #gbm_test_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_gbm_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig1 = px.line(gbm_test_plot,color_discrete_map={
                                "Predicted Stock Price": "#0000cd",
                                "Real Stock Price": "#008080"
                            })
                st.write(fig1)
                st.write('**漲跌幅狀況**')
                #st.pyplot(gbm_change_plot)
                gbm_change_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gbm_change_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #gbm_change_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_gbm_change_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                fig2 = px.line(gbm_change_plot,color_discrete_map={
                                "Predicted Stock Price Changing": "#0000cd",
                                "Real Stock Price Changing": "#008080"
                            })
                st.write(fig2)
                st.write('**重要特徵分佈**')
                st.write('未來可以根據影響力較大的指數做該原物料的分析觀測')
                exp =  [('pre_open','前30天之股市開盤價'),
                ('close-open','當日收開盤之價差'),
                ('high-low','當日最高與最低之價差'),
                ('price_change','今日與前日之漲跌'),
                ('p_change','今日與前日之漲跌百分比'),
                ('MA5','5日移動平均線'),
                ('MA10','10日移動平均線'),
                ('MA20','20日移動平均線'),
                ('RSI6','6日相對強弱指標，數值與市場熱度成正比'),
                ('RSI12','12日相對強弱指標，數值與市場熱度成正比'),
                ('RSI24','24日相對強弱指標，數值與市場熱度成正比'),
                ('KAMA','考夫曼自適應移動平均，能根據市場趨勢變化速度自主調節，值越大越好'),
                ('upper','布林帶上線，為推測股價的可能上限，一般信賴區間設置為95%'),
                ('middle','布林帶中線，為股價的移動平均線'),
                ('lower','布林帶下線，為推測股價的可能下限，一般信賴區間設置為95%'),
                ('MOM','動能指標，用來觀察價格走勢的變化幅度，以及行情的趨動方向'),
                ('EMA12','12日指數移動平均線，相較SMA多考量權重分數，用於判斷價格未來走勢的變動趨勢'),
                ('EMA26','26日指數移動平均線，相較SMA多考量權重分數，用於判斷價格未來走勢的變動趨勢'),
                ('DIFF','快線，計算兩個不同時間長短的EMA之間差距，通常是EMA12-EMA26'),
                ('DEA','慢線，以9日DIFF值計算之EMA'),
                ('MACD','指數平滑異同移動平均線，可顯示市場趨勢變化，為快線與慢線之差值')]
                explain = pd.DataFrame(exp,columns=["指數名稱","指數定義"])
                explain.set_index('指數名稱',inplace = True)
                st.dataframe(explain)
                #st.pyplot(gbm_importance_plot.figure)
                gbm_importance_plot = load_data('./{}/1 MONTH/{}{}/{}_for_gbm_importance_plot.xlsx'.format(stock_number,yyyy,mm,product))
                #gbm_importance_plot = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_gbm_importance_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
                gbm_importance_plot.sort_values('importance',ascending=False,inplace=True)
                gbm_importance_plot = gbm_importance_plot[:15].sort_values('importance')
                fig=px.bar(gbm_importance_plot,x=gbm_importance_plot['importance'],y=gbm_importance_plot.index, orientation='h')
                st.write(fig)

    elif predicted_interval == 3:
        compare = load_data('./{}/3 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
        #compare = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/3 MONTH/comparison_{}{}.xlsx'.format(stock_number,yyyy,mm),index_col=0)
        st.write('***')
        st.write("""最佳預測模型：{}""".format(compare.iloc[0,0]))
        if compare.iloc[0,0] == 'WMA':
            wma_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
            #wma_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_wma_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            st.header('{}月份預測數值名單'.format(mm))
            st.write('''🔔\f
                如果遇到國定假日，該平日請直接忽略預測值''')
            st.dataframe(wma_df.style.applymap(color_survived, subset=['漲跌']))
            #st.dataframe(wma_df)
            df_xlsx = to_excel(wma_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
            st.write('***')
            st.header('**模型訓練狀況**')
            st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='WMA','score'].values[0],2)))
            st.write('**訓練集訓練概況**')
            ma_train = load_data('./{}/3 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #ma_train = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_train_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig1 = px.line(ma_train,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig1)
            #st.pyplot(ma_train)
            st.write('**測試集預測概況**')
            #st.pyplot(ma_test)
            ma_test = load_data('./{}/3 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #ma_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig2 = px.line(ma_test,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig2)
        elif compare.iloc[0,0] == 'ARIMA':
            arima_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
            #arima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_arima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            st.header('{}月份預測數值名單'.format(mm))
            st.write('''🔔\f
                如果遇到國定假日，該平日請直接忽略預測值''')
            #st.dataframe(arima_df)
            st.dataframe(arima_df.style.applymap(color_survived, subset=['漲跌']))
            df_xlsx = to_excel(arima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
            st.write('***')
            st.header('**模型訓練狀況**')
            st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
            #st.write('**趨勢圖**')
            #st.pyplot(arima_trend)
            st.write('**測試集預測概況**')
            #st.pyplot(arima_test)
            arima_test = load_data('./{}/3 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #arima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_arima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig1 = px.line(arima_test,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig1)
            #st.write('***')
            #st.header('備註(參考用)'.format(mm))
            #st.write('**ETS圖表**')
            #st.pyplot(ets_a)
            #st.pyplot(ets_b)
            #st.pyplot(ets_c)
            #st.pyplot(ets_d)
            #st.write('**ACF圖表**') 
            #st.pyplot(acf)
        elif compare.iloc[0,0] == 'SARIMA':
            sarima_df = load_data('./{}/3 MONTH/{}{}/{}_Best_by_sarima.xlsx'.format(stock_number,yyyy,mm,product))
            #sarima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_sarima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            st.header('{}月份預測數值名單'.format(mm))
            st.write('''🔔\f
                如果遇到國定假日，該平日請直接忽略預測值''')
            #st.dataframe(sarima_df)
            st.dataframe(sarima_df.style.applymap(color_survived, subset=['漲跌']))
            df_xlsx = to_excel(sarima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
            st.write('***')
            st.header('**模型訓練狀況**')
            st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
            st.write('**測試集預測概況**')
            #st.pyplot(sarima_test)
            sarima_test = load_data('./{}/3 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #sarima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_sarima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig = px.line(sarima_test,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig)

    elif predicted_interval == 6:
        compare = load_data('./{}/6 MONTH/{}{}/comparison.xlsx'.format(stock_number,yyyy,mm))
        #compare = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/6 MONTH/comparison_{}{}.xlsx'.format(stock_number,yyyy,mm),index_col=0)
        st.write('***')
        st.write("""最佳預測模型：{}""".format(compare.iloc[0,0]))
        if compare.iloc[0,0] == 'WMA':
            wma_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_wma.xlsx'.format(stock_number,yyyy,mm,product))
            #wma_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_wma_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            st.header('{}月份預測數值名單'.format(mm))
            st.write('''🔔\f
                如果遇到國定假日，該平日請直接忽略預測值''')
            st.dataframe(wma_df.style.applymap(color_survived, subset=['漲跌']))
            #st.dataframe(wma_df)
            df_xlsx = to_excel(wma_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
            st.write('***')
            st.header('**模型訓練狀況**')
            st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='WMA','score'].values[0],2)))
            st.write('**訓練集訓練概況**')
            ma_train = load_data('./{}/6 MONTH/{}{}/{}_for_wma_train_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #ma_train = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_train_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig1 = px.line(ma_train,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig1)
            #st.pyplot(ma_train)
            st.write('**測試集預測概況**')
            #st.pyplot(ma_test)
            ma_test = load_data('./{}/6 MONTH/{}{}/{}_for_wma_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #ma_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_wma_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig2 = px.line(ma_test,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig2)
        elif compare.iloc[0,0] == 'ARIMA':
            arima_df = load_data('./{}/6 MONTH/{}{}/{}_Best_by_arima.xlsx'.format(stock_number,yyyy,mm,product))
            #arima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_arima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            st.header('{}月份預測數值名單'.format(mm))
            st.write('''🔔\f
                如果遇到國定假日，該平日請直接忽略預測值''')
            #st.dataframe(arima_df)
            st.dataframe(arima_df.style.applymap(color_survived, subset=['漲跌']))
            df_xlsx = to_excel(arima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
            st.write('***')
            st.header('**模型訓練狀況**')
            st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='ARIMA','score'].values[0],2)))
            #st.write('**趨勢圖**')
            #st.pyplot(arima_trend)
            st.write('**測試集預測概況**')
            #st.pyplot(arima_test)
            arima_test = load_data('./{}/6 MONTH/{}{}/{}_for_arima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #arima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_arima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig1 = px.line(arima_test,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig1)
            #st.write('***')
            #st.header('備註(參考用)'.format(mm))
            #st.write('**ETS圖表**')
            #st.pyplot(ets_a)
            #st.pyplot(ets_b)
            #st.pyplot(ets_c)
            #st.pyplot(ets_d)
            #st.write('**ACF圖表**') 
            #st.pyplot(acf)
        elif compare.iloc[0,0] == 'SARIMA':
            sarima_df = load_data('./{}/6 MONTH/{}_Best_by_sarima_{}{}.xlsx'.format(stock_number,product,yyyy,mm))
            #sarima_df = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_Best_by_sarima_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            st.header('{}月份預測數值名單'.format(mm))
            st.write('''🔔\f
                如果遇到國定假日，該平日請直接忽略預測值''')
            #st.dataframe(sarima_df)
            st.dataframe(sarima_df.style.applymap(color_survived, subset=['漲跌']))
            df_xlsx = to_excel(sarima_df)
            st.write('📥')
            st.download_button(label='Download FCST',data=df_xlsx,mime='text/xlsx',file_name= '{}月份_{}預測.xlsx'.format(mm,product))
            st.write('***')
            st.header('**模型訓練狀況**')
            st.write('平均預測震動程度為:{} (真實與預測值平均變動率差異)'.format(round(compare.loc[compare.model=='SARIMA','score'].values[0],2)))
            st.write('**測試集預測概況**')
            #st.pyplot(sarima_test)
            sarima_test = load_data('./{}/6 MONTH/{}{}/{}_for_sarima_test_plot.xlsx'.format(stock_number,yyyy,mm,product))
            #sarima_test = pd.read_excel('/Users/jennings.chan/Desktop/FCST App_Test/{}/1 MONTH/{}_for_sarima_test_plot_{}{}.xlsx'.format(stock_number,product,yyyy,mm),index_col=0)
            fig = px.line(sarima_test,color_discrete_map={
                            "Predicted Stock Price": "#0000cd",
                            "Real Stock Price": "#008080"
                        })
            st.write(fig)
