# Importing dependencies
import math 
import numpy as np
import pandas as pd
import datetime
from pandas import Series, DataFrame
## Note: Install pandas_datareader
## pip install pandas-datareader
import pandas_datareader.data as web
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def parseDate(date):
    format_str = "%m/%d/%y" # The format
    datetime_obj = datetime.datetime.strptime(date, format_str).date().isoformat()
    print(datetime_obj)
    return datetime_obj





def forecast(ma1,ma2,ticker,from_date,to_date):
    plt.clf()
  
    # start_date = datetime.datetime(2016, 5, 10)
    if to_date == '0':
        end_date = datetime.datetime.now().date().isoformat()
    else:
        end_date = to_date

    start_date = parseDate(from_date)
    
    # end_date = parseDate(to_date) 

    stocks_df = web.DataReader(ticker, 'yahoo', start_date, end_date)

    closing_price_df= stocks_df['Adj Close']

    closing_price_df.index = pd.to_datetime(closing_price_df.index)

    print(closing_price_df.tail(1))

    ## calc moving averages
    ## temp

    print(closing_price_df)
    ## Calculate 50 day Moving Average
    ma_1 = closing_price_df.rolling(window=ma1).mean()
    ma_1.index = pd.to_datetime(ma_1.index)
    ma_1.dropna(inplace=True)

    ma_2 = closing_price_df.rolling(window=ma2).mean()
    ma_2.index = pd.to_datetime(ma_2.index)
    ma_2.dropna(inplace=True)


    #high low percentage 
    dfreg = stocks_df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (stocks_df['High'] - stocks_df['Low']) / stocks_df['Close'] * 100.0
    #percentage change 
    dfreg['PCT_change'] = (stocks_df['Close'] - stocks_df['Open']) / stocks_df['Open']  * 100.0

    #drop missing value 
    dfreg.fillna(value=99999, inplace = True)
    forecast_out = int(math.ceil(0.06*len(dfreg)))

    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    #linear regression
    #X = preprocessing.scale(X)

    #train for model generation and evaluation 
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    y = np.array(dfreg['label'])
    y = y[:-forecast_out]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)
    #quadratic regression
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    #KNN Regression 
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    # XGBOOST Regression
    clfxgb = RandomForestRegressor(n_estimators=1000, max_depth=7)
    clfxgb.fit(X_train, y_train)

    #evaluation 
    conf_reg = clfreg.score(X_test, y_test)
    confpoly2 = clfpoly2.score(X_test, y_test)
    confpoly3 = clfpoly3.score(X_test, y_test)
    confidenceknn = clfknn.score(X_test, y_test)
    #confxgb = clfxgb.score(X_test, y_test)

    forecast_set = clfxgb.predict(X_lately)
    dfreg['Forecast'] = np.nan

    #plotting prediciton 
    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]


    dfreg['Adj Close'].tail(500).plot(color='black')
    dfreg['Forecast'].tail(500).plot(color='orange',label='Forecast')

    print(dfreg)
    forecastHTML = pd.DataFrame(dfreg['Forecast'].tail(8)).to_html()
    #dfreg.to_csv('dfreg.csv', index = True)
    
    #################################################################
    ### Exraction of some Parameters
    
    dfreg['Date'] = pd.to_datetime(dfreg.index) 
    from datetime import date 
  
    # Returns the current local date 
    today = date.today() 
    dfreg_forecast = dfreg[dfreg['Date'] >= str(today)]
    dfreg_forecast = dfreg_forecast[['Date', 'Forecast']].reset_index(drop = True)

    
    def end_of_week_(day):
        day_of_week = day.weekday()
        to_end_of_week = datetime.timedelta(days=6 - day_of_week)
        end_of_week = day + to_end_of_week

        return end_of_week

    def last_day_of_month(any_day):
        # this will never fail
        # get close to the end of the month for any day, and add 4 days 'over'
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
        # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
        return next_month - datetime.timedelta(days=next_month.day)


    dfreg_forecast['Date'] = pd.to_datetime(dfreg_forecast['Date']) 
    
    # Add this as the "End of Day = 
    end_of_day = np.round(dfreg_forecast.iloc[0, 1],3)
    # Add this as the "End of Week = 
    end_of_week = np.round(dfreg_forecast[dfreg_forecast['Date'] == end_of_week_(today).strftime('%Y-%m-%d')],3)
    # Add this as the "End of Month = 
    end_of_month = np.round(dfreg_forecast[dfreg_forecast['Date'] == last_day_of_month(today).strftime('%Y-%m-%d')],3)

    # Add this as the "End of 7 Days =    
    end_of_seven_days = np.round(dfreg_forecast.iloc[7, 1],3)   
    # Add this as the ""End of 30 Days =
    end_of_thirty_days = np.round(dfreg_forecast.iloc[-1, 1],3) 

    ###### Replace this table with the existing 7 days
    print("++++++++++++++",dfreg_forecast)
    forecastHTML = dfreg_forecast.to_html()
    
    ### Done
    #################################################################    

    def trend():
        forecastDF = dfreg['Forecast'].tail(500).dropna()
        if forecastDF.tail(1).squeeze() > forecastDF.head(1).squeeze(): 
            return 'Bullish'
        elif forecastDF.tail(1).squeeze() < forecastDF.head(1).squeeze():
            return 'Bearish'
        else:
            return 'Neutral'
    forecastedTrend = trend()

    

    plt.title(ticker)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(ma_1, label=f'{ma1} Day SMA', linewidth = 1.5,color='pink')
    plt.plot(ma_2, label=f'{ma2} Day SMA', linewidth = 1.5,color='aqua')
    plt.legend(loc='best')
    # plt.show()

    plt.savefig('public/static/predict.png')
    plt.close()

    results = {
        'trend':forecastedTrend,
        'html':forecastHTML,
        'end_of_day':end_of_day,
        'end_of_week':end_of_week,
        'end_of_month':end_of_month,
        'end_of_seven_days':end_of_seven_days,
        'end_of_thirty_days':end_of_thirty_days,


        
    }

    return results
