from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import time

from task_utils import connect_mysql
from task_utils import update_taskstatus
from task_utils import check_taskstatus


def get_asset_value(ticker, date_begin, date_end) :
    df = fdr.DataReader(ticker, date_begin, date_end)

    SQL_INSERT_DATA = """INSERT INTO TB_ASSETVALUE_RAW
                        (DATE_MARKET, TICKER, OPEN_P, HIGH_P, LOW_P, CLOSE_P, VOLUME, DATE_UPDATE)
                        VALUES
                        ('%s', '%s', %f, %f, %f, %f, %f, NOW())"""

    with connect_mysql() as con:
        with con.cursor() as cur:
            for idx, row in zip(df.index, df.values) :
                try :
                    SQL_INSERT = SQL_INSERT_DATA %(idx.strftime('%Y-%m-%d'),
                                               #datetime.date(index).strftime('%Y-%m-%d'),
                                               ticker,
                                               float(row[0]), 
                                               float(row[1]), 
                                               float(row[2]), 
                                               float(row[3]), 
                                               float(row[4])
                                               )
                    cur.execute(SQL_INSERT)
                    con.commit()

                except Exception as e:
                    print (str(e))
    return None
    
def preprocess_asset_value(ticker, date_begin, date_end) :
    SQL_SELECT_DATA = """SELECT DATE_MARKET, OPEN_P, HIGH_P, LOW_P, CLOSE_P, VOLUME
                        FROM TB_ASSETVALUE_RAW
                        WHERE TICKER = %s AND DATE_MARKET >= %s and DATE_MARKET <= %s 
                        ORDER BY DATE_MARKET
                        """
    
    SQL_INSERT_DATA = """INSERT INTO TB_ASSETVALUE_ADJ
                        (DATE_MARKET, TICKER, OPEN_P, HIGH_P, LOW_P, CLOSE_P, VOLUME, DATE_UPDATE)
                        VALUES
                        ('%s', '%s', %f, %f, %f, %f, %f, NOW())"""
    with connect_mysql() as con:
        with con.cursor() as cur:
            cur.execute(SQL_SELECT_DATA, (ticker, date_begin, date_end, ))
            fetch_row = cur.fetchall()

    col_date_market, col_open_p, col_high_p, col_low_p, col_close_p, col_volume = \
        [], [], [], [], [], []
    for row in fetch_row :
        col_date_market.append(str(row[0]))
        col_open_p.append(float(row[1]))
        col_high_p.append(float(row[2]))
        col_low_p.append(float(row[3]))
        col_close_p.append(float(row[4]))
        col_volume.append(float(row[5]))

    df = pd.DataFrame({
            'date_market': col_date_market,
            'open_p': col_open_p,
            'high_p': col_high_p,
            'low_p': col_low_p,
            'close_p': col_close_p,
            'volume': col_volume
    })
    
    # df['Date'] : string -> datetime
    df['date_market'] =  pd.to_datetime(df['date_market'])
    
    # Date가 '2000-01-01' 이전 혹은 datetime.now() 이후인 데이터 필터
    df = df[(df['date_market'] >= pd.to_datetime('2000-01-01')) & (df['date_market'] < datetime.now())]

    # Date가 입력한 날싸 사이로 필터
    df = df[(df['date_market'] >= pd.to_datetime(date_begin)) & (df['date_market'] < pd.to_datetime(date_end))]

    # Date가 Null인 데이터 필터
    df = df[~df['date_market'].isnull()]

    # High 가격보다 Low 가격이 높은 데이터 필터
    df = df[df['low_p'] <= df['high_p']]

    # High, Low, Close 중 하나라도 Null 값이면, 전날 High, Low, Close의 데이터를 복사해서 입력
    df[['open_p', 'high_p', 'low_p', 'close_p']] = df[['open_p', 'high_p', 'low_p', 'close_p']].fillna(method='ffill')

    # df['Date'] : datetime -> string
    df['date_market'] = df['date_market'].dt.strftime('%Y-%m-%d')

    with connect_mysql() as con:
        with con.cursor() as cur:
            for row in df.values :
                try :
                    SQL_INSERT = SQL_INSERT_DATA %(str(row[0]),
                                               #datetime.date(index).strftime('%Y-%m-%d'),
                                               ticker,
                                               float(row[1]), 
                                               float(row[2]), 
                                               float(row[3]), 
                                               float(row[4]),
                                               float(row[5])
                                               )
                    cur.execute(SQL_INSERT)
                    con.commit()

                except Exception as e:
                    print (str(e))
    return None


def build_model(data_set, hyperparameter_set) :
    (ticker,
    input_epoch,
    input_batch_size,
    input_learning_rate,
    input_ratio_train_set,
    input_size_sequence,
    input_size_units_input,
    input_size_units_hidden,
    input_size_units_output 
    ) = hyperparameter_set 


    # Compute Mid Price
    price_date = data_set['Date'].values
    price_high = data_set['High'].values
    price_low = data_set['Low'].values
    price_mid = (price_high + price_low) / 2.

    # Create Window
    size_window = input_size_sequence + 1

    price_windowed = []
    price_windowed_normalized = []
    for index in range(len(price_mid) - size_window):
        price_windowed.append(price_mid[index: index + size_window])

    for window in price_windowed:
        window_normalized = [((float(p) / float(window[0])) - 1) for p in window]
        price_windowed_normalized.append(window_normalized)

    price_windowed_normalized_numpyed = np.array(price_windowed_normalized)

    # split train and test data
    number_division = int(round(price_windowed_normalized_numpyed.shape[0] * input_ratio_train_set))
    price_date[:number_division]
    dataset_train = price_windowed_normalized_numpyed[:number_division, :]
    np.random.shuffle(dataset_train)

    X_train = dataset_train[:, :-1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = dataset_train[:, -1]

    X_test = price_windowed_normalized_numpyed[number_division:, :-1]
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = price_windowed_normalized_numpyed[number_division:, -1]

    date_begin_train = price_date[0]
    date_end_train = price_date[number_division-1]
    date_begin_test = price_date[number_division-1]
    date_end_test = price_date[-1]
    
    # LSTM model
    model = Sequential()
    model.add(LSTM(input_size_units_input,
                    return_sequences = True,
                    input_shape = (input_size_sequence, 1))
                    )
    model.add(LSTM(input_size_units_hidden,
                    return_sequences = False)
                    )
    model.add(Dense(input_size_units_output,
                    activation = 'linear')
                    )
    model.compile(loss='mse',
                  optimizer = Adam(learning_rate = input_learning_rate),
                  metrics = ["accuracy"]
                  )

    history = model.fit(X_train,
                        y_train,
                        validation_data = (X_test, y_test),
                        batch_size = input_batch_size,
                        epochs = input_epoch
                        )
       
    results = model.evaluate(X_test, y_test)
    y_pred  = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    loss    = results[0]
    acc     = results[1] 
     
    date_begin_train = "".join(price_date[0].split("-"))
    date_end_train = "".join(price_date[number_division - 1].split("-"))
    date_begin_test = "".join(price_date[number_division].split("-"))
    date_end_test = "".join(price_date[-1].split("-"))

    return date_begin_test, date_end_test, date_begin_train, date_end_train, acc, loss, mse






if __name__ == "__main__":
    ticker = 'AAPL'
    date_begin = '2013-01-01'
    date_end = '2020-03-01'
    retry_count = 30
    retry_sleep = 5

    if check_taskstatus('GETTING_STOCKDATA1', ticker) == 'NotExecuted' or \
        check_taskstatus('GETTING_STOCKDATA1', ticker) == 'Error' :
        try :
            get_asset_value(ticker, date_begin, date_end)
            update_taskstatus('GETTING_STOCKDATA1', ticker, 'Completed', None)
        except Exception as e :
            update_taskstatus('GETTING_STOCKDATA1', ticker, 'Error', str(e))

    if check_taskstatus('GETTING_STOCKDATA2', ticker) ==  'NotExecuted' or \
        check_taskstatus('GETTING_STOCKDATA2', ticker) == 'Error' :
        try :
            preprocess_asset_value(ticker, date_begin, date_end)
            update_taskstatus('GETTING_STOCKDATA2', ticker, 'Completed', None)
        except Exception as e :
            update_taskstatus('GETTING_STOCKDATA2', ticker, 'Error', None)


 
    SQL_SELECT_PARAMETER = """SELECT H.CODE_TASK, H.EPOCH, H.BATCH_SIZE, H.LEARNING_RATE, 
                              H.RATIO_TRAIN_SET, H.SIZE_SEQUENCE, H.SIZE_UNITS_INPUT, H.SIZE_UNITS_HIDDEN, H.SIZE_UNITS_OUTPUT
                              FROM TB_HYPERPARAMETER H LEFT JOIN TB_MODELRESULT M ON H.CODE_TASK = M.CODE_TASK
                              WHERE M.CODE_TASK IS NULL ORDER BY H.CODE_TASK ASC
                              """
    
    SQL_SELECT_ASSETVALUE = """SELECT DATE_MARKET, OPEN_P, HIGH_P, LOW_P, CLOSE_P, VOLUME
                              FROM TB_ASSETVALUE_ADJ
                              WHERE TICKER = %s AND DATE_MARKET >= %s and DATE_MARKET <= %s 
                              ORDER BY DATE_MARKET
                              """

    SQL_INSERT_RESULT = """INSERT INTO TB_MODELRESULT
                             (CODE_TASK, TICKER, VERSION_MODEL, DATE_B_TRAIN, DATE_E_TRAIN, DATE_B_TEST, DATE_E_TEST,
                             LOSS, ACCURACY, MSE, DATE_UPDATE)
                             VALUES
                             ('%s', '%s', %s, %s, %s, %s, %s, %f, %f, %f, NOW())
                             """
   
    with connect_mysql() as con:
        with con.cursor() as cur:
            cur.execute(SQL_SELECT_PARAMETER, )
            hyperparameter_row = cur.fetchall()

            cur.execute(SQL_SELECT_ASSETVALUE, (ticker, date_begin, date_end, ))
            data_row = cur.fetchall()

    data_set = pd.DataFrame(data_row, columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    for hyperparameter_set in hyperparameter_row :
        code_task = str(hyperparameter_set[0])
        for _ in range(retry_count) :
            try :
                date_begin_test, date_end_test, date_begin_train, date_end_train, \
                    accuracy, loss, mse = build_model(data_set, hyperparameter_set)                             
                update_taskstatus(code_task, ticker, 'Completed', None)
                break
                
            except Exception as e :
                print ('[ERROR] %s' %str(e))
                update_taskstatus(code_task, ticker, 'Error', str(e))
                time.sleep(retry_sleep)

        with connect_mysql() as con:
            with con.cursor() as cur:
                try :
                    SQL_INSERT = SQL_INSERT_RESULT %(code_task,
                                                    ticker,
                                                    1,
                                                    date_begin_test,
                                                    date_end_test,
                                                    date_begin_train,
                                                    date_end_train,
                                                    loss,
                                                    accuracy,
                                                    mse
                                                    )
                    cur.execute(SQL_INSERT)
                    con.commit()

                except Exception as e:
                    print (str(e))
                            
                                       
        
       