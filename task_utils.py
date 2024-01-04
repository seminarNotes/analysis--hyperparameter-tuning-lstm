import mysql.connector
from itertools import product
import os


PATH = os.getcwd().split('\\')

MAIN_PATH = r''
for ii in range(len(PATH)) :
    MAIN_PATH += str(PATH[ii] + r'/') 

def connect_mysql() :
    con = mysql.connector.connect(
        host = HOST,
        user = USER,
        password = PASSWORD,
        database = DATABASE
    )
    return con


def check_table(table_name) :
    SQL_CHECK_TABLE = """SHOW TABLES LIKE '%s'""" % table_name

    with connect_mysql() as con:
        with con.cursor() as cur:
            cur.execute(SQL_CHECK_TABLE)
            result = cur.fetchall()

    if len(result) == 0:
        return False
    else :
        return True
    
    
def create_table(table_name) :
    if table_name.upper() == 'TB_ASSETVALUE_RAW' :
        SQL_CREATE_TABLE = """CREATE TABLE TB_ASSETVALUE_RAW(
                            DATE_MARKET VARCHAR(20),
                            TICKER VARCHAR(20),
                            OPEN_P FLOAT,
                            HIGH_P FLOAT,
                            LOW_P FLOAT,
                            CLOSE_P FLOAT,
                            VOLUME FLOAT,
                            DATE_UPDATE DATETIME,
                            PRIMARY KEY (DATE_MARKET, TICKER)
        )"""

    elif table_name.upper() == 'TB_ASSETVALUE_ADJ' :
        SQL_CREATE_TABLE = """CREATE TABLE TB_ASSETVALUE_ADJ (
                            DATE_MARKET VARCHAR(20),
                            TICKER VARCHAR(20),
                            OPEN_P FLOAT,
                            HIGH_P FLOAT,
                            LOW_P FLOAT,
                            CLOSE_P FLOAT,
                            VOLUME FLOAT,
                            DATE_UPDATE DATETIME,
                            PRIMARY KEY (DATE_MARKET, TICKER)
        )"""

    elif table_name.upper() == 'TB_HYPERPARAMETER' :
        SQL_CREATE_TABLE = """CREATE TABLE IF NOT EXISTS TB_HYPERPARAMETER (
                            CODE_TASK VARCHAR(20),
                            EPOCH INT,
                            BATCH_SIZE INT,
                            LEARNING_RATE FLOAT,
                            RATIO_TRAIN_SET FLOAT,
                            SIZE_SEQUENCE INT,
                            SIZE_UNITS_INPUT INT,
                            SIZE_UNITS_HIDDEN INT,
                            SIZE_UNITS_OUTPUT INT,
                            DATE_UPDATE DATETIME,
                            PRIMARY KEY (CODE_TASK)   
        )"""
    
    elif table_name.upper() == 'TB_MODELRESULT' :
        SQL_CREATE_TABLE = """CREATE TABLE TB_MODELRESULT (
                            CODE_TASK VARCHAR(20),
                            TICKER VARCHAR(20),
                            VERSION_MODEL INT,
                            DATE_B_TRAIN VARCHAR(20),
                            DATE_E_TRAIN VARCHAR(20),
                            DATE_B_TEST VARCHAR(20),
                            DATE_E_TEST VARCHAR(20),
                            LOSS FLOAT,
                            ACCURACY FLOAT,
                            MSE FLOAT,
                            DATE_UPDATE DATETIME,
                            PRIMARY KEY (CODE_TASK, TICKER)
        )"""
    
    elif table_name.upper() == 'TB_TASKSTATUS' :
        SQL_CREATE_TABLE = """CREATE TABLE IF NOT EXISTS TB_TASKSTATUS (
                            CODE_TASK VARCHAR(20),
                            TICKER VARCHAR(20),
                            IDX_TASK INT,
                            STATUS VARCHAR(20) DEFAULT 'NotExecuted',
                            NOTE VARCHAR(300) DEFAULT NULL,
                            DATE_UPDATE DATETIME,
                            PRIMARY KEY (CODE_TASK)
        )"""
    else :
        raise Exception('Improper Table Name : %s' %table_name)
       
    with connect_mysql() as con:
        with con.cursor() as cur:
            cur.execute(SQL_CREATE_TABLE)
            con.commit() 
    return None

def initiate_table_hyperparameter() :
    
    SQL_INITIATE_SQL = """INSERT INTO TB_HYPERPARAMETER
                        (CODE_TASK, EPOCH, BATCH_SIZE, LEARNING_RATE,
                        RATIO_TRAIN_SET, SIZE_SEQUENCE, SIZE_UNITS_INPUT,
                        SIZE_UNITS_HIDDEN, SIZE_UNITS_OUTPUT, DATE_UPDATE)
                        VALUES
                        ('%s', %f, %f, %f, %f, %f, %f, %f, %f, NOW())
                        """
    
    list_epoch = [20, 30]           
    list_batch_size = [10, 20] 
    list_learning_rate = [0.001, 0.005] 
    list_ratio_train_set = [0.9, 0.8]
    list_size_sequence = [50]

    input_size_units_input = [50, 64]
    input_size_units_hidden = [64, 128]
    input_size_units_output = [1]

    cartesian_product = list(product(list_epoch,
                                    list_batch_size,
                                    list_learning_rate,
                                    list_ratio_train_set,
                                    list_size_sequence,
                                    input_size_units_input,
                                    input_size_units_hidden,
                                    input_size_units_output
                                    ))
    with connect_mysql() as con:
        with con.cursor() as cur:
            idx = 0
            for row in cartesian_product :
                idx += 1
                task_number = f"{idx:02}" 
                try :
                    SQL_INSERT = SQL_INITIATE_SQL %("TASK" + task_number,
                                                    float(row[0]),
                                                    float(row[1]),
                                                    float(row[2]),
                                                    float(row[3]),
                                                    float(row[4]),
                                                    float(row[5]),
                                                    float(row[6]),
                                                    float(row[7])
                    )                           
                    cur.execute(SQL_INSERT)
                    con.commit()

                except Exception as e:
                    print (str(e))
    return None

def initiate_table_taskstatus(ticker) :
  
    SQL_INITIATE_SQL = """INSERT INTO TB_TASKSTATUS
                        (CODE_TASK, TICKER, IDX_TASK, STATUS, NOTE)
                        VALUES 
                        ('%s', '%s', %s, 'NotExecuted', NULL)
                        """
    
    SQL_SELECT_DATA = """SELECT DISTINCT CODE_TASK
                        FROM TB_HYPERPARAMETER
                        WHERE CODE_TASK NOT IN (
                            SELECT DISTINCT CODE_TASK
                            FROM TB_TASKSTATUS WHERE TICKER = %s
                        )
                        """
    idx_task = 0
    with connect_mysql() as con:
        with con.cursor() as cur:
            cur.execute(SQL_SELECT_DATA, (ticker, ))
            result = cur.fetchall()

            if len(result) == 0:
                return None
            
            else :
                for row in ['SETTING_DBTABLE', 'GETTING_STOCKDATA1', 'GETTING_STOCKDATA2'] :
                    idx_task += 1
                    SQL_INSERT = SQL_INITIATE_SQL %(str(row),
                                                    str(ticker),
                                                    idx_task
                    )                 
                    cur.execute(SQL_INSERT)
                    con.commit()

                for row in result :
                    idx_task += 1
                    SQL_INSERT = SQL_INITIATE_SQL %(str(row[0]),
                                                    str(ticker),
                                                    idx_task
                    )                 
                    cur.execute(SQL_INSERT)
                    con.commit()
    return None

def check_taskstatus(code_task, ticker) :
    SQL_SELECT_STATUS = """SELECT STATUS FROM TB_TASKSTATUS
                            WHERE CODE_TASK = %s AND TICKER = %s
                           """
    with connect_mysql() as con:
        with con.cursor() as cur:
            cur.execute(SQL_SELECT_STATUS, (code_task, ticker, ))
            status = cur.fetchone()

    return status[0]


def update_taskstatus(code_task, ticker, status, err_message) :

    SQL_UPDATE_COMPLETE = """UPDATE TB_TASKSTATUS
                           SET STATUS = '%s', DATE_UPDATE = NOW()
                           WHERE CODE_TASK = '%s' AND TICKER = '%s'
                           """

    SQL_UPDATE_ERROR = """UPDATE TB_TASKSTATUS
                           SET STATUS = '%s', NOTE = '%s', DATE_UPDATE = NOW()
                           WHERE CODE_TASK = '%s' AND TICKER = '%s'
                           """
    
    with connect_mysql() as con:
        with con.cursor() as cur:
            try :
                if status == 'Completed' :
                    SQL_UPDATE = SQL_UPDATE_COMPLETE %(status,
                                                    code_task,
                                                    ticker                            
                    ) 
                elif status == 'Error' :
                    SQL_UPDATE = SQL_UPDATE_ERROR %(status,
                                                    err_message,
                                                    code_task,
                                                    ticker
                    )                      
                cur.execute(SQL_UPDATE)
                con.commit()

            except Exception as e:
                print (str(e))
    return None

if __name__ == '__main__' :
    
    ticker = 'AAPL'
    
    list_table = ['TB_ASSETVALUE_RAW',
                'TB_ASSETVALUE_ADJ',
                'TB_HYPERPARAMETER',
                'TB_MODELRESULT',
                'TB_TASKSTATUS']
    
    for table in list_table :
        if check_table(table) :
            print ('Table %s already exists.' %table)
        else : 
            try :
                create_table(table)
                print ('Table %s created.' %table)
            except Exception as e :
                print ('[ERROR] %s' %str(e))

    initiate_table_hyperparameter()

    initiate_table_taskstatus(ticker)

    update_taskstatus('SETTING_DBTABLE', ticker, 'Completed', None)