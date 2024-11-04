import pprint
import time
import datetime
import pandas as pd
import os
from tqdm import tqdm

import pandas as pd
# import pandas_ta as ta
import numpy as np
from tabulate import tabulate

# from openapi_client import openapi
# from matplotlib.dates import date2num
from pathlib import Path

from tinkoff.invest import Client
from tinkoff.invest import Client
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest import Client, SecurityTradingStatus
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.utils import quotation_to_decimal

from datetime import timedelta
from pathlib import Path

import os
from datetime import timedelta

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now

from sqlalchemy import create_engine
import pymysql

# from graph import Graph

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.set_option('mode.chained_assignment', None)

def df_print(df, count_rows=False):
    """
    Выводит датафрейм в виде красивой таблицы в консоль с использованием формата fancy_grid.

    Parameters:
    - df (pd.DataFrame): Датафрейм для вывода.

    Returns:
    - None
    """
    if count_rows:
        if count_rows < 0:
            count_rows = abs(count_rows)
            print(df.tail(count_rows).to_markdown(tablefmt="fancy_grid"))
        else:
            print(df.head(count_rows).to_markdown(tablefmt="fancy_grid"))
    else:
        print(df.to_markdown(tablefmt="fancy_grid"))


class APITinkoff:
    '''
    Ссылка на документацию - https://github.com/RussianInvestments/invest-python
    Библиотека - pip install tinkoff-investments
    Примеры - https://tinkoff.github.io/invest-python/examples/#query
    '''

    timeframes = {
        '1m': CandleInterval.CANDLE_INTERVAL_1_MIN,  # Интервал от 1 минуты до 1 дня.
        '3m': CandleInterval.CANDLE_INTERVAL_3_MIN,  # Интервал от 3 минут до 1 дня.
        '5m': CandleInterval.CANDLE_INTERVAL_5_MIN,  # Интервал от 5 минут до 1 дня.
        '15m': CandleInterval.CANDLE_INTERVAL_15_MIN,  # Интервал от 15 минут до 1 дня.
        '30m': CandleInterval.CANDLE_INTERVAL_30_MIN,  # Интервал от 30 минут до 2 дней.
        '1h': CandleInterval.CANDLE_INTERVAL_HOUR,  # Интервал от 1 часа до 1 недели.
        '4h': CandleInterval.CANDLE_INTERVAL_4_HOUR,  # Интервал от 4 часов до 1 месяца.
        '1d': CandleInterval.CANDLE_INTERVAL_DAY,  # Интервал от 1 дня до 1 года.
        '1w': CandleInterval.CANDLE_INTERVAL_WEEK,  # Интервал от 1 недели до 2 лет.
        '1M': CandleInterval.CANDLE_INTERVAL_MONTH  # Интервал от 1 месяца до 10 лет.
    }
    delta = {
        '1m': 1,  # Интервал 1 минута.
        '3m': 3,  # Интервал 3 минуты.
        '5m': 5,  # Интервал 5 минут.
        '15m': 15,  # Интервал 15 минут.
        '30m': 30,  # Интервал 30 минут.
        '1h': 60,  # Интервал 1 час.
        '4h': 240,  # Интервал 4 часа.
        '1d': 1440,  # Интервал 1 день (24 часа).
        '1w': 1440 * 7,  # Интервал 1 неделя (7 дней).
        '1M': 1440 * 30  # Интервал 1 месяц (30 дней).
    }
    # max_delta = {
    #     '1m': 1440,  # Интервал от 1 минуты до 1 дня. (24 часа * 60 минут)
    #     '3m': 1440,  # Интервал от 3 минут до 1 дня. (24 часа * 60 минут)
    #     '5m': 1440,  # Интервал от 5 минут до 1 дня. (24 часа * 60 минут)
    #     '15m': 1440,  # Интервал от 15 минут до 1 дня. (24 часа * 60 минут)
    #     '30m': 1440 * 2,  # Интервал от 30 минут до 2 дней. (48 часов * 60 минут)
    #     '1h': 1440 * 7,  # Интервал от 1 часа до 1 недели. (7 дней * 24 часа * 60 минут)
    #     '4h': 1440 * 30,  # Интервал от 4 часов до 1 месяца. (30 дней * 24 часа)
    #     '1d': 1440 * 365,  # Интервал от 1 дня до 1 года. (365 дней)
    #     '1w': 1440 * 365 * 2,  # Интервал от 1 недели до 2 лет. (2 года * 52 недели)
    #     '1M': 1440 * 365 * 10,  # Интервал от 1 месяца до 10 лет. (10 лет * 12 месяцев)
    # }
    max_delta = {
        '1m': 1440 * 7,  # Интервал от 1 минуты до 1 дня. (24 часа * 60 минут)
        '3m': 1440 * 7,  # Интервал от 3 минут до 1 дня. (24 часа * 60 минут)
        '5m': 1440 * 7,  # Интервал от 5 минут до 1 дня. (24 часа * 60 минут)
        '15m': 1440 * 7,  # Интервал от 15 минут до 1 дня. (24 часа * 60 минут)
        '30m': 1440 * 7,  # Интервал от 30 минут до 2 дней. (48 часов * 60 минут)
        '1h': 1440 * 365,  # Интервал от 1 часа до 1 недели. (7 дней * 24 часа * 60 минут)
        '4h': 1440 * 365,  # Интервал от 4 часов до 1 месяца. (30 дней * 24 часа)
        '1d': 1440 * 365 * 10,  # Интервал от 1 дня до 1 года. (365 дней)
        '1w': 1440 * 365 * 10,  # Интервал от 1 недели до 2 лет. (2 года * 52 недели)
        '1M': 1440 * 365 * 10,  # Интервал от 1 месяца до 10 лет. (10 лет * 12 месяцев)
    }

    def __init__(self, token, url):
        self.token, self.url = token, url
        self.catalog = self.get_info_all_tickers(filter_types=["shares", "currencies"])

    def __get_settings(self, timeframe, count_bars=False):
        delta = self.delta[timeframe]
        max_delta = self.max_delta[timeframe]
        user_delta = delta * count_bars

        if not count_bars or user_delta >= max_delta:
            time_delta = max_delta
        else:
            time_delta = user_delta

        interval = self.timeframes[timeframe]
        starttime = now() - timedelta(minutes=time_delta)
        return interval, starttime

    def get_info_all_tickers(self, filter_tickers=[], filter_types=[]):
        filtered_data = []
        with Client(self.token) as client:
            instruments: InstrumentsService = client.instruments
            for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
                for item in getattr(instruments, method)().instruments:
                    filt = True
                    if filter_tickers and item.ticker not in filter_tickers:
                        filt = False
                    if filter_types and method not in filter_types:
                        filt = False
                    if filt:
                        filtered_data.append(
                            {
                                "name": item.name,
                                "ticker": item.ticker,
                                "class_code": item.class_code,
                                "figi": item.figi,
                                "uid": item.uid,
                                "type": method,
                                "min_price_increment": quotation_to_decimal(item.min_price_increment),
                                "scale": 9 - len(str(item.min_price_increment.nano)) + 1,
                                "lot": item.lot,
                                "trading_status": str(SecurityTradingStatus(item.trading_status).name),
                                "api_trade_available_flag": item.api_trade_available_flag,
                                "currency": item.currency,
                                "exchange": item.exchange,
                                "buy_available_flag": item.buy_available_flag,
                                "sell_available_flag": item.sell_available_flag,
                                "short_enabled_flag": item.short_enabled_flag,
                                "klong": quotation_to_decimal(item.klong),
                                "kshort": quotation_to_decimal(item.kshort),
                            }
                        )
        return pd.DataFrame(filtered_data)

    def get_figi_by_ticker_from_memory(self, ticker):
        filt = self.catalog.loc[self.catalog['ticker'] == ticker]
        if not filt.empty:
            if len(filt.index) == 1:
                figi = filt.to_dict('records')[0]['figi']
            else:
                raise ValueError('Error: Найдено несколько вариантов figi')
        else:
            raise ValueError('Error: Найдено 0 вариантов figi')
        return figi

    def get_bars(self, timeframe, ticker=False, figi=False, count_bars=False):
        candles_data = []
        interval, starttime = self.__get_settings(timeframe, count_bars)
        if not figi and ticker:
            figi = self.get_figi_by_ticker_from_memory(ticker)
        with Client(self.token) as client:
            for candle in client.get_all_candles(
                    figi=figi,
                    from_=starttime,
                    interval=interval,
            ):
                price_scale = 1e9  # 1 миллиард для корректной конвертации дробной части
                candles_data.append({
                    'start': candle.time.replace(tzinfo=None),  # + timedelta(hours=3),
                    # 'end': np.nan,
                    'volume': candle.volume,
                    'open': candle.open.units + candle.open.nano / price_scale,
                    'close': candle.close.units + candle.close.nano / price_scale,
                    'high': candle.high.units + candle.high.nano / price_scale,
                    'low': candle.low.units + candle.low.nano / price_scale
                })

        # Создайте датафрейм из списка данных свечей
        df = pd.DataFrame(candles_data)
        # df['end'] = df['start'].shift(-1) - timedelta(microseconds=1)
        df = df.reset_index(drop=False)
        df.rename(columns={'index': 'id'}, inplace=True)
        return df



def save_to_db(data, table_name, username, password, host, port, drop_table=True):
    table_name = table_name.lower()
    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/bars_data")
    df = pd.DataFrame(data)
    # if drop_table:
    #     drop_table_query = f'DROP TABLE IF EXISTS {table_name}'
    #     with engine.connect() as connection:
    #         # Execute the DROP TABLE query
    #         connection.execute(drop_table_query)
    df.to_sql(table_name, con=engine, if_exists='append', index=False)

def get_bars(count_bars):
    init = APITinkoff(token, url)
    info = init.get_info_all_tickers(filter_types=["shares", "currencies"])

    for i, s in tqdm(info.iterrows(), total=len(info), desc="Processing tickers"):
        temp = pd.DataFrame()
        for tf in ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']:
            bars = pd.DataFrame()
            try:
                bars = init.get_bars(ticker=s['ticker'], timeframe=tf, count_bars=count_bars)
            except Exception as e:
                time.sleep(5)
                try:
                    bars = init.get_bars(ticker=s['ticker'], timeframe=tf, count_bars=count_bars)
                except Exception as e:
                    time.sleep(5)
                    try:
                        bars = init.get_bars(ticker=s['ticker'], timeframe=tf, count_bars=count_bars)
                    except Exception as e:
                        print(i, tf, '=======>', e)

            if not bars.empty:
                bars['ticker'] = s['ticker']
                bars['timeframe'] = tf
                temp = pd.concat([temp, bars])
        save_to_db(temp, 'all', username, password, host, port)





'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''

url = "https://api-invest.tinkoff.ru/openapi"
token = "t.r8abIy3gkK-JfS8MOXXSxRF0gDUTPPKYEEe_ZawY9Uad37R2RYXUzxUmM10wjKOQFKNdbnhcruE4EULIkhn9Qw"
username, password, host, port = 'bot', 'bot', 'localhost', 3306

'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''


# ticker = 'SBER'
# timeframe = '1M'
count_bars = 10000
bars_def = get_bars(count_bars=count_bars)


# df_print(bars_def, count_rows=-10)

# Graph(ResearchResult, ticker, timeframe, perc_par='fullres', par=False, hard=75, medium=50, lite=25, show=False)._basic_graph()



# MoveAnalise_bars
# IndicatorAnalise_bars = IndicatorAnalise(bars_def.copy()).get_result()
# IndicatorAnalise_bars
# save_to_db(MoveAnalise_bars, 'MoveAnalise_bars', username, password, host, port)
