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