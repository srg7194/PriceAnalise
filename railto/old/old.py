


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


class BarAnalise:
    def __init__(self, df):
        self.df = df

    def get_situation(self, compare_list, curr_num, compare_num):
        compare = {
            0: '',
            1: '>',
            2: '=',
            3: '<',
        }
        for el1 in compare_list:
            for el2 in compare_list:
                iter_res = (
                        ((self.df[el1].shift(curr_num) > self.df[el2].shift(compare_num)) * 1) |
                        ((self.df[el1].shift(curr_num) == self.df[el2].shift(compare_num)) * 2) |
                        ((self.df[el1].shift(curr_num) < self.df[el2].shift(compare_num)) * 3)
                ).replace(compare)
                iter_res = iter_res.replace(compare)
                self.df[f'situation_{el2}{curr_num}_{el1}{compare_num}'] = iter_res
        return self.df

    def get_volatility(self):
        def calc(highline, lowline):
            name = f'{highline}_{lowline}'
            self.df[f'volatility_perc_{name}'] = round(
                abs(self.df[highline] - self.df[lowline]) / self.df['open'] * 100)
            return self.df

        self.df = calc('high', 'low')
        self.df = calc('open', 'close')
        self.df['volatility_perc_delta'] = self.df.volatility_perc_high_low - self.df.volatility_perc_open_close
        return self.df

    def calc_bar_proportions(self, round_step):
        def calc(row):
            price_range = row.high - row.low
            proportions_dict = {
                "us": 0,
                "b": 0,
                "ls": 0
            }
            if price_range > 0:
                body = abs(row.close - row.open)
                upper_shadow = abs(row.high - max(row.open, row.close))
                lower_shadow = abs(min(row.open, row.close) - row.low)

                raw_body = (body / price_range) * 100
                raw_upper_shadow = (upper_shadow / price_range) * 100
                raw_lower_shadow = (lower_shadow / price_range) * 100
                proportions_dict = {
                    "us": round(raw_upper_shadow / round_step) * round_step,
                    "b": round(raw_body / round_step) * round_step,
                    "ls": round(raw_lower_shadow / round_step) * round_step
                }
            return pd.Series(proportions_dict)

        proportions_bar = self.df.apply(calc, axis=1)
        self.df['proposion_bar_upper_shadow'] = proportions_bar['us']
        self.df['proposion_bar_body'] = proportions_bar['b']
        self.df['proposion_bar_lower_shadow'] = proportions_bar['ls']
        return self.df

    def get_result(self):
        self.df = self.get_volatility()
        self.df = self.calc_bar_proportions(round_step=5)
        self.df = self.get_situation(['open', 'close', 'high', 'low'], curr_num=0, compare_num=1)
        self.df = self.get_situation(['open', 'close', 'high', 'low'], curr_num=0, compare_num=2)
        return self.df


class AdditionalMethods:
    def __init__(self, df):
        self.df = df

    def __get_pivots(self, name):
        res_name = f'pivot_{name}'
        self.df[res_name] = (
                (
                        (self.df[name] >= self.df[name].shift(1)) & (self.df[name] >= self.df[name].shift(-1))
                ) * 1 |
                (
                        (self.df[name] <= self.df[name].shift(1)) & (self.df[name] <= self.df[name].shift(-1))
                ) * -1
        )
        self.df[res_name] = self.df[res_name].replace(0, np.nan)
        return res_name

    def __get_crossing_zero(self, name):
        res_name = f'crossing_zero_{name}'
        self.df[res_name] = (
            (
                    ((self.df[name] > 0) & (self.df[name].shift(1).isna())) * 1 |
                    ((self.df[name] > 0) & (self.df[name].shift(1) < 0)) * 1 |
                    ((self.df[name] < 0) & (self.df[name].shift(1).isna())) * -1 |
                    ((self.df[name] < 0) & (self.df[name].shift(1) > 0)) * -1
            )
        )
        self.df[res_name] = self.df[res_name].replace(0, np.nan)
        self.df[res_name] = self.df[res_name].ffill()

        group_num = 0
        prev_value = None
        for i, row in self.df.iterrows():
            current_value = row[res_name]
            if i > 0 and not pd.isna(current_value) and current_value != prev_value:
                group_num += 1
            self.df.loc[i, res_name] = current_value * group_num
            prev_value = current_value
        return res_name

    def __get_wave_pivots(self, name, pivot_name, crossing_zero_name):
        res_name = f'wave_pivots_{name}'
        filt = self.df.loc[self.df[pivot_name].notna()]
        result_max = filt.groupby(crossing_zero_name)[name].idxmax()
        result_max = result_max.loc[result_max.index > 0]
        result_min = filt.groupby(crossing_zero_name)[name].idxmin()
        result_min = result_min.loc[result_min.index < 0]
        result = pd.concat([result_max, result_min], axis=0)
        result = result.reset_index()
        result.columns = [res_name, 'id']
        result[res_name] = result[res_name] / abs(result[res_name])
        self.df = self.df.merge(result, on='id', how='left')
        return res_name

    @staticmethod
    def __get_divergence_long(val1, val2, base_name, compare_name, volatility_part):
        result = {
            'classic': np.nan,
            'latent': np.nan,
            'extend': np.nan
        }
        if not val1:
            return result
        elif val1[compare_name] > val2[compare_name] and val1[base_name] < val2[base_name]:
            result.update({'classic': 1})
        elif val1[compare_name] < val2[compare_name] and val1[base_name] > val2[base_name]:
            result.update({'latent': 1})
        elif val2[compare_name] - volatility_part <= val1[compare_name] <= val2[compare_name] + volatility_part \
                and val1[base_name] < val2[base_name]:
            result.update({'extend': 1})
        return result

    @staticmethod
    def __get_divergence_short(val1, val2, base_name, compare_name, volatility_part):
        result = {
            'classic': np.nan,
            'latent': np.nan,
            'extend': np.nan
        }
        if not val1:
            return result
        elif val1[compare_name] < val2[compare_name] and val1[base_name] > val2[base_name]:
            result.update({'classic': 1})
        elif val1[compare_name] > val2[compare_name] and val1[base_name] < val2[base_name]:
            result.update({'latent': 1})
        elif val2[compare_name] - volatility_part <= val1[compare_name] <= val2[compare_name] + volatility_part \
                and val1[base_name] > val2[base_name]:
            result.update({'extend': 1})
        return result

    def __get_divergence_base(self, add_name, filter_column, base_name, compare_name, long, perc_volatility=10):
        def filt():
            if long:
                df = self.df.loc[self.df[filter_column] < 0]
            else:
                df = self.df.loc[self.df[filter_column] > 0]
            return df

        def update_df():
            update_dict = {}
            div_type = 'long' if long else 'short'
            for prefix, div_dict in [
                (f"divergence_{div_type}_{add_name}_1_2", div_1_2),
                (f"divergence_{div_type}_{add_name}_1_3", div_1_3)
            ]:
                for k, v in div_dict.items():
                    column_name = f'{prefix}_{k}_{base_name}_{compare_name}'
                    update_dict[column_name] = v
            self.df.loc[curr_row['Index'], update_dict.keys()] = update_dict.values()

        df = filt()
        prev_row = None
        pprev_row = None
        for curr_row in df.itertuples():
            curr_row = curr_row._asdict()
            volatility = abs(curr_row['high'] - curr_row['low'])
            volatility_part = volatility / 100 * perc_volatility

            if long:
                div_1_2 = self.__get_divergence_long(prev_row, curr_row, base_name, compare_name, volatility_part)
                div_1_3 = self.__get_divergence_long(pprev_row, curr_row, base_name, compare_name, volatility_part)
            else:
                div_1_2 = self.__get_divergence_short(prev_row, curr_row, base_name, compare_name, volatility_part)
                div_1_3 = self.__get_divergence_short(pprev_row, curr_row, base_name, compare_name, volatility_part)

            update_df()
            pprev_row = prev_row
            prev_row = curr_row
        return self.df

    def get_divergence(self, name, pivot, wave):
        pivot_name, crossing_zero_name, wave_pivots_name = False, False, False

        if pivot or wave:
            pivot_name = self.__get_pivots(name)

        if wave:
            crossing_zero_name = self.__get_crossing_zero(name)
            wave_pivots_name = self.__get_wave_pivots(name, pivot_name, crossing_zero_name)

        if pivot:
            self.df = self.__get_divergence_base('pivot', pivot_name, name, 'close', long=True, perc_volatility=10)
            self.df = self.__get_divergence_base('pivot', pivot_name, name, 'close', long=False, perc_volatility=10)
            self.df = self.__get_divergence_base('pivot', pivot_name, 'low', name, long=True, perc_volatility=10)
            self.df = self.__get_divergence_base('pivot', pivot_name, 'high', name, long=False, perc_volatility=10)

        if wave:
            self.df = self.__get_divergence_base('wave', wave_pivots_name, name, 'close', long=True, perc_volatility=10)
            self.df = self.__get_divergence_base('wave', wave_pivots_name, name, 'close', long=False,
                                                 perc_volatility=10)
            self.df = self.__get_divergence_base('wave', wave_pivots_name, 'low', name, long=True, perc_volatility=10)
            self.df = self.__get_divergence_base('wave', wave_pivots_name, 'high', name, long=False, perc_volatility=10)
        return self.df


class IndicatorAnalise:
    '''
    brew install ta-lib
    pip install TA-Lib
    pip install pandas_talib
    '''

    def __init__(self, df):
        self.df = df
        self.add_methods = AdditionalMethods(df)

    def calc_MACD(self, signal: int, fast: int, slow: int):
        fast, signal, slow = int(fast), int(signal), int(slow)
        name = f'{signal}_{fast}_{slow}'
        data = ta.macd(self.df['close'], signal=signal, fast=fast, slow=slow)

        self.df[f'MACDh_{name}'] = data[data.columns[1]]
        self.df[f'MACDf_{name}'] = data[data.columns[0]]
        self.df[f'MACDs_{name}'] = data[data.columns[2]]

        self.df = self.add_methods.get_divergence(f'MACDh_{name}', pivot=True, wave=True)
        self.df = self.add_methods.get_divergence(f'MACDf_{name}', pivot=True, wave=True)
        self.df = self.add_methods.get_divergence(f'MACDs_{name}', pivot=True, wave=True)

        self.df[f'sum_MACD_long_{name}'] = self.df[[col for col in self.df.columns if 'divergence_long' in col]].sum(
            axis=1)
        self.df[f'sum_MACD_short_{name}'] = self.df[[col for col in self.df.columns if 'divergence_short' in col]].sum(
            axis=1)
        return self.df

    def calc_RSI(self, period):
        period = int(period)
        name = f'RSI_{period}'
        self.df[name] = ta.rsi(self.df['close'], period)
        return self.df

    def calc_CCI(self, period):
        name = f'CCI_{period}'
        self.df[name] = ta.cci(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
        return self.df

    def calc_WPR(self, period):
        name = f'WPR_{period}'
        self.df[name] = ta.willr(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
        return self.df

    def calc_EFI(self, period):
        name = f'EFI_{period}'
        self.df[name] = ta.efi(self.df['close'], self.df['vol'], timeperiod=period)
        return self.df

    def calc_CMF(self, period):
        name = f'CMF_{period}'
        self.df[name] = ta.cmf(self.df['high'], self.df['low'], self.df['close'], self.df['vol'], timeperiod=period)
        return self.df

    def calc_ROC(self, period):
        name = f'ROC_{period}'
        self.df[name] = ta.roc(self.df['close'], timeperiod=period)
        return self.df

    def calc_SMA(self, period):
        name = f'SMA_{period}'
        self.df[name] = ta.sma(self.df['close'], timeperiod=period)
        return self.df

    def calc_EMA(self, period):
        name = f'EMA_{period}'
        self.df[name] = ta.ema(self.df['close'], timeperiod=period)
        return self.df

    def get_result(self):
        self.df = self.calc_MACD(signal=9, fast=12, slow=26)
        # self.df = self.calc_RSI(14)
        # self.df = self.calc_CCI(14)
        # self.df = self.calc_WPR(14)
        # self.df = self.calc_EFI(14)
        # self.df = self.calc_CMF(14)
        # self.df = self.calc_ROC(14)
        # self.df = self.calc_SMA(14)
        # self.df = self.calc_EMA(14)
        return self.df


class MoveAnalise:
    def __init__(self, df):
        self.df = df

    def get_type_situation_bar(self):
        bar_types = {
            0: 'unknown',
            1: 'up',
            2: 'down',
            3: 'vol+',
            4: 'vol-',
            5: 'zero'

        }
        name = f'type_situation_bar'
        self.df[name] = (
                (((self.df['high'] >= self.df['high'].shift(1)) & (self.df['low'] > self.df['low'].shift(1))) * 1) |
                (((self.df['high'] < self.df['high'].shift(1)) & (self.df['low'] <= self.df['low'].shift(1))) * 2) |
                (((self.df['high'] > self.df['high'].shift(1)) & (self.df['low'] < self.df['low'].shift(1))) * 3) |
                (((self.df['high'] < self.df['high'].shift(1)) & (self.df['low'] > self.df['low'].shift(1))) * 4) |
                (((self.df['high'] == self.df['high'].shift(1)) & (self.df['low'] == self.df['low'].shift(1))) * 5)
        )
        self.df[name] = self.df[name].replace(bar_types)
        return self.df

    def get_pivots(self, add_name, highline, lowline):
        cond1 = (self.df[highline].shift(1) <= self.df[highline]) & (
                self.df[highline] >= self.df[highline].shift(-1))
        cond2 = (self.df[lowline].shift(1) >= self.df[lowline]) & (self.df[lowline] <= self.df[lowline].shift(-1))

        self.df[f'pivot_{add_name}_{highline}_{lowline}'] = np.where(
            cond1 & cond2, 0,
            np.where(
                cond1, 1,
                np.where(cond2, -1, np.nan)
            )
        )
        return self.df

    def get_trend(self):
        # Filter data for downward trend
        down_trend_df = self.df.loc[self.df['pivot_bar_high_low'] >= 0].copy()
        down_trend_df['down'] = np.where(down_trend_df['high'] > down_trend_df['high'].shift(-1), -1, 0)

        # Filter data for upward trend
        up_trend_df = self.df.loc[self.df['pivot_bar_high_low'] <= 0].copy()
        up_trend_df['up'] = np.where(up_trend_df['low'] < up_trend_df['low'].shift(-1), 1, 0)

        # Joining downward trend data to the main dataframe
        self.df = self.df.join(up_trend_df['up'])
        self.df = self.df.join(down_trend_df['down'])
        self.df['up'] = self.df['up'].ffill()
        self.df['down'] = self.df['down'].ffill()
        self.df['res'] = self.df['up'] + self.df['down']
        # Returning upward trend data
        return self.df


    def get_result(self):
        self.df = self.get_pivots('bar', 'high', 'low')
        # self.df = self.get_type_situation_bar()
        self.df = self.get_trend()
        # self.df = self.get_level_power()

        # self.df['starttime'] = np.nan #время окончания
        # self.df['endtime'] = np.nan #время начала
        # self.df['type'] = np.nan #тип тренда
        # self.df['max_price'] = np.nan  #максимальная цена
        # self.df['min_price'] = np.nan #минимальная цена
        # self.df['pivot_move'] = np.nan #потенциал по пивотам
        # self.df['close_move'] = np.nan #потенциал по close
        # self.df['bars_from_start'] = np.nan #количество баров с начала тренда
        # self.df['bars_to_end'] = np.nan #количество баров до окончания тренда
        # self.df['length'] = np.nan #количество баров в тренде






        # self.df = self.get_pivots('bar', 'vol', 'vol')
        # self.df = self.test_get_trend()
        # self.df = self.test_get_level_power()

        # self.df = self.get_level_power('bar', 'high', 'low')
        # self.df = self.get_pivots('bar', 'close', 'close')
        # self.df =self.get_signal_pivot()

        # self.df = self.get_wave('bar', 'close', 'close')
        # хай клосе лоу клосе

        return self.df


def get_bars(ticker, timeframe, count_bars):
    init = APITinkoff(token, url)
    bars_def = init.get_bars(ticker=ticker, timeframe=timeframe, count_bars=count_bars)
    return bars_def

def save_to_db(data, table_name, username, password, host, port, drop_table=True):
    table_name = table_name.lower()
    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/bars_data")
    df = pd.DataFrame(data)
    # if drop_table:
    #     drop_table_query = f'DROP TABLE IF EXISTS {table_name}'
    #     with engine.connect() as connection:
    #         # Execute the DROP TABLE query
    #         connection.execute(drop_table_query)
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)


class Research:
    """
    Docs
    default_bars_data:
        type - DataFrame
        columns:
            t_open - time_open, str, UTC
            open - price_open, float
            high - price_high, float
            low - price_low, float
            close - price_close, float
            volume - volume, float
            t_close - time_close, str, UTC
    """

    def __init__(self, default_bars_data, period=14, period_fast=7, period_slow=28):
        self.bars_data = default_bars_data
        self.def_columns = default_bars_data.columns
        self.per = period
        self.per_f = period_fast
        self.per_s = period_slow

    def __levels(self, name: str):
        """
        1 - уровень максимума
        -1 - уровень минимума
        """
        p_name = self.bars_data[f'{name}'].shift(periods=1)
        c_name = self.bars_data[f'{name}']
        n_name = self.bars_data[f'{name}'].shift(periods=-1)
        max_ = ((p_name <= c_name) & (n_name <= c_name)).astype(float)
        min_ = ((p_name >= c_name) & (n_name >= c_name)).astype(float)
        self.bars_data[f'{name}/level'] = max_ - min_

    def __out_of_range(self, h_range, l_range, name):
        higher_h_range = -(self.bars_data[name] >= h_range).astype(float)
        lover_l_range = (self.bars_data[name] <= l_range).astype(float)
        # central_range = h_range - (h_range - l_range)/2
        # higher_с_range = -(self.bars_data[name] >= central_range).astype(float)
        # lover_с_range = (self.bars_data[name] <= central_range).astype(float)
        # self.bars_data[f'{name}/out_of_range'] = lover_l_range + higher_h_range + higher_с_range + lover_с_range
        self.bars_data[f'{name}/out_of_range'] = lover_l_range + higher_h_range

    def __osc_levels(self, name: str):
        """
        Метод можно использовать для индикаторов проходящих через 0
        """

        def h_max(filter_max_hard, df):
            start, finish = None, None
            for ind, str in filter_max_hard[:-1].iterrows():
                if start is None:
                    start = ind
                if str['reverce'] == 1 and not finish:
                    finish = ind + 1
                if start != None and finish != None:
                    if start == finish:
                        def_ind = filter_max_hard.loc[start, 'index']
                        df.loc[def_ind, f'{name}/level'] += 1
                    else:
                        def_ind = filter_max_hard.loc[filter_max_hard[name][start:finish].idxmax(), 'index']
                        df.loc[def_ind, f'{name}/level'] += 1
                    start, finish = ind + 1, None

        def h_min(filter_min_hard, df):
            start, finish = None, None
            for ind, str in filter_min_hard[:-1].iterrows():
                if start is None:
                    start = ind
                if str['reverce'] == -1 and not finish:
                    finish = ind + 1
                if start != None and finish != None:
                    if start == finish:
                        def_ind = filter_min_hard.loc[start, 'index']
                        df.loc[def_ind, f'{name}/level'] -= 1
                    else:
                        def_ind = filter_min_hard.loc[filter_min_hard[name][start:finish].idxmin(), 'index']
                        df.loc[def_ind, f'{name}/level'] -= 1
                    start, finish = ind + 1, None

        # находим уровни индикатора
        self.__levels(name)

        # фильтруем уровни индикатора
        filter_hard = self.bars_data.loc[self.bars_data[f'{name}/level'] != 0]

        # находим последние максимумы и минимумы в волне (до переходя через 0)
        max_ = ((filter_hard[f'{name}'] > 0) & (filter_hard[f'{name}'].shift(periods=-1) < 0)).astype(float)
        min_ = ((filter_hard[f'{name}'] < 0) & (filter_hard[f'{name}'].shift(periods=-1) > 0)).astype(float)
        filter_hard['reverce'] = max_ - min_

        # готовим фильтры
        filter_max_hard = filter_hard.loc[(filter_hard[f'{name}/level'] == 1)].reset_index()
        filter_min_hard = filter_hard.loc[(filter_hard[f'{name}/level'] == -1)].reset_index()

        h_max(filter_max_hard, self.bars_data)
        h_min(filter_min_hard, self.bars_data)

    def __divergence(self, name: str):
        def search_divergence(name, surname, df, df_max, df_min, div12=True, div13=True):
            def d_max(c_name, p_name, c_price, p_price, mid_name=None):
                if mid_name is not None:
                    return ((c_price >= p_price) & (c_name < p_name)) | ((c_price < p_price) & (c_name > p_name)) \
                           & mid_name < c_name
                else:
                    return ((c_price >= p_price) & (c_name < p_name)) | ((c_price < p_price) & (c_name > p_name))

            def d_min(c_name, p_name, c_price, p_price, mid_name=None):
                if mid_name is not None:
                    return ((c_price <= p_price) & (c_name > p_name)) | ((c_price > p_price) & (c_name < p_name)) \
                           & mid_name > c_name
                else:
                    return ((c_price <= p_price) & (c_name > p_name)) | ((c_price > p_price) & (c_name < p_name))

            df = df.copy()
            c_name = df[f'{name}']
            c_max_price = df['high']
            c_min_price = df['low']

            ##### MAX #####
            ### 1-2 ###
            if div12:
                df[f'{name}({surname}12)max_price'] = df_max['high']
                df[f'{name}({surname}12)max_value'] = df_max[f'{name}']
                p_m12max_name = (df[f'{name}({surname}12)max_value'].fillna(method='ffill')).shift(periods=1)
                p_m12max_price = (df[f'{name}({surname}12)max_price'].fillna(method='ffill')).shift(periods=1)
                df[f'{name}/div({surname}12)_max'] = -d_max(c_name, p_m12max_name, c_max_price, p_m12max_price).astype(
                    float)

            ### 1-3 ###
            if div13:
                df[f'{name}({surname}13)max_price'] = df_max['high'].shift(periods=1)
                df[f'{name}({surname}13)max_value'] = df_max[f'{name}'].shift(periods=1)
                p_m13max_name = (df[f'{name}({surname}13)max_value'].fillna(method='ffill')).shift(periods=1)
                p_m13max_price = (df[f'{name}({surname}13)max_price'].fillna(method='ffill')).shift(periods=1)
                df[f'{name}/div({surname}13)_max'] = -d_max(c_name, p_m13max_name, c_max_price, p_m13max_price,
                                                            p_m12max_name).astype(float)

            ##### MIN #####
            ### 1-2 ###
            if div12:
                df[f'{name}({surname}12)min_price'] = df_min['low']
                df[f'{name}({surname}12)min_value'] = df_min[f'{name}']
                p_m12min_name = (df[f'{name}({surname}12)min_value'].fillna(method='ffill')).shift(periods=1)
                p_m12min_price = (df[f'{name}({surname}12)min_price'].fillna(method='ffill')).shift(periods=1)
                df[f'{name}/div({surname}12)_min'] = d_min(c_name, p_m12min_name, c_min_price, p_m12min_price).astype(
                    float)

            ### 1-3 ###
            if div13:
                df[f'{name}({surname}13)min_price'] = df_min['low'].shift(periods=1)
                df[f'{name}({surname}13)min_value'] = df_min[f'{name}'].shift(periods=1)
                p_m13min_name = (df[f'{name}({surname}13)min_value'].fillna(method='ffill')).shift(periods=1)
                p_m13min_price = (df[f'{name}({surname}13)min_price'].fillna(method='ffill')).shift(periods=1)
                df[f'{name}/div({surname}13)_min'] = d_min(c_name, p_m13min_name, c_min_price, p_m13min_price,
                                                           p_m12min_name).astype(float)

            return df

        self.__osc_levels(name)
        df_list = []

        ##### BAR to BAR - lite #####
        # lite_max = self.bars_data
        # lite_min = self.bars_data
        # df_list.append(search_divergence(name, 'l', self.bars_data, lite_max, lite_min))

        # ##### LEVEL to LEVEL - medium #####
        med_max = self.bars_data.loc[self.bars_data[f'{name}/level'] > 0]
        med_min = self.bars_data.loc[self.bars_data[f'{name}/level'] < 0]
        df_list.append(search_divergence(name, 'm', self.bars_data, med_max, med_min, div12=True, div13=False))

        # ##### STRONG_LEVEL to STRONG_LEVEL - hard #####
        hard_max = med_max.loc[self.bars_data[f'{name}/level'] == 2]
        hard_min = med_min.loc[self.bars_data[f'{name}/level'] == -2]
        df_list.append(search_divergence(name, 'h', self.bars_data, hard_max, hard_min, div12=True, div13=True))

        for dframe in df_list:
            for column in dframe.columns:
                if '/div(' in column:
                    self.bars_data[column] = dframe[column]

    def __MACD(self, surname: str, period: int, period_slow: int, period_fast: int, drop_values=True):

        name = f'MACD({surname})'
        data = self.bars_data[surname]
        macd = ta.macd(data, period, period_slow, period_fast)
        self.bars_data[f'{name}f'], self.bars_data[f'{name}h'], self.bars_data[f'{name}s'] = \
            macd[macd.columns[0]], macd[macd.columns[1]], macd[macd.columns[2]]
        # fast
        self.__divergence(f'{name}f')
        # histohram
        self.__divergence(f'{name}h')
        # slow
        self.__divergence(f'{name}s')

        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[f'{name}f', f'{name}h', f'{name}s'])

    def __RSI(self, surname: str, period: int, h_range=100, l_range=-100, out_of_range=True, drop_values=True):
        name = f'RSI({surname})'
        self.bars_data[name] = ta.rsi(self.bars_data['close'], length=period)
        if out_of_range:
            self.__out_of_range(h_range, l_range, name)
        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[name])

    def __CCI(self, surname: str, period: int, h_range=100, l_range=-100, out_of_range=True, drop_values=True):
        name = f'CCI({surname})'
        self.bars_data[name] = ta.cci(self.bars_data['high'], self.bars_data['low'], self.bars_data['close'],
                                      length=period)
        self.__divergence(name)
        if out_of_range:
            self.__out_of_range(h_range, l_range, name)
        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[name])

    def __WPR(self, surname: str, period: int, h_range=-20, l_range=-80, out_of_range=True, drop_values=True):
        name = f'WPR({surname})'
        self.bars_data[name] = ta.willr(self.bars_data['high'], self.bars_data['low'], self.bars_data['close'],
                                        length=period)
        self.__divergence(name)
        if out_of_range:
            self.__out_of_range(h_range, l_range, name)
        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[name])

    def __EFI(self, surname: str, period: int, drop_values=True):
        name = f'EFI({surname})'
        self.bars_data[name] = ta.efi(self.bars_data['close'], self.bars_data['volume'], length=period)
        self.__divergence(name)
        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[name])

    def __CMF(self, surname: str, period: int, drop_values=True):
        name = f'CMF({surname})'
        self.bars_data[name] = ta.cmf(self.bars_data['high'], self.bars_data['low'], self.bars_data['close'],
                                      self.bars_data['volume'], length=period)
        self.__divergence(name)
        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[name])

    def __ROC(self, surname: str, period: int, drop_values=True):
        name = f'ROC({surname})'
        self.bars_data[name] = ta.roc(self.bars_data['close'], length=period)
        self.__divergence(name)
        if drop_values:
            self.bars_data = self.bars_data.drop(columns=[name])

    def __research_bars(self):
        if len(self.bars_data) >= 2 * self.per_f:
            self.__WPR('f', self.per_f, h_range=-20, l_range=-80, out_of_range=True, drop_values=True)
            self.__CCI('f', self.per_f, h_range=100, l_range=-100, out_of_range=True, drop_values=True)
            self.__RSI('f', self.per_f, h_range=70, l_range=30, out_of_range=True, drop_values=True)
            self.__EFI('f', self.per_f, drop_values=True)
            self.__ROC('f', self.per_f, drop_values=True)
            self.__CMF('f', self.per_f, drop_values=True)

        if len(self.bars_data) >= 2 * self.per:
            self.__WPR('m', self.per, h_range=-20, l_range=-80, out_of_range=True, drop_values=True)
            self.__CCI('m', self.per, h_range=100, l_range=-100, out_of_range=True, drop_values=True)
            self.__RSI('m', self.per, h_range=70, l_range=30, out_of_range=True, drop_values=True)
            self.__EFI('m', self.per, drop_values=True)
            self.__CMF('m', self.per, drop_values=True)
            self.__ROC('m', self.per, drop_values=True)

        if len(self.bars_data) >= 2 * self.per_s:
            # self.__MACD('close', self.per, self.per_s, self.per_f, drop_values=True)

            self.__CCI('s', self.per_s, h_range=150, l_range=-150, out_of_range=True, drop_values=True)
            self.__WPR('s', self.per_s, h_range=-20, l_range=-80, out_of_range=True, drop_values=True)
            self.__RSI('s', self.per_s, h_range=80, l_range=20, out_of_range=True, drop_values=True)
            self.__EFI('s', self.per_s, drop_values=True)
            self.__CMF('s', self.per_s, drop_values=True)
            self.__ROC('s', self.per_s, drop_values=True)

    def __results(self, drop_values=True):
        df = self.bars_data
        div_columns_f, ind_levels_f, out_of_range_f = [], [], []
        div_columns_m, ind_levels_m, out_of_range_m = [], [], []
        div_columns_s, ind_levels_s, out_of_range_s = [], [], []

        for column in df.columns:
            if '/div(' in column:
                if '(f)' in column:
                    div_columns_f.append(column)
                elif '(m)' in column:
                    div_columns_m.append(column)
                elif '(s)' in column:
                    div_columns_s.append(column)
                elif 'close)' in column:
                    div_columns_s.append(column)
            if '/level' in column:
                if '(f)' or 'f/level' in column:
                    ind_levels_f.append(column)
                elif '(m)' or 'h/level' in column:
                    ind_levels_m.append(column)
                elif '(s)' or 's/level' in column:
                    ind_levels_s.append(column)
            if '/out_of_range' in column:
                if '(f)' in column:
                    out_of_range_f.append(column)
                elif '(m)' in column:
                    out_of_range_m.append(column)
                elif '(s)' in column:
                    out_of_range_s.append(column)

        df['res_div(f)'] = df[div_columns_f].sum(axis=1) / len(div_columns_f) * 100
        df['res_out_of_range(f)'] = df[out_of_range_f].sum(axis=1) / len(out_of_range_f) * 100
        df['res_levels(f)'] = -df[ind_levels_f].sum(axis=1) / len(ind_levels_f) * 100

        df['res_div(m)'] = df[div_columns_m].sum(axis=1) / len(div_columns_m) * 100
        df['res_out_of_range(m)'] = df[out_of_range_m].sum(axis=1) / len(out_of_range_m) * 100
        df['res_levels(m)'] = -df[ind_levels_m].sum(axis=1) / len(ind_levels_m) * 100

        df['res_div(s)'] = df[div_columns_s].sum(axis=1) / len(div_columns_s) * 100
        df['res_out_of_range(s)'] = df[out_of_range_s].sum(axis=1) / (len(out_of_range_s) * 2) * 100
        df['res_levels(s)'] = -df[ind_levels_s].sum(axis=1) / len(ind_levels_s) * 100

        if drop_values:
            drop_list = div_columns_f + div_columns_s + div_columns_m + ind_levels_s + ind_levels_m + ind_levels_f + out_of_range_s + out_of_range_f + out_of_range_m
            df = df.drop(columns=drop_list)

    # @time_lag
    def get_results(self):
        self.__research_bars()
        self.__results()
        return self.bars_data



'''++++++++++++++++++++++++++++'''
url = "https://api-invest.tinkoff.ru/openapi"
token = "t.5Y7vhb8EHS4BZn1e5VwTQiFyZ0WiTALAG24U069TE8M0MiBAJ6BDtqmoPcuLBIsNNfh7qKWi_gWgKb3X8EglGw"

ticker = 'SBER'
timeframe = '1d'
count_bars = 10000

username, password, host, port = 'bot', 'bot', 'localhost', 3306
bars_def = get_bars(ticker=ticker, timeframe=timeframe, count_bars=count_bars)


# BarAnalise_bars = BarAnalise(bars_def.copy()).get_result()
# MoveAnalise_bars = MoveAnalise(bars_def.copy()).get_result()

ResearchResult = Research(default_bars_data=bars_def).get_results()

params_list =[
    "res_div(f)",
    "res_out_of_range(f)",
    "res_levels(f)",
    "res_div(m)",
    "res_out_of_range(m)",
    "res_levels(m)",
    "res_div(s)",
    "res_out_of_range(s)",
    "res_levels(s)"
]

ResearchResult['fullres'] = 0
for k in params_list:
    ResearchResult[k] = ResearchResult[k].fillna(0)
    ResearchResult['fullres'] += ResearchResult[k]

df_print(ResearchResult, count_rows=-10)
Graph(ResearchResult, ticker, timeframe, perc_par='fullres', par=False, hard=75, medium=50, lite=25, show=False)._basic_graph()



# MoveAnalise_bars
# IndicatorAnalise_bars = IndicatorAnalise(bars_def.copy()).get_result()
# IndicatorAnalise_bars
# save_to_db(MoveAnalise_bars, 'MoveAnalise_bars', username, password, host, port)
