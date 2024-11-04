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