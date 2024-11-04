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
