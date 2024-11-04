

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