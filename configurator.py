from addition.api import *
from addition.other import *

class Config:
    def __init__(self, creds_path, local_start):
        self.creds_path = creds_path
        self.save_path_xslx = 'files/config/config.xlsx'
        self.save_path_json = 'files/config/config.json'
        self.creds = json_to_dict(self.creds_path)

        if not local_start:
            self.save_config()
        self.default_config = read_excel_file(self.save_path_xslx)

    def save_config(self):
        service_account_path = self.creds['config']['path']
        file_link = self.creds['config']['link']

        init = GoogleDriveAPI(service_account_path)
        fileid = init.get_file_id(file_link)
        init.download_file(real_file_id=fileid, file_path=self.save_path_xslx)
        return self.save_path_xslx

    def get_config(self):
        config = {
            'credential': self.creds,
            'loader': Loader(self.default_config).get_config()
        }
        dict_to_json(config, self.save_path_json)
        return config


class Loader:
    def __init__(self, default_config):
        self.default_config = default_config

    def filter_enabled(self):
        connection = self.default_config['connection']
        enable_connection = connection.loc[connection['enable'] == True]
        return enable_connection

    def update_data(self):
        timeframe = self.default_config['timeframe']
        timeframe = timeframe.map(lambda x: None if pd.isna(x) else x)
        return timeframe

    @staticmethod
    def get_data_loader(enable_connection, timeframe):
        config = []
        for i, s in enable_connection.iterrows():
            temp = dict(s)
            for el in ['enable', 'id']:
                temp.pop(el)
            enable_timeframe = (timeframe.loc[timeframe['Id_connection'] == s['id']])
            enable_timeframe.pop('Id_connection')
            enable_timeframe = enable_timeframe.to_dict('records')
            temp.update({'timeframe': enable_timeframe})
            config.append(temp)
        return config

    def get_config(self):
        enable_connection = self.filter_enabled()
        timeframe = self.update_data()
        config = self.get_data_loader(enable_connection, timeframe)
        return config

