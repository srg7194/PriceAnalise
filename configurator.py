from pprint import pprint
from addition.api import *
from addition.other import *


class ConfiguratorAnaliser:
    def __init__(self, creds_path, local_start=False):
        self.creds_path = creds_path
        self.save_path = 'files/config/config.xlsx'
        self.creds = json_to_dict(self.creds_path)

        if not local_start:
            self.save_config()
        self.default_config = read_excel_file(self.save_path)

    def save_config(self):
        service_account_path = self.creds['config']['path']
        file_link = self.creds['config']['link']

        init = GoogleDriveAPI(service_account_path)
        fileid = init.get_file_id(file_link)
        init.download_file(real_file_id=fileid, file_path=self.save_path)
        return self.save_path

    def get_data_loader(self):
        # Фильтры
        connection = self.default_config['Подключения']
        enable_connection = connection.loc[connection['enable'] == True]

        timeframe = self.default_config['Таймфреймы']
        timeframe = timeframe.applymap(lambda x: None if pd.isna(x) else x)

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
        config = {
            'credential': self.creds,
            'data_loader': self.get_data_loader()
        }
        return config


class ConfiguratorTelegram:
    def __init__(self, creds_path, local_start=False):
        self.creds_path = creds_path
        self.save_path = 'files/config/config.xlsx'
        self.sheet_names = ['Подключения', 'Таймфреймы', 'Эмулятор']
        self.creds = json_to_dict(self.creds_path)

        if not local_start:
            self.save_config()
        self.default_config = read_excel_file(self.save_path, self.sheet_names)

    def save_config(self):
        service_account_path = self.creds['config']['path']
        file_link = self.creds['config']['link']

        init = GoogleDriveAPI(service_account_path)
        fileid = init.get_file_id(file_link)
        init.download_file(real_file_id=fileid, file_path=self.save_path)
        return self.save_path

    def prepare_connection(self):
        return False

    def prepare_timeframe(self):
        return False

    def prepare_emulator(self):
        return False

    def get_config(self):
        config = {
            'credential': self.creds,
            'connection': self.prepare_connection(),
            'timeframe': self.prepare_timeframe(),
            'emulator': self.prepare_emulator()
        }
        return config